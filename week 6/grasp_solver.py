"""GRASP — Gradient-based Randomized Adaptive Search Planner.

A model-based planner for LeWM (or any `Costable` world model exposing
``get_cost(info_dict, action_candidates) -> (batch, num_samples)`` that is
differentiable w.r.t. the actions).

Motivation
----------
The two standard planners have complementary weaknesses on the rugged cost
landscape induced by an autoregressive world-model rollout:

* **CEM** (sampling): global but gradient-free — needs many samples/iterations
  to refine precise actions, so it is slow (wall-clock heavy) and its elite
  distribution can collapse prematurely.
* **Vanilla gradient descent**: fast local refinement, but a single (or few)
  start gets trapped in local minima, is sensitive to initialization, and can
  diverge (exploding gradients through a long rollout).

GRASP is the classic *Greedy Randomized Adaptive Search Procedure* specialized
to trajectory optimization: it interleaves the two.

Algorithm (per replan, per env, all vectorized)
------------------------------------------------
1. **Randomized adaptive construction** — sample a population of candidate
   action sequences around the (warm-started) mean; keep the incumbent as
   sample 0 (greedy elitism).
2. **Local search** — refine the whole population in parallel with `inner_steps`
   of Adam through the differentiable cost, with gradient clipping + action
   clamping for stability.
3. **Adaptive restart with elite memory** — refit a Gaussian to the top-`elite`
   candidates (CEM-style), keep the best-so-far, and resample the population for
   the next round. Repeat for `rounds`.
4. Return the single lowest-cost action sequence found.

This drops into `stable_worldmodel` exactly like `CEMSolver` / `GradientSolver`
(same `model=` constructor, `configure()`, and `solve()` contract), so it can be
used by `WorldModelPolicy` and le-wm's `eval.py` with no other changes.
"""

import time
from typing import Any

import numpy as np
import torch

try:  # use the library's warm-start helper when available
    from stable_worldmodel.solver.utils import prepare_init_action
except Exception:  # offline / unit-test fallback (matches the swm semantics)
    def prepare_init_action(model, info_dict, init_action, horizon, n_envs, action_dim):
        n_prev = init_action.shape[1] if init_action is not None else 0
        remaining = horizon - n_prev
        if remaining <= 0:
            return init_action
        device = init_action.device if init_action is not None else "cpu"
        tail = torch.zeros(n_envs, remaining, action_dim, device=device)
        return tail if init_action is None else torch.cat([init_action, tail], dim=1)


class GRASPSolver:
    """Gradient-based Randomized Adaptive Search Planner.

    Args:
        model: world model implementing the Costable protocol (``get_cost``).
        num_restarts: population size per env (parallel candidate sequences).
        rounds: number of adaptive-restart rounds (outer loop).
        inner_steps: gradient (Adam) steps of local search per round.
        elite: number of elites kept to refit the sampling distribution.
        lr: Adam learning rate for the local search.
        var_scale: initial std of the construction sampling.
        var_floor: minimum elite std (prevents premature collapse).
        grad_clip: max grad-norm for the action tensor (stability).
        action_noise: optional exploration noise added after each Adam step.
        batch_size: envs processed at once (None = all).
        device, seed: as usual.
    """

    def __init__(
        self,
        model,
        num_restarts: int = 32,
        rounds: int = 3,
        inner_steps: int = 10,
        elite: int = 8,
        lr: float = 0.1,
        var_scale: float = 1.0,
        var_floor: float = 1e-2,
        grad_clip: float | None = 1.0,
        action_noise: float = 0.0,
        batch_size: int | None = None,
        device: str | torch.device = "cpu",
        seed: int = 1234,
        callbacks: list | None = None,
    ) -> None:
        self.model = model
        self.num_restarts = num_restarts
        self.rounds = rounds
        self.inner_steps = inner_steps
        self.elite = min(elite, num_restarts)
        self.lr = lr
        self.var_scale = var_scale
        self.var_floor = var_floor
        self.grad_clip = grad_clip
        self.action_noise = action_noise
        self.batch_size = batch_size
        self.device = device
        self.torch_gen = torch.Generator(device=device).manual_seed(seed)
        self.callbacks = list(callbacks) if callbacks else []
        try:
            self._dtype = next(model.parameters()).dtype
        except (AttributeError, StopIteration):
            self._dtype = torch.float32

    # -- Solver protocol -----------------------------------------------------
    def configure(self, *, action_space, n_envs: int, config: Any) -> None:
        self._action_space = action_space
        self._n_envs = n_envs
        self._config = config
        self._action_dim = int(np.prod(action_space.shape[1:]))
        # per-dim action bounds (flattened, tiled over the action_block)
        low = np.broadcast_to(action_space.low, action_space.shape)
        high = np.broadcast_to(action_space.high, action_space.shape)
        self._low = float(np.min(low))
        self._high = float(np.max(high))
        self._configured = True

    @property
    def n_envs(self) -> int:
        return self._n_envs

    @property
    def action_dim(self) -> int:
        return self._action_dim * self._config.action_block

    @property
    def horizon(self) -> int:
        return self._config.horizon

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def __call__(self, *args, **kwargs) -> dict:
        return self.solve(*args, **kwargs)

    # -- helpers -------------------------------------------------------------
    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.torch_gen, device=self.device, dtype=self.dtype)

    def _expand_infos(self, info_dict, start, end, pop):
        """Expand a batch slice of the info dict to (bs, pop, ...)."""
        out = {}
        for k, v in info_dict.items():
            vb = v[start:end]
            if torch.is_tensor(v):
                dt = self.dtype if vb.is_floating_point() else None
                vb = vb.to(device=self.device, dtype=dt).unsqueeze(1).expand(end - start, pop, *vb.shape[1:])
            elif isinstance(v, np.ndarray):
                vb = np.repeat(vb[:, None, ...], pop, axis=1)
            out[k] = vb
        return out

    @torch.no_grad()
    def _cost(self, infos, cands):
        return self.model.get_cost(infos, cands)

    # -- main ----------------------------------------------------------------
    def solve(self, info_dict: dict, init_action: torch.Tensor | None = None) -> dict:
        t0 = time.time()
        total_envs = len(next(iter(info_dict.values())))
        R, H, A = self.num_restarts, self.horizon, self.action_dim

        with torch.no_grad():
            init = prepare_init_action(self.model, info_dict, init_action, H, total_envs, A)
            init = init.to(self.device, self.dtype)

        bs_cfg = self.batch_size or total_envs
        best_actions, best_costs, cost_curves = [], [], []

        for s in range(0, total_envs, bs_cfg):
            e = min(s + bs_cfg, total_envs)
            bs = e - s
            infos = self._expand_infos(info_dict, s, e, R)
            batch_idx = torch.arange(bs, device=self.device)

            mean = init[s:e]                                   # (bs, H, A)
            # 1. randomized adaptive construction
            pop = mean.unsqueeze(1) + self.var_scale * self._randn(bs, R, H, A)
            pop[:, 0] = mean                                  # keep incumbent
            pop.clamp_(self._low, self._high)

            best_act = pop[:, 0].clone()
            best_cost = torch.full((bs,), float("inf"), device=self.device, dtype=self.dtype)
            curve = []

            for _ in range(self.rounds):
                # 2. local search: Adam through the differentiable cost
                leaf = pop.clone().detach().requires_grad_(True)
                opt = torch.optim.Adam([leaf], lr=self.lr)
                for _ in range(self.inner_steps):
                    costs = self.model.get_cost(infos, leaf)        # (bs, R)
                    loss = costs.sum()
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    # stability: drop non-finite grads, clip, step, clamp
                    g = leaf.grad
                    if g is not None:
                        g.nan_to_num_(0.0, 0.0, 0.0)
                        if self.grad_clip is not None:
                            torch.nn.utils.clip_grad_norm_([leaf], self.grad_clip)
                    opt.step()
                    with torch.no_grad():
                        if self.action_noise > 0:
                            leaf += self.action_noise * self._randn(bs, R, H, A)
                        leaf.clamp_(self._low, self._high)
                    curve.append(costs.detach().min().item())
                pop = leaf.detach()

                # evaluate refined population
                costs = self._cost(infos, pop)                      # (bs, R)
                cmin, cidx = costs.min(dim=1)
                improved = cmin < best_cost
                best_cost = torch.where(improved, cmin, best_cost)
                cand_best = pop[batch_idx, cidx]
                best_act = torch.where(improved.view(bs, 1, 1), cand_best, best_act)

                # 3. adaptive restart: refit Gaussian to elites, resample
                topv, topi = torch.topk(costs, k=self.elite, dim=1, largest=False)
                elites = pop[batch_idx.unsqueeze(1), topi]          # (bs, elite, H, A)
                mean = elites.mean(dim=1)
                std = elites.std(dim=1).clamp_min(self.var_floor)
                pop = mean.unsqueeze(1) + std.unsqueeze(1) * self._randn(bs, R, H, A)
                pop[:, 0] = best_act                                # elitism
                pop.clamp_(self._low, self._high)

            best_actions.append(best_act.detach().cpu())
            best_costs.extend(best_cost.detach().cpu().tolist())
            cost_curves.append(curve)

        outputs = {
            "actions": torch.cat(best_actions, dim=0),
            "cost": best_costs,
            "cost_curve": cost_curves,
        }
        outputs["solve_time"] = time.time() - t0
        return outputs
