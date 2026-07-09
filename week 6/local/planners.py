"""Compact CEM / vanilla-GD / GRASP planners for the local LeWM.

All three optimize the same differentiable latent cost
``model.get_cost(info, action_candidates)`` where
``action_candidates`` has shape (B, S, T, A) and the cost is (B, S). They return
the best action sequence per env, shape (B, T, A). B is the number of envs
planned in parallel (here 1 per call; the env benchmark plans each env).

GRASP is the same algorithm as w6/grasp_solver.py, distilled to this minimal
interface (no gym Space / Hydra config needed).
"""

import time

import torch


class _Planner:
    def __init__(self, model, horizon, action_dim, low, high, device, seed=0):
        self.model, self.H, self.A = model, horizon, action_dim
        self.low, self.high, self.device = low, high, device
        self.center = 0.5 * (low + high)          # actions are absolute positions
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.total_time, self.n_plans = 0.0, 0

    def _base(self, B, init):
        """Default action prior = arena center (not zeros: 0 is a corner)."""
        if init is not None:
            return init
        return torch.full((B, self.H, self.A), self.center, device=self.device)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.device)

    def plan(self, info, init=None):
        t0 = time.time()
        out = self._solve(info, init)
        if self.device == "mps":
            torch.mps.synchronize()
        self.total_time += time.time() - t0
        self.n_plans += info["pixels"].shape[0]
        return out

    @property
    def mean_plan_ms(self):
        return 1e3 * self.total_time / max(self.n_plans, 1)


class CEM(_Planner):
    def __init__(self, *a, num_samples=200, n_steps=10, topk=20, var_scale=0.3, **k):
        super().__init__(*a, **k)
        self.S, self.n_steps, self.topk, self.var0 = num_samples, n_steps, topk, var_scale

    @torch.no_grad()
    def _solve(self, info, init):
        B = info["pixels"].shape[0]
        mean = self._base(B, init).clone()
        var = self.var0 * torch.ones(B, self.H, self.A, device=self.device)
        bidx = torch.arange(B, device=self.device)
        for _ in range(self.n_steps):
            c = mean.unsqueeze(1) + var.unsqueeze(1) * self._randn(B, self.S, self.H, self.A)
            c[:, 0] = mean
            c.clamp_(self.low, self.high)
            costs = self.model.get_cost(info, c)                       # (B,S)
            tv, ti = torch.topk(costs, self.topk, dim=1, largest=False)
            elites = c[bidx.unsqueeze(1), ti]
            mean, var = elites.mean(1), elites.std(1).clamp_min(1e-3)
        return mean


class GD(_Planner):
    """Vanilla gradient descent: single start, Adam, gradient clipping."""

    def __init__(self, *a, num_starts=1, n_steps=40, lr=0.05, var_scale=0.3, grad_clip=1.0, **k):
        super().__init__(*a, **k)
        self.S, self.n_steps, self.lr, self.var0, self.clip = num_starts, n_steps, lr, var_scale, grad_clip

    def _solve(self, info, init):
        B = info["pixels"].shape[0]
        base = self._base(B, init)
        pop = base.unsqueeze(1) + self.var0 * self._randn(B, self.S, self.H, self.A)
        pop.clamp_(self.low, self.high)
        leaf = pop.clone().detach().requires_grad_(True)
        opt = torch.optim.Adam([leaf], lr=self.lr)
        for _ in range(self.n_steps):
            cost = self.model.get_cost(info, leaf).sum()
            opt.zero_grad(); cost.backward()
            if self.clip:
                torch.nn.utils.clip_grad_norm_([leaf], self.clip)
            opt.step()
            with torch.no_grad():
                leaf.clamp_(self.low, self.high)
        with torch.no_grad():
            costs = self.model.get_cost(info, leaf)
            idx = costs.argmin(1)
        return leaf.detach()[torch.arange(B), idx]


class GRASP(_Planner):
    """Gradient-based Randomized Adaptive Search Planner (see w6/grasp_solver.py)."""

    def __init__(self, *a, num_restarts=24, rounds=3, inner_steps=8, elite=6,
                 lr=0.05, var_scale=0.3, var_floor=1e-2, grad_clip=1.0, **k):
        super().__init__(*a, **k)
        self.R, self.rounds, self.inner = num_restarts, rounds, inner_steps
        self.elite, self.lr = min(elite, num_restarts), lr
        self.var0, self.var_floor, self.clip = var_scale, var_floor, grad_clip

    def _solve(self, info, init):
        B = info["pixels"].shape[0]
        base = self._base(B, init)
        bidx = torch.arange(B, device=self.device)
        pop = base.unsqueeze(1) + self.var0 * self._randn(B, self.R, self.H, self.A)
        pop[:, 0] = base
        pop.clamp_(self.low, self.high)
        best_act = pop[:, 0].clone()
        best_cost = torch.full((B,), float("inf"), device=self.device)

        for _ in range(self.rounds):
            leaf = pop.clone().detach().requires_grad_(True)      # local search
            opt = torch.optim.Adam([leaf], lr=self.lr)
            for _ in range(self.inner):
                cost = self.model.get_cost(info, leaf).sum()
                opt.zero_grad(); cost.backward()
                if leaf.grad is not None:
                    leaf.grad.nan_to_num_(0., 0., 0.)
                    if self.clip:
                        torch.nn.utils.clip_grad_norm_([leaf], self.clip)
                opt.step()
                with torch.no_grad():
                    leaf.clamp_(self.low, self.high)
            pop = leaf.detach()
            with torch.no_grad():
                costs = self.model.get_cost(info, pop)            # (B,R)
                cmin, cidx = costs.min(1)
                imp = cmin < best_cost
                best_cost = torch.where(imp, cmin, best_cost)
                best_act = torch.where(imp.view(B, 1, 1), pop[bidx, cidx], best_act)
                # adaptive restart: refit to elites, keep incumbent
                tv, ti = torch.topk(costs, self.elite, dim=1, largest=False)
                elites = pop[bidx.unsqueeze(1), ti]
                mean, std = elites.mean(1), elites.std(1).clamp_min(self.var_floor)
                pop = mean.unsqueeze(1) + std.unsqueeze(1) * self._randn(B, self.R, self.H, self.A)
                pop[:, 0] = best_act
                pop.clamp_(self.low, self.high)
        return best_act


def make_planner(name, model, horizon, action_dim, low, high, device, seed=0):
    kw = dict(model=model, horizon=horizon, action_dim=action_dim, low=low, high=high,
              device=device, seed=seed)
    return {"cem": CEM, "gd": GD, "grasp": GRASP}[name](**kw)
