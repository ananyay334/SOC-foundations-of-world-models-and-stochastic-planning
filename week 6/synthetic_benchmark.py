"""Offline benchmark of CEM vs vanilla GD vs GRASP on a rugged cost.

We can't run the real Push-T env locally (needs the trained LeWM + CUDA), so we
stress the *optimizers* on a controllable differentiable landscape that mimics
what makes world-model planning hard: a Rastrigin-style cost, which is smooth
and differentiable (GD can flow) but riddled with local minima (single-start GD
gets trapped). Each "env" is an independent planning problem; we report solution
quality, success rate, wall-clock, and stability across seeds.

The reference CEM/GD here mirror `stable_worldmodel`'s algorithms so the
comparison is apples-to-apples; on Colab, `benchmark.py` uses the library's own
`CEMSolver`/`GradientSolver` with the real LeWM. GRASP is the exact same
`grasp_solver.GRASPSolver` used everywhere.

Run:  python synthetic_benchmark.py
"""

import math
import time
import types

import numpy as np
import torch
import torch.nn as nn

from grasp_solver import GRASPSolver

torch.manual_seed(0)
DEVICE = "cpu"
DT = torch.float32


# --------------------------------------------------------------------------- #
#  Synthetic differentiable world-model cost (Rastrigin over the trajectory)   #
# --------------------------------------------------------------------------- #
class RastriginCostModel(nn.Module):
    """get_cost(info, cands) -> (bs, S). Global min where the whole action
    sequence equals a per-env target; many local minima around it.

    `amp` controls ruggedness: a dominant quadratic basin (gradients informative)
    with sinusoidal ripples (local minima that trap naive gradient descent) — a
    fair proxy for a world-model cost, unlike a maximally-adversarial landscape.
    Counts cost evaluations for budget-matched comparison."""

    def __init__(self, targets, amp=2.5, freq=0.7):
        super().__init__()
        self.register_buffer("targets", targets)          # (n_envs, H, A)
        self.amp, self.freq = amp, freq
        self._p = nn.Parameter(torch.zeros(1))            # for dtype detection
        self.n_evals = 0

    def get_cost(self, info, cands):
        # cands: (bs, S, H, A); targets sliced to this batch live in info["_tgt"]
        self.n_evals += cands.shape[0] * cands.shape[1]   # count candidate evals
        tgt = info["_tgt"]                                # (bs, S, H, A)
        d = cands - tgt
        quad = d.pow(2)
        ripple = self.amp * (1 - torch.cos(2 * math.pi * self.freq * d))
        return (quad + ripple).mean(dim=(2, 3))          # (bs, S)


def make_infos(targets, n_envs):
    # info dict whose leading dim is n_envs (so solvers read total_envs); the
    # per-env target is carried in "_tgt" and expanded by the solver.
    return {"_tgt": targets, "state": torch.zeros(n_envs, 1)}


# --------------------------------------------------------------------------- #
#  Reference CEM / GD (mirror stable_worldmodel; model= interface)            #
# --------------------------------------------------------------------------- #
class _Base:
    def configure(self, *, action_space, n_envs, config):
        self._as, self._n, self._cfg = action_space, n_envs, config
        self._ad = int(np.prod(action_space.shape[1:]))
        self.low, self.high = float(action_space.low.min()), float(action_space.high.max())

    @property
    def horizon(self): return self._cfg.horizon
    @property
    def action_dim(self): return self._ad * self._cfg.action_block
    def __call__(self, *a, **k): return self.solve(*a, **k)

    def _expand(self, info, S):
        out = {}
        for k, v in info.items():
            out[k] = v.unsqueeze(1).expand(v.shape[0], S, *v.shape[1:]) if torch.is_tensor(v) else v
        return out


class RefCEM(_Base):
    def __init__(self, model, num_samples=300, n_steps=30, topk=30, var_scale=1.0, seed=0, **_):
        self.model, self.num_samples, self.n_steps = model, num_samples, n_steps
        self.topk, self.var_scale = topk, var_scale
        self.gen = torch.Generator().manual_seed(seed)

    @torch.no_grad()
    def solve(self, info, init_action=None):
        t0 = time.time()
        E = next(iter(info.values())).shape[0]
        H, A, S = self.horizon, self.action_dim, self.num_samples
        mean = torch.zeros(E, H, A); var = self.var_scale * torch.ones(E, H, A)
        infos = self._expand(info, S)
        bidx = torch.arange(E)
        for _ in range(self.n_steps):
            c = torch.randn(E, S, H, A, generator=self.gen) * var.unsqueeze(1) + mean.unsqueeze(1)
            c[:, 0] = mean
            c.clamp_(self.low, self.high)
            costs = self.model.get_cost(infos, c)
            tv, ti = torch.topk(costs, self.topk, dim=1, largest=False)
            elites = c[bidx.unsqueeze(1), ti]
            mean, var = elites.mean(1), elites.std(1).clamp_min(1e-3)
        costs = self.model.get_cost(infos, mean.unsqueeze(1))[:, 0]
        return {"actions": mean, "cost": costs.tolist(), "solve_time": time.time() - t0}


class RefGD(_Base):
    def __init__(self, model, n_steps=30, num_samples=1, lr=0.1, var_scale=1.0,
                 grad_clip=None, seed=0, **_):
        self.model, self.n_steps, self.num_samples = model, n_steps, num_samples
        self.lr, self.var_scale, self.grad_clip = lr, var_scale, grad_clip
        self.gen = torch.Generator().manual_seed(seed)

    def solve(self, info, init_action=None):
        t0 = time.time()
        E = next(iter(info.values())).shape[0]
        H, A, S = self.horizon, self.action_dim, self.num_samples
        infos = self._expand(info, S)
        pop = torch.randn(E, S, H, A, generator=self.gen) * self.var_scale
        pop.clamp_(self.low, self.high)
        leaf = pop.clone().requires_grad_(True)
        opt = torch.optim.Adam([leaf], lr=self.lr)
        diverged = False
        for _ in range(self.n_steps):
            costs = self.model.get_cost(infos, leaf)
            opt.zero_grad(set_to_none=True)
            costs.sum().backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_([leaf], self.grad_clip)
            opt.step()
            with torch.no_grad():
                leaf.clamp_(self.low, self.high)
        with torch.no_grad():
            costs = self.model.get_cost(infos, leaf)
            cmin, cidx = costs.min(dim=1)
            best = leaf[torch.arange(E), cidx]
            if not torch.isfinite(cmin).all():
                diverged = True
        return {"actions": best, "cost": cmin.tolist(), "solve_time": time.time() - t0,
                "diverged": diverged}


class RefGD_SGD(_Base):
    """Deliberately unstable GD: plain SGD, high lr, NO clip / NO clamp — shows
    the exploding-gradient failure mode (actions blow up to non-finite cost)."""

    def __init__(self, model, n_steps=30, num_samples=1, lr=3.0, seed=0, **_):
        self.model, self.n_steps, self.num_samples, self.lr = model, n_steps, num_samples, lr
        self.gen = torch.Generator().manual_seed(seed)

    def solve(self, info, init_action=None):
        t0 = time.time()
        E = next(iter(info.values())).shape[0]
        H, A, S = self.horizon, self.action_dim, self.num_samples
        infos = self._expand(info, S)
        leaf = (torch.randn(E, S, H, A, generator=self.gen)).requires_grad_(True)
        opt = torch.optim.SGD([leaf], lr=self.lr)
        for _ in range(self.n_steps):
            costs = self.model.get_cost(infos, leaf)
            opt.zero_grad(set_to_none=True)
            torch.nan_to_num(costs, 1e9, 1e9, -1e9).sum().backward()
            opt.step()                                    # no clip, no clamp
        with torch.no_grad():
            costs = self.model.get_cost(infos, leaf)
            cmin, cidx = torch.nan_to_num(costs, float("inf")).min(dim=1)
            best = leaf[torch.arange(E), cidx]
        return {"actions": best, "cost": cmin.tolist(), "solve_time": time.time() - t0,
                "diverged": bool((~torch.isfinite(cmin)).any())}


# --------------------------------------------------------------------------- #
def run(planner, model, info):
    model.n_evals = 0
    out = planner.solve(info)
    costs = torch.tensor(out["cost"])
    return costs, out["solve_time"], model.n_evals, out.get("diverged", False)


def main():
    N_ENVS, H, A = 64, 5, 2          # 64 independent planning problems
    SUCCESS_THRESH = 0.5             # cost below this ~ found the global basin
    SEEDS = [0, 1, 2]

    cfg = types.SimpleNamespace(horizon=H, action_block=1, receding_horizon=H)
    bound = 4.0
    action_space = types.SimpleNamespace(
        shape=(N_ENVS, A), low=np.array(-bound, dtype=np.float32),
        high=np.array(bound, dtype=np.float32))

    # ~1000-eval budget shared by the budget-matched planners
    planners = [
        ("CEM (big, 9000 evals)", lambda m, s: RefCEM(m, num_samples=300, n_steps=30, topk=30, seed=s)),
        ("CEM (matched, ~1000)", lambda m, s: RefCEM(m, num_samples=64, n_steps=16, topk=8, seed=s)),
        ("GD single-start (no clip)", lambda m, s: RefGD(m, n_steps=30, num_samples=1, lr=0.15, grad_clip=None, seed=s)),
        ("GD SGD hi-lr (diverges)", lambda m, s: RefGD_SGD(m, n_steps=30, num_samples=1, lr=20.0, seed=s)),
        ("GRASP (~1000)", lambda m, s: GRASPSolver(m, num_restarts=32, rounds=3, inner_steps=10,
                                                    elite=8, lr=0.15, grad_clip=1.0, seed=s)),
    ]

    rows = []
    for name, factory in planners:
        best, succ, tsum, ev, div = [], [], [], [], 0
        for s in SEEDS:
            g = torch.Generator().manual_seed(100 + s)
            targets = (torch.rand(N_ENVS, H, A, generator=g) * 4 - 2)   # in [-2,2]
            model = RastriginCostModel(targets)
            info = make_infos(targets, N_ENVS)
            info["_tgt"] = targets   # solver will expand to (bs,S,H,A)
            planner = factory(model, s)
            planner.configure(action_space=action_space, n_envs=N_ENVS, config=cfg)
            costs, dt, nev, diverged = run(planner, model, info)
            sane = torch.isfinite(costs) & (costs < 1e6)   # "blowup" counts as divergence
            best.append(costs[sane].mean().item() if sane.any() else float("nan"))
            succ.append((costs < SUCCESS_THRESH).float().mean().item())
            tsum.append(dt); ev.append(nev)
            div += int(diverged or (~sane).float().mean().item() > 0.5)
        rows.append((name, np.nanmean(best), np.mean(succ) * 100, np.mean(tsum),
                     int(np.mean(ev)), div))

    def fmt_cost(c):
        return "blowup" if (not np.isfinite(c)) else f"{c:.3f}"

    hdr = f"{'planner':<28}{'cost':>9}{'success%':>10}{'time(s)':>10}{'evals':>9}{'diverged':>10}"
    print("\n" + hdr); print("-" * len(hdr))
    for name, c, sr, t, e, d in rows:
        print(f"{name:<28}{fmt_cost(c):>9}{sr:>10.1f}{t:>10.3f}{e:>9d}{d:>8d}/3")
    print(f"\n64 problems x 3 seeds; success = global basin (cost < {SUCCESS_THRESH}).")
    print("Lower cost / higher success / lower time better. `evals` = cost calls "
          "(compute budget). `diverged` = seeds with non-finite cost.")

    with open("synthetic_results.md", "w") as f:
        f.write("| planner | mean cost | success % | wall-clock (s) | cost-evals | diverged |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for name, c, sr, t, e, d in rows:
            f.write(f"| {name} | {fmt_cost(c)} | {sr:.1f} | {t:.3f} | {e} | {d}/3 |\n")
    print("wrote synthetic_results.md")


if __name__ == "__main__":
    main()
