"""Benchmark GRASP vs CEM vs vanilla gradient descent on Push-T with a trained
LeWM. Runs closed-loop MPC in the real env for each planner on the SAME
start/goal pairs, and reports success rate + planning wall-clock time.

Runs on Colab / a CUDA box (needs the trained LeWM checkpoint + the Push-T env).
Adapted from le-wm/eval.py; the planners share the `swm.solver` interface, so
GRASP (grasp_solver.GRASPSolver) drops in beside the library's CEMSolver /
GradientSolver with no other changes.

Example (from le-wm/, with grasp_solver.py + this file on PYTHONPATH):
    xvfb-run -a python benchmark.py \
        --model pusht/lewm/weights_epoch_20.pt --img-size 112 --num-eval 20
"""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")   # harmless for pymunk PushT

import numpy as np
import torch
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms

import stable_pretraining as spt
import stable_worldmodel as swm

from grasp_solver import GRASPSolver


# --------------------------------------------------------------------------- #
class TimedSolver:
    """Wraps a solver to accumulate wall-clock time and call count of solve().
    Satisfies the runtime-checkable Solver protocol by delegation."""

    def __init__(self, solver, name):
        self.solver = solver
        self.name = name
        self.total_time = 0.0
        self.n_calls = 0
        self.n_plans = 0

    def configure(self, **kw):
        return self.solver.configure(**kw)

    @property
    def action_dim(self): return self.solver.action_dim
    @property
    def n_envs(self): return self.solver.n_envs
    @property
    def horizon(self): return self.solver.horizon

    def solve(self, info_dict, init_action=None):
        n = len(next(iter(info_dict.values())))
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        out = self.solver.solve(info_dict, init_action=init_action)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.total_time += time.time() - t0
        self.n_calls += 1
        self.n_plans += n           # number of (env) plans produced
        return out

    __call__ = solve

    @property
    def mean_plan_ms(self):
        return 1e3 * self.total_time / max(self.n_plans, 1)


def img_transform(img_size):
    return transforms.Compose([
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(**spt.data.dataset_stats.ImageNet),
        transforms.Resize(size=img_size),
    ])


def episodes_length(dataset, episodes, col):
    ep = dataset.get_col_data(col)
    step = dataset.get_col_data("step_idx")
    return np.array([np.max(step[ep == e]) + 1 for e in episodes])


def build_solver(name, model, device, seed):
    """Construct each planner with a comparable-ish budget. Wall-clock is what
    we actually report, so exact budgets need not match."""
    if name == "cem":
        return swm.solver.CEMSolver(model, num_samples=300, n_steps=30, topk=30,
                                    var_scale=1.0, device=device, seed=seed)
    if name == "gd":   # vanilla gradient descent: single start, Adam, clipped
        return swm.solver.GradientSolver(
            model, n_steps=30, num_samples=1, var_scale=1.0, device=device, seed=seed,
            optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 0.1}, grad_clip=1.0)
    if name == "gd_multi":  # stronger GD reference: multi-start
        return swm.solver.GradientSolver(
            model, n_steps=30, num_samples=32, var_scale=1.0, device=device, seed=seed,
            optimizer_cls=torch.optim.Adam, optimizer_kwargs={"lr": 0.1}, grad_clip=1.0)
    if name == "grasp":
        return GRASPSolver(model, num_restarts=32, rounds=3, inner_steps=10, elite=8,
                           lr=0.1, var_scale=1.0, grad_clip=1.0, device=device, seed=seed)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint under $STABLEWM_HOME/checkpoints")
    ap.add_argument("--dataset", default="pusht_expert_train")
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--num-eval", type=int, default=20)
    ap.add_argument("--goal-offset", type=int, default=25)
    ap.add_argument("--eval-budget", type=int, default=50)
    ap.add_argument("--planners", default="cem,gd,grasp",
                    help="comma list: cem,gd,gd_multi,grasp")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="grasp_benchmark.md")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    # -- load model once (shared across planners) --
    model = swm.wm.utils.load_pretrained(args.model).to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True

    # -- dataset + fixed start/goal episodes (same for every planner) --
    dataset = swm.data.HDF5Dataset(args.dataset,
                                   keys_to_cache=["action", "proprio", "state"],
                                   cache_dir=Path(swm.data.utils.get_cache_dir()))
    col = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices = np.unique(dataset.get_col_data(col))

    process = {}
    for c in ["action", "proprio", "state"]:
        if c not in dataset.column_names:
            continue
        sc = preprocessing.StandardScaler()
        d = dataset.get_col_data(c)
        sc.fit(d[~np.isnan(d).any(axis=1)])
        process[c] = sc
        if c != "action":
            process[f"goal_{c}"] = sc

    ep_len = episodes_length(dataset, ep_indices, col)
    max_start = {e: ep_len[i] - args.goal_offset - 1 for i, e in enumerate(ep_indices)}
    per_row = np.array([max_start[e] for e in dataset.get_col_data(col)])
    valid = np.nonzero(dataset.get_col_data("step_idx") <= per_row)[0]
    rng = np.random.default_rng(args.seed)
    pick = np.sort(valid[rng.choice(len(valid) - 1, size=args.num_eval, replace=False)])
    eval_eps = dataset.get_row_data(pick)[col].tolist()
    eval_start = dataset.get_row_data(pick)["step_idx"].tolist()

    tf = {"pixels": img_transform(args.img_size), "goal": img_transform(args.img_size)}
    plan_cfg = swm.PlanConfig(horizon=5, receding_horizon=5, action_block=5)

    callables = [
        {"method": "_set_state", "args": {"state": {"value": "state"}}},
        {"method": "_set_goal_state", "args": {"goal_state": {"value": "goal_state"}}},
    ]

    rows = []
    for name in args.planners.split(","):
        name = name.strip()
        print(f"\n===== planner: {name} =====")
        world = swm.World(env_name="swm/PushT-v1", num_envs=args.num_eval,
                          max_episode_steps=2 * args.eval_budget, image_shape=(224, 224))
        solver = TimedSolver(build_solver(name, model, device, args.seed), name)
        policy = swm.policy.WorldModelPolicy(solver=solver, config=plan_cfg,
                                             process=process, transform=tf)
        world.set_policy(policy)

        t0 = time.time()
        metrics = world.evaluate(
            dataset=dataset, start_steps=eval_start, goal_offset=args.goal_offset,
            eval_budget=args.eval_budget, episodes_idx=eval_eps, callables=callables)
        wall = time.time() - t0

        sr = metrics.get("success_rate", metrics.get("success", float("nan")))
        rows.append((name, sr, solver.mean_plan_ms, solver.n_calls, wall))
        print(f"  success={sr}  mean_plan={solver.mean_plan_ms:.1f} ms  total={wall:.1f}s")

    # -- comparison table --
    print(f"\n{'planner':<10}{'success':>10}{'plan(ms)':>12}{'#plans':>9}{'total(s)':>10}")
    print("-" * 51)
    for n, sr, ms, nc, wall in rows:
        srv = f"{sr:.3f}" if isinstance(sr, (int, float)) else str(sr)
        print(f"{n:<10}{srv:>10}{ms:>12.1f}{nc:>9}{wall:>10.1f}")

    with open(args.out, "w") as f:
        f.write(f"# GRASP vs CEM vs GD on Push-T (LeWM `{args.model}`, "
                f"{args.num_eval} episodes, img={args.img_size})\n\n")
        f.write("| planner | success rate | mean plan time (ms) | # replans | total wall (s) |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for n, sr, ms, nc, wall in rows:
            srv = f"{sr:.3f}" if isinstance(sr, (int, float)) else str(sr)
            f.write(f"| {n} | {srv} | {ms:.1f} | {nc} | {wall:.1f} |\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
