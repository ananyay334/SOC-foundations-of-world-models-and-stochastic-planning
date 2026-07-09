"""Closed-loop Push-T benchmark: GRASP vs CEM vs vanilla GD on the local LeWM.

For each planner we run receding-horizon MPC in the real gym-pusht env on the
SAME episode seeds, planning in the LeWM latent space against a fixed goal
image. Reports success rate, best coverage, and mean wall-clock planning time.

    python benchmark.py --model "../../week 4/local/lewm_local.pt" \
        --data "../../week 4/local/pusht_data.npz" --num-eval 12 --planners cem,gd,grasp
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week 4" / "local"))
from lewm_local import LeWMLocal
from planners import make_planner

ACT_MAX = 512.0


def get_device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path, device):
    ckpt = torch.load(path, map_location="cpu")
    c = ckpt["cfg"]
    m = LeWMLocal(action_dim=c["action_dim"], dim=c["dim"], history_size=c["history"])
    m.load_state_dict(ckpt["state_dict"])
    return m.to(device).eval().requires_grad_(False)


def to_chw(img, device):
    """(H,W,3) uint8 -> (1,3,H,W) float [0,1] tensor."""
    t = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t.to(device)


def run_planner(name, model, goal_img, device, args):
    env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos",
                   render_mode="rgb_array", observation_width=64, observation_height=64)
    goal = to_chw(goal_img, device)
    planner = make_planner(name, model, args.horizon, 2, 0.0, 1.0, device, seed=args.seed)

    successes, best_covs = 0, []
    for ep in range(args.num_eval):
        obs, info = env.reset(seed=1000 + ep)
        init = None
        best_cov = 0.0
        success = False
        buffer = []
        for step in range(args.eval_budget):
            if not buffer:                                     # replan
                pix = to_chw(obs["pixels"], device)
                plan = planner.plan({"pixels": pix, "goal": goal}, init=init)  # (1,H,2) norm
                plan = plan[0].clamp(0, 1)
                exec_n = min(args.receding, args.horizon)
                buffer = list(plan[:exec_n])
                # warm-start next replan with the unused tail
                rest = plan[exec_n:]
                init = torch.cat([rest, torch.zeros(args.horizon - rest.shape[0], 2, device=device)]
                                 ).unsqueeze(0) if rest.shape[0] else None
            a_norm = buffer.pop(0)
            a = (a_norm.cpu().numpy() * ACT_MAX).astype(np.float32)
            obs, r, term, trunc, info = env.step(a)
            best_cov = max(best_cov, float(info.get("coverage", 0.0)))
            if info.get("is_success", False):
                success = True
                break
            if term or trunc:
                break
        successes += int(success)
        best_covs.append(best_cov)
    env.close()
    return {
        "planner": name, "success_rate": successes / args.num_eval,
        "mean_best_coverage": float(np.mean(best_covs)),
        "mean_plan_ms": planner.mean_plan_ms, "n_plans": planner.n_plans,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../../week 4/local/lewm_local.pt")
    ap.add_argument("--data", default="../../week 4/local/pusht_data.npz")
    ap.add_argument("--num-eval", type=int, default=12)
    ap.add_argument("--eval-budget", type=int, default=40)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--receding", type=int, default=4)
    ap.add_argument("--planners", default="cem,gd,grasp")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="pusht_benchmark.md")
    args = ap.parse_args()

    device = get_device()
    print(f"device={device}")
    model = load_model(args.model, device)
    goal_img = np.load(args.data)["goal_image"]

    rows = []
    for name in args.planners.split(","):
        name = name.strip()
        print(f"\n===== {name} =====")
        t0 = time.time()
        res = run_planner(name, model, goal_img, device, args)
        res["wall_s"] = time.time() - t0
        rows.append(res)
        print(f"  success={res['success_rate']:.2f}  best_cov={res['mean_best_coverage']:.3f}  "
              f"plan={res['mean_plan_ms']:.1f}ms  total={res['wall_s']:.1f}s")

    hdr = f"{'planner':<8}{'success':>9}{'best_cov':>10}{'plan(ms)':>11}{'#plans':>8}{'total(s)':>10}"
    print("\n" + hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['planner']:<8}{r['success_rate']:>9.2f}{r['mean_best_coverage']:>10.3f}"
              f"{r['mean_plan_ms']:>11.1f}{r['n_plans']:>8}{r['wall_s']:>10.1f}")

    with open(args.out, "w") as f:
        f.write(f"# Push-T planner benchmark (local LeWM, {args.num_eval} episodes, "
                f"budget {args.eval_budget}, horizon {args.horizon})\n\n")
        f.write("| planner | success rate | mean best coverage | mean plan time (ms) | # replans | total wall (s) |\n")
        f.write("|---|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(f"| {r['planner']} | {r['success_rate']:.2f} | {r['mean_best_coverage']:.3f} "
                    f"| {r['mean_plan_ms']:.1f} | {r['n_plans']} | {r['wall_s']:.1f} |\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
