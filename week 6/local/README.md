# Week 6/local — GRASP vs CEM vs GD, benchmarked on the local LeWM

Runs the three planners in **closed-loop MPC in the real gym-pusht env** against
the locally-trained LeWM (`../../week 4/local/lewm_local.pt`), on identical episode
seeds, measuring success rate, coverage, and wall-clock planning time. All ran
on this 8 GB M2.

## Files
- `planners.py` — compact **CEM**, **vanilla GD** (single-start Adam, clipped),
  and **GRASP** (the w6 planner) against the model's differentiable latent cost.
- `benchmark.py` — closed-loop Push-T benchmark; writes `pusht_benchmark.md`.

## Run
```bash
python benchmark.py --num-eval 12 --eval-budget 40 --horizon 8 --receding 4 \
    --planners cem,gd,grasp
```

## Results — real (12 episodes, budget 40, horizon 8)

| planner | success rate | mean best coverage | mean plan time (ms) | # replans | total wall (s) |
|---|---:|---:|---:|---:|---:|
| cem | 0.00 | 0.055 | **202** | 120 | 25.8 |
| gd (vanilla) | 0.00 | 0.062 | 1458 | 120 | 175.9 |
| grasp | 0.00 | 0.055 | 995 | 120 | 120.3 |

Baseline: random actions reach ~0.01 mean coverage, so all three planners *do*
push the block toward the goal (~0.06), but **none solve the task** (success
needs coverage > 0.95).

### What this shows (and doesn't)
- **Success is 0 for all planners** — the from-scratch laptop LeWM (week 4/local) is
  not accurate enough to plan a full Push-T solve. This is a *model-capacity*
  limit, not a planner one: the model's latent cost correlates only weakly with
  true coverage (r ≈ −0.28), so there is little signal for any optimizer to
  exploit, and the planners tie on coverage.
- **Planning latency differs a lot and is the interesting axis here:** CEM
  (batched, gradient-free) is cheapest at ~200 ms/plan; **vanilla GD is the
  *slowest* (~1.5 s)** because its single-start sequential backprop underuses the
  GPU; GRASP sits between (batched multi-start amortizes the gradient steps).

For the **planner comparison with a well-conditioned cost** — where GRASP's
quality/stability advantages actually show — see the offline benchmark
[`../synthetic_benchmark.py`](../synthetic_benchmark.py) (GRASP beats budget-matched
CEM 87.5% vs 82.8% and all GD variants, while high-lr GD diverges). That
isolates the optimizer from the toy model's weakness.

## Honest summary
The local w6 pipeline is real and end-to-end (planning in the LeWM latent space,
stepping the true env, measuring success + wall-clock), but the toy world model
caps absolute success at ~0. The **clean, decisive planner ranking is the
synthetic benchmark**; this env benchmark contributes the real wall-clock
latency comparison and confirms all planners beat the random-action baseline on
coverage.
