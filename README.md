# SOC: Foundations of World Models & Stochastic Planning

Coursework for a Summer of Code track on world models and planning. Each week is
a self-contained project with its own README, code, and results. Everything was
built and trained from scratch — no pretrained checkpoints.

> Hardware note: most of this ran on an **8 GB Apple M2 (MPS, no CUDA)**. Where a
> project needs a datacenter GPU, there's a turnkey Colab notebook; where it can
> run locally, there's a `local/` variant that actually did.

## Weeks

### [`week 1-2/`](week%201-2) — Vision Transformer on CIFAR-10
A from-scratch ViT encoder (patch embedding, `[CLS]` token, positional
embeddings, pre-norm transformer blocks) trained on CIFAR-10.
**Result: 73.6% test accuracy** (20 epochs, MPS). See [its README](week%201-2/README.md).

### [`week 4/`](week%204) — LeWorldModel (LeWM) on Push-T
Replication of **LeWM**, a JEPA world model from pixels (next-embedding loss +
SIGReg), on the Push-T manipulation task, with decoded dream rollouts and a
t-SNE of the latent space.
- `LeWM_PushT_Colab.ipynb` — turnkey full-scale run on a Colab T4.
- [`week 4/local/`](week%204/local) — a **compact LeWM actually trained locally**
  on MPS (~1.3 M params, 64px): trained checkpoint, loss/SIGReg curves,
  `dream.png/gif`, `tsne.png`.

### [`week 6/`](week%206) — GRASP planner: benchmark vs CEM & vanilla GD
**GRASP** (Gradient-based Randomized Adaptive Search Planner) on top of the
trained LeWM, benchmarked against CEM and vanilla gradient descent for success
rate and wall-clock planning time.
- `grasp_solver.py` — drop-in `stable_worldmodel` solver.
- `synthetic_benchmark.py` — offline optimizer comparison (**ran**: GRASP 87.5%
  vs budget-matched CEM 82.8%, GD diverges).
- [`week 6/local/`](week%206/local) — closed-loop Push-T benchmark against the
  local LeWM (**ran**: real coverage + wall-clock table).

## Repo layout & reproducing

```
week 1-2/   ViT on CIFAR-10        (get_data.py, vit.py, train.py)
week 4/     LeWM on Push-T         (le-wm/ upstream code, lewm_extras/, local/)
week 6/     GRASP planning         (grasp_solver.py, benchmark.py, local/)
```

Each folder's README has exact commands. The CIFAR-10 dataset and Python caches
are gitignored (regenerable via `week 1-2/get_data.py`). `week 4/le-wm/` is the
upstream [LeWorldModel](https://github.com/lucas-maes/le-wm) repo vendored for
reference (see its own README/LICENSE for attribution).

## Environment

```bash
# week 1-2
pip install torch torchvision
# week 4 / week 6 local
pip install gym-pusht "pymunk>=6.4,<7" matplotlib scikit-learn imageio
```
