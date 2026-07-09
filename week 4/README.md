# Week 4 — LeWorldModel (LeWM) on Push-T

From-scratch replication of **LeWM** (JEPA world model from pixels,
[repo](https://github.com/lucas-maes/le-wm), [paper](https://arxiv.org/abs/2603.19312))
on the **Push-T** task, plus decoded dream rollouts and a t-SNE of the latent
space.

The real workflow is **CUDA-only** (the paper uses H200s; the code hardcodes
`accelerator: gpu`, `precision: bf16`, `.to("cuda")`). This machine is an 8 GB
Apple‑M2 (MPS, no CUDA), so the runnable deliverable is a **turnkey Colab
notebook** targeting a free **T4 GPU**, with all glue/visualization code written
and the model-side logic validated locally against the real `jepa.py`.

## Contents

| Path | What |
|---|---|
| [`LeWM_PushT_Colab.ipynb`](LeWM_PushT_Colab.ipynb) | **Run this on Colab (T4).** End-to-end: install → download HDF5 → train from scratch → validate success rate → dream rollouts → t-SNE → zip artifacts. Self-contained. |
| [`le-wm/`](le-wm) | The upstream paper repo (model + train/eval), cloned for reference. |
| [`lewm_extras/`](lewm_extras) | The scripts the paper repo lacks: probe `decoder.py`, `train_decoder.py`, `dream_rollout.py`, `tsne_latents.py`, shared `lewm_common.py`. |
| `lewm_extras/_local_test.py` | Offline shape-validation of the latent pipeline against the real `jepa.py` (no GPU/data needed). |
| `build_notebook.py` | Regenerates the notebook (embeds `lewm_extras/*` as `%%writefile` cells). |

## How to run

1. Upload `LeWM_PushT_Colab.ipynb` to [Colab](https://colab.research.google.com), set **Runtime → T4 GPU**.
2. Run all cells top to bottom. Steps 2–3 (download + train) are the long ones.
3. The last cells display the plots and download `lewm_deliverables.zip`.

## 1. Training command & config

The notebook calls the paper repo's own `train.py` (real model + SIGReg
regularizer + next-embedding loss). From-scratch is enforced by
`encoder.pretrained=false` in `config/train/model/lewm.yaml`.

```bash
# from le-wm/, with $STABLEWM_HOME set and the Lance dataset in place
python train.py \
  data=pusht \
  img_size=112 \
  trainer.max_epochs=20 \
  trainer.precision=16-mixed \        # T4 is Turing: fp16, NOT bf16
  trainer.accelerator=gpu trainer.devices=1 \
  loader.batch_size=96 \
  num_workers=2 \
  output_model_name=pusht/lewm
```

Model / objective (paper defaults, unchanged):

| Component | Value |
|---|---|
| Encoder | ViT‑tiny, patch 14, **from scratch** (`pretrained=false`) |
| Predictor | AR transformer, depth 6, 16 heads, AdaLN‑zero conditioning on actions |
| Embed dim | 192 · history 3 · 1‑step prediction · frameskip 5 |
| Loss | `pred_loss` (next‑embedding MSE) + `0.09 · SIGReg` (isotropic‑Gaussian regularizer) — the paper's two‑term objective |
| Optim | AdamW, lr 5e‑5, wd 1e‑3, cosine + warmup, grad‑clip 1.0 |
| Params | ~15 M |

**Deviations from the paper config** (to fit a free single T4 session), all via
CLI overrides — the model/loss are untouched:

| Knob | Paper | Here | Why |
|---|---|---|---|
| `img_size` | 224 | 112 | ~4× less compute; ViT interpolates pos‑enc |
| `max_epochs` | 100 | 20 | fits one free session (~1.5–3 h) |
| `precision` | bf16 | 16‑mixed | T4 lacks bf16 |
| `batch_size` | 128 | 96 | T4 16 GB VRAM |

## 2. Validation results

Produced by notebook **step 4** (CEM planning in the real Push‑T env, 20 expert
start/goal pairs). Success = agent+block position error < 20 px **and** block
angle error < π/9. Paste your run's numbers here:

```
success_rate: ____   (metrics printed by eval.py)
eval episodes: 20
checkpoint:   pusht/lewm/weights_epoch_20.pt
```

Expectation: with the reduced budget above, from‑scratch success will be **below
the paper's full‑config numbers** — this budget demonstrates the full train →
plan → evaluate loop end‑to‑end, not a state‑of‑the‑art number. Raise
`img_size=224`, `max_epochs=100`, `batch_size=128` on an A100/H200 (bf16) with
the *same command* to approach the paper.

## 3. Rollout visualizations

Notebook **step 7** → `dream.png` (top row ground truth, bottom row decoded
dream) and `dream.gif`. The world model imagines future latents open-loop from 3
context frames under the real action sequence; a probe decoder (step 6) renders
them. The decoder is trained **after** and on **frozen** latents, so it never
affects the from-scratch world-model training.

## 4. t-SNE plot

Notebook **step 8** → `tsne.png`: LeWM latents of ~3000 frames projected to 2D,
colored by block angle / block‑x / agent‑x. Structure along these axes indicates
the latent space encodes physical state (the paper's probing result).

## 5. Replication notes

- **Data path.** HF ships `pusht_expert_train.h5.zst` (~13 GB). We decompress to
  `.h5` (used by `eval.py`'s `HDF5Dataset`) and convert a copy to a compact
  Lance table (~0.8 GB) that `train.py` reads for fast random access — the
  "download HDF5 from HuggingFace" step, faithful to the repo.
- **No decoder upstream.** JEPA never reconstructs pixels, so the repo has no
  decoder. `lewm_extras/decoder.py` + `train_decoder.py` add a small probe
  decoder purely for visualization — consistent with how the paper produces
  decoded dream rollouts.
- **Push‑T is Pymunk/pygame**, not MuJoCo, so eval only needs a headless
  framebuffer (`xvfb`); the `MUJOCO_GL=egl` line in `eval.py` is a harmless
  no‑op here.
- **Checkpoint path.** `output_model_name=pusht/lewm` makes `train.py` save to
  `$STABLEWM_HOME/checkpoints/pusht/lewm/weights_epoch_N.pt`, which `eval.py`
  loads via `policy=pusht/lewm/weights_epoch_20.pt` (pass the explicit epoch
  file — a bare folder is ambiguous when several epochs are saved).
- **Local validation.** `lewm_extras/_local_test.py` drives the real `jepa.py`
  with a stub encoder and confirms `encode`/`predict`/`rollout` shapes and the
  decoder/t‑SNE path (`predicted_emb` → `(B, 1, ctx+future+1, 192)`). The
  swm‑dependent glue (dataset, env eval) is written against the documented API
  and runs on Colab.
- **From‑scratch only.** No pretrained LeWM checkpoint is used anywhere; the
  probe decoder is also trained from scratch.
