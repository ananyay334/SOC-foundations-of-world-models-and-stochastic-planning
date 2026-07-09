# Week 4/local — LeWM trained locally on an 8 GB M2 (no Colab, no CUDA)

A compact **from-scratch LeWM** (JEPA world model from pixels) trained end-to-end
on this laptop, plus the w4 deliverables (decoded dream rollouts + latent t-SNE).
Everything here actually ran on MPS.

## What it is

Faithful to the LeWM recipe — next-embedding prediction loss **+ SIGReg**
(isotropic-Gaussian regularizer, the paper's core idea), an autoregressive
latent predictor, and **no decoder** — but sized for a laptop: a small conv
encoder at **64×64** (instead of a 224px ViT), latent dim 128, ~1.3 M params.
Reuses the paper's `SIGReg`/`ARPredictor` from `../le-wm/module.py`.

## Pipeline (all ran locally)

```bash
pip install gym-pusht "pymunk>=6.4,<7"       # Push-T env (pymunk 6.x!)
python collect_data.py --episodes 200 --steps 90   # -> pusht_data.npz (18k transitions)
python train.py --epochs 14                         # -> lewm_local.pt + loss_curve.png  (~10 min MPS)
python viz.py                                        # -> dream.png/gif + tsne.png
```

Data = a mix of a **scripted goal-pusher** (55%), push-toward-block (25%), and
random (20%) actions, so the model sees near-goal states.

## Results (real, from the runs above)

**Training** (14 epochs, ~43 s/epoch on MPS):

| | epoch 1 | epoch 14 |
|---|---:|---:|
| total loss | 0.540 | **0.316** |
| pred MSE | 0.253 | 0.208 |
| SIGReg | 3.19 | **1.20** |

The falling SIGReg is the key LeWM signal — the latent distribution is becoming
the target isotropic Gaussian without collapse. See `loss_curve.png`.

**Artifacts:**
- `dream.png` / `dream.gif` — decoded dream rollouts. Top row = ground truth,
  bottom = the model's imagined future latents (decoded by a probe decoder
  trained afterward on frozen latents). The imagined block tracks the true
  block's motion/rotation; blur is expected from decoding a 128-d global latent.
- `tsne.png` — t-SNE of latents colored by block angle / block-x / coverage. The
  latent space is smoothly organized by **block position** and the few high-
  coverage frames cluster together → the representation encodes physical state
  (the paper's probing result), at toy scale.
- `lewm_local.pt` — the trained checkpoint (used by w6).

## Honest notes

- This is a **toy**: 18k transitions + a 1.3 M-param model on an 8 GB laptop, vs
  the paper's 2000 expert episodes + a 224px ViT on a datacenter GPU. The
  representation-learning deliverables (dreams, t-SNE) come out well; the model
  is **not** accurate enough for high-success planning (see week 6/local).
- The probe decoder exists only for visualization and never touches the world
  model's weights (LeWM has no decoder by design).
- `viz.py` sets `KMP_DUPLICATE_LIB_OK` / `OMP_NUM_THREADS=1` to dodge a macOS
  torch+sklearn OpenMP crash in t-SNE.
