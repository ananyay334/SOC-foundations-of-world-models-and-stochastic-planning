# Toy ViT on CIFAR-10

A small, from-scratch **Vision Transformer** encoder trained on CIFAR-10.
Everything is plain PyTorch — no timm, no pretrained weights. Runs on Apple
MPS, CUDA, or CPU.

## Files

| File          | What it is                                                        |
|---------------|-------------------------------------------------------------------|
| `vit.py`      | The ViT encoder: patch embedding, `[CLS]` token, positional embeddings, N pre-norm transformer blocks (multi-head self-attention + MLP), classifier head. |
| `train.py`    | Training loop on CIFAR-10 (AdamW, cosine schedule, augmentation).  |
| `get_data.py` | Downloads + extracts CIFAR-10 into ImageFolder layout.            |

## Setup

```bash
pip install torch torchvision
python get_data.py          # downloads CIFAR-10 (~168 MB) to ./data/cifar10
```

## Quick check (≈1 min)

```bash
python vit.py               # sanity-check the model builds and runs
python train.py --smoke     # 1 epoch on a 2k subset — verifies the pipeline
```

## Train for real

```bash
python train.py --epochs 30
```

The best checkpoint is saved to `vit_cifar10.pt`.

## Model

Default config (`ViTConfig` in `vit.py`), ~2.7M params:

| Hyperparameter | Value |
|----------------|-------|
| image size     | 32    |
| patch size     | 4 → 64 patches |
| embed dim      | 192   |
| depth          | 6 blocks |
| heads          | 3     |
| MLP ratio      | 4     |

## Expected results

A toy ViT trained from scratch on CIFAR-10 (no pretraining) typically reaches
**~75–80% test accuracy** in ~30 epochs. ViTs lack the built-in spatial priors
of CNNs, so on small datasets they need augmentation and more epochs to compete
— that trade-off is exactly what this toy illustrates.
