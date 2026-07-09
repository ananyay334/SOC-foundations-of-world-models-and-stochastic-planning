"""Shared helpers for the LeWM visualization scripts (run on Colab / a CUDA box).

These depend on the `stable_worldmodel` / `stable_pretraining` stack and the
paper repo (`le-wm`), so they are meant to run in the same environment where
you trained the model. The model-side tensor logic here is validated offline in
`_local_test.py` against the real `jepa.py`.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch

# Make the paper repo importable (jepa.py, module.py, utils.py).
LEWM_REPO = os.environ.get("LEWM_REPO", str(Path(__file__).resolve().parents[1] / "le-wm"))
sys.path.insert(0, LEWM_REPO)

import stable_pretraining as spt
import stable_worldmodel as swm
from utils import get_img_preprocessor, get_column_normalizer  # from le-wm/utils.py

# ImageNet stats used by the training img preprocessor (to invert for display).
IMAGENET_MEAN = torch.tensor(spt.data.dataset_stats.ImageNet["mean"]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor(spt.data.dataset_stats.ImageNet["std"]).view(1, 3, 1, 1)


def get_device():
    return "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


def load_model(run_name, device=None):
    """Load a trained LeWM checkpoint. `run_name` is a path under
    $STABLEWM_HOME/checkpoints, e.g. 'pusht/lewm/weights_epoch_20.pt'."""
    device = device or get_device()
    model = swm.wm.utils.load_pretrained(run_name)
    model = model.to(device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    return model


def build_transform(dataset, img_size, norm_cols=("action", "proprio", "state")):
    """Reproduce the exact training transform: ImageNet-normalize + resize the
    image, z-score every low-dim column (fit on the dataset)."""
    tfs = [get_img_preprocessor(source="pixels", target="pixels", img_size=img_size)]
    for col in norm_cols:
        if col in dataset.column_names:
            tfs.append(get_column_normalizer(dataset, col, col))
    return spt.data.transforms.Compose(*tfs)


def load_clip_dataset(name, clip_len, frameskip=5, img_size=112, cache_dir=None,
                      keys=("pixels", "action", "proprio", "state")):
    """Load the Push-T dataset as contiguous clips of length `clip_len`."""
    keys = [k for k in keys]
    ds = swm.data.load_dataset(
        name, transform=None, cache_dir=cache_dir,
        num_steps=clip_len, frameskip=frameskip,
        keys_to_load=keys,
        keys_to_cache=[k for k in keys if not k.startswith("pixels")],
    )
    ds.transform = build_transform(ds, img_size)
    return ds


def denorm_image(x):
    """Invert ImageNet normalization for display. x: (..., 3, H, W) -> [0,1] numpy HWC."""
    x = x.detach().cpu().float()
    shape = x.shape
    x = x.reshape(-1, *shape[-3:])
    x = x * IMAGENET_STD + IMAGENET_MEAN
    x = x.clamp(0, 1)
    x = x.permute(0, 2, 3, 1).numpy()                 # (N, H, W, 3)
    return x.reshape(*shape[:-3], *x.shape[1:])


@torch.no_grad()
def encode_frames(model, pixels, device=None):
    """Encode frames to LeWM latents.
    pixels: (N, 3, H, W) normalized -> returns emb (N, D)."""
    device = device or get_device()
    info = {"pixels": pixels.unsqueeze(1).to(device)}   # (N, T=1, C,H,W)
    emb = model.encode(info)["emb"]                      # (N, 1, D)
    return emb[:, 0]


@torch.no_grad()
def dream_latents(model, context_pixels, actions, history_size=3, device=None):
    """Faithful open-loop dream: encode `history_size` context frames, then
    autoregressively predict future latents conditioned on `actions`.

    context_pixels: (history_size, 3, H, W)  normalized
    actions:        (T, act_in_dim)          normalized, T >= history_size
    returns:        predicted_emb (T', D)     (context + dreamed future)
    """
    device = device or get_device()
    pixels = context_pixels.unsqueeze(0).unsqueeze(0).to(device)  # (B=1,S=1,H,C,H,W)
    act = actions.unsqueeze(0).unsqueeze(0).to(device)            # (B=1,S=1,T,A)
    info = model.rollout({"pixels": pixels}, act, history_size=history_size)
    return info["predicted_emb"][0, 0]                            # (T', D)
