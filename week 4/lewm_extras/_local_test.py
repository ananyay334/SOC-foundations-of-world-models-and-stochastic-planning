"""Local validation of the LeWM latent pipeline WITHOUT the heavy swm stack.

We import the REAL model code (jepa.py, module.py from ../le-wm) and drive it
with a stub encoder that mimics the HuggingFace ViT interface. This checks that
our decoder, dream-rollout, and t-SNE logic line up with the actual model's
tensor shapes and APIs before we ever run on Colab.
"""

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

LEWM = Path(__file__).resolve().parents[1] / "le-wm"
sys.path.insert(0, str(LEWM))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from jepa import JEPA                       # real paper model
from module import ARPredictor, Embedder, MLP
from decoder import PixelDecoder

D = 192            # embed_dim (ViT-tiny hidden)
HS = 3             # history_size
ACT = 2            # pusht action dim
FRAMESKIP = 5
IMG = 112


class StubViT(nn.Module):
    """Mimics stable_pretraining vit_hf output: obj.last_hidden_state[:, 0]=cls."""

    def __init__(self, dim=D, patch=14):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(3, dim, kernel_size=patch, stride=patch)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x, interpolate_pos_encoding=False):
        # x: (N, 3, H, W) -> tokens (N, 1+num_patches, dim)
        p = self.proj(x).flatten(2).transpose(1, 2)          # (N, np, dim)
        cls = self.cls.expand(x.size(0), -1, -1)
        tokens = torch.cat([cls, p], dim=1)
        return types.SimpleNamespace(last_hidden_state=tokens)


def build_model():
    mlp = lambda: MLP(input_dim=D, output_dim=D, hidden_dim=256,
                      norm_fn=lambda n: nn.BatchNorm1d(n))
    return JEPA(
        encoder=StubViT(),
        predictor=ARPredictor(num_frames=HS, input_dim=D, hidden_dim=D,
                              output_dim=D, depth=2, heads=4, mlp_dim=256,
                              dim_head=48, dropout=0.0),
        action_encoder=Embedder(input_dim=FRAMESKIP * ACT, emb_dim=D),
        projector=mlp(),
        pred_proj=mlp(),
    )


def main():
    torch.manual_seed(0)
    model = build_model().eval()
    B, T = 4, HS + 1

    # ---- 1. training-style encode + predict ----
    info = {
        "pixels": torch.randn(B, T, 3, IMG, IMG),
        "action": torch.randn(B, T, FRAMESKIP * ACT),
    }
    out = model.encode(info)
    emb = out["emb"]
    print(f"[encode]  emb {tuple(emb.shape)}  act_emb {tuple(out['act_emb'].shape)}")
    assert emb.shape == (B, T, D)
    pred = model.predict(emb[:, :HS], out["act_emb"][:, :HS])
    print(f"[predict] pred {tuple(pred.shape)}  (targets emb[:, 1:] {tuple(emb[:, 1:].shape)})")
    assert pred.shape == (B, HS, D)

    # ---- 2. faithful dream rollout via model.rollout() ----
    n_future = 6
    ctx_pixels = torch.randn(B, 1, HS, 3, IMG, IMG)          # (B, S=1, H_ctx, C,H,W)
    actions = torch.randn(B, 1, HS + n_future, FRAMESKIP * ACT)
    info2 = {"pixels": ctx_pixels}
    info2 = model.rollout(info2, actions, history_size=HS)
    dreamed = info2["predicted_emb"]                         # (B, S, T', D)
    print(f"[rollout] predicted_emb {tuple(dreamed.shape)}")
    assert dreamed.shape[0] == B and dreamed.shape[-1] == D

    # ---- 3. decode latents (real + dreamed) ----
    dec = PixelDecoder(latent_dim=D, img_size=IMG)
    frames = dec(emb.reshape(-1, D))
    dream_frames = dec(dreamed.reshape(-1, D))
    print(f"[decode]  frames {tuple(frames.shape)}  dream_frames {tuple(dream_frames.shape)}")
    assert frames.shape[1:] == (3, IMG, IMG)

    # ---- 4. t-SNE smoke on encoded embeddings ----
    from sklearn.manifold import TSNE
    Z = emb.reshape(-1, D).detach().numpy()
    emb2d = TSNE(n_components=2, perplexity=5, init="pca").fit_transform(Z)
    print(f"[t-SNE]   {Z.shape} -> {emb2d.shape}")

    print("\n✅ all shape checks passed — scripts are consistent with the real model")


if __name__ == "__main__":
    main()
