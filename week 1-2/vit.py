"""A small, from-scratch Vision Transformer (ViT) encoder.

This is a toy but faithful implementation of the encoder described in
"An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020), sized down so it
trains on CIFAR-10 on a laptop.

Architecture:
    image -> patch embedding (conv) -> prepend [CLS] token -> + positional
    embedding -> N transformer encoder blocks -> LayerNorm -> CLS token -> head
"""

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class ViTConfig:
    image_size: int = 32       # CIFAR-10 images are 32x32
    patch_size: int = 4        # -> (32/4)^2 = 64 patches
    in_channels: int = 3
    num_classes: int = 10
    dim: int = 192             # embedding / hidden dimension
    depth: int = 6             # number of transformer blocks
    heads: int = 3             # attention heads (dim must be divisible by heads)
    mlp_ratio: float = 4.0     # hidden dim of the MLP = dim * mlp_ratio
    dropout: float = 0.1

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2


class PatchEmbedding(nn.Module):
    """Split the image into patches and linearly embed each one.

    Implemented with a strided convolution: a kernel of size `patch_size` with
    stride `patch_size` is exactly a per-patch linear projection.
    """

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.proj = nn.Conv2d(
            cfg.in_channels, cfg.dim,
            kernel_size=cfg.patch_size, stride=cfg.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W) -> (B, dim, H/p, W/p) -> (B, num_patches, dim)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        assert cfg.dim % cfg.heads == 0, "dim must be divisible by heads"
        self.heads = cfg.heads
        self.head_dim = cfg.dim // cfg.heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(cfg.dim, cfg.dim * 3)
        self.proj = nn.Linear(cfg.dim, cfg.dim)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.proj_drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        # (B, N, 3D) -> (3, B, heads, N, head_dim)
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = attn @ v                               # (B, heads, N, head_dim)
        out = out.transpose(1, 2).reshape(B, N, D)   # (B, N, D)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class MLP(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        hidden = int(cfg.dim * cfg.mlp_ratio)
        self.fc1 = nn.Linear(cfg.dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, cfg.dim)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class EncoderBlock(nn.Module):
    """Pre-norm transformer encoder block with residual connections."""

    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.dim)
        self.attn = MultiHeadSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.dim)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ViT(nn.Module):
    def __init__(self, cfg: ViTConfig):
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbedding(cfg)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, cfg.num_patches + 1, cfg.dim)
        )
        self.pos_drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.depth)])
        self.norm = nn.LayerNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.zeros_(m.bias)
                nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)                       # (B, num_patches, dim)

        cls = self.cls_token.expand(B, -1, -1)        # (B, 1, dim)
        x = torch.cat([cls, x], dim=1)                # (B, num_patches+1, dim)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_out = x[:, 0]                              # take the [CLS] token
        return self.head(cls_out)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    cfg = ViTConfig()
    model = ViT(cfg)
    dummy = torch.randn(2, 3, cfg.image_size, cfg.image_size)
    out = model(dummy)
    print(f"num_patches: {cfg.num_patches}")
    print(f"params:      {model.num_params():,}")
    print(f"output:      {tuple(out.shape)}  (expected (2, {cfg.num_classes}))")
