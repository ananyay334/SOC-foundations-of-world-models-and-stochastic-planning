"""A small pixel decoder for visualizing LeWM latents.

LeWM is a JEPA: it has NO decoder, because it never reconstructs pixels during
training. To *visualize* what a latent (or a dreamed/predicted latent) "looks
like", we train a lightweight decoder AFTER the fact, on top of the FROZEN
encoder+projector, to map a latent embedding back to an image. This is exactly
the probe-decoder trick the paper uses for its decoded dream rollouts: the
decoder is never used to train the world model, only to peek into its latents.

Input : latent embedding of shape (B, D)   (D = projector output dim, e.g. 192)
Output: image of shape (B, 3, H, W)
"""

import torch
import torch.nn as nn


class PixelDecoder(nn.Module):
    """DCGAN-style transposed-conv decoder: latent vector -> RGB image.

    `img_size` must be a multiple of 16 (4 upsampling stages of x2 from a 4x4
    base ... actually 5 stages: 4->8->16->32->64->... ). We compute the number
    of upsampling blocks from img_size so it works for 64/112/128/224 etc.
    """

    def __init__(self, latent_dim: int = 192, img_size: int = 112, base_ch: int = 256):
        super().__init__()
        self.img_size = img_size

        # Start from a 4x4 feature map and upsample x2 until we reach img_size.
        # For sizes that are not powers of two (e.g. 112, 224) we upsample to
        # the next power of two >= img_size, then resize down at the end.
        start = 4
        target = 1
        while target < img_size:
            target *= 2
        self.render_size = target                     # power-of-two >= img_size
        n_up = 0
        s = start
        while s < target:
            s *= 2
            n_up += 1

        self.fc = nn.Linear(latent_dim, base_ch * start * start)
        self.base_ch = base_ch
        self.start = start

        layers = []
        ch = base_ch
        for i in range(n_up):
            out_ch = max(base_ch // (2 ** (i + 1)), 32)
            layers += [
                nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
            ]
            ch = out_ch
        self.up = nn.Sequential(*layers)
        self.to_rgb = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(-1, self.base_ch, self.start, self.start)
        x = self.up(x)
        x = torch.sigmoid(self.to_rgb(x))             # (B, 3, render_size, render_size)
        if self.render_size != self.img_size:
            x = nn.functional.interpolate(
                x, size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            )
        return x


if __name__ == "__main__":
    for size in (64, 112, 128, 224):
        dec = PixelDecoder(latent_dim=192, img_size=size)
        z = torch.randn(2, 192)
        out = dec(z)
        n = sum(p.numel() for p in dec.parameters())
        print(f"img_size={size:3d} -> out {tuple(out.shape)}  params {n/1e6:.2f}M")
