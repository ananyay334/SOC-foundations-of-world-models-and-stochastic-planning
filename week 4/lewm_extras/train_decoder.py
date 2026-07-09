"""Train a pixel decoder from scratch on FROZEN LeWM latents.

The world model stays frozen; we only learn decoder: latent -> image. This is
the probe decoder used for the decoded dream rollouts. Nothing here touches the
world model's weights, so it does not affect the from-scratch WM training.

Example:
    python train_decoder.py --model pusht/lewm/weights_epoch_20.pt \
        --img-size 112 --epochs 8 --out decoder_pusht.pt
"""

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from decoder import PixelDecoder
from lewm_common import (get_device, load_model, load_clip_dataset,
                         denorm_image, IMAGENET_MEAN, IMAGENET_STD)


def to01(x):
    """normalized (B,3,H,W) -> [0,1] target for reconstruction."""
    return (x.cpu() * IMAGENET_STD + IMAGENET_MEAN).clamp(0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="checkpoint path under $STABLEWM_HOME/checkpoints")
    ap.add_argument("--dataset", default="pusht_expert_train.lance")
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--clip-len", type=int, default=4)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-steps", type=int, default=2000, help="cap steps/epoch for speed")
    ap.add_argument("--latent-dim", type=int, default=192)
    ap.add_argument("--out", default="decoder_pusht.pt")
    args = ap.parse_args()

    device = get_device()
    print(f"device={device}")

    model = load_model(args.model, device)
    ds = load_clip_dataset(args.dataset, args.clip_len, args.frameskip, args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=0, drop_last=True)

    decoder = PixelDecoder(latent_dim=args.latent_dim, img_size=args.img_size).to(device)
    opt = torch.optim.AdamW(decoder.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        decoder.train()
        running = seen = 0
        for step, batch in enumerate(loader, 1):
            if step > args.max_steps:
                break
            pixels = batch["pixels"].to(device)                 # (B,T,3,H,W)
            B, T = pixels.shape[:2]
            flat = pixels.reshape(B * T, *pixels.shape[2:])
            with torch.no_grad():
                emb = model.encode({"pixels": flat.unsqueeze(1)})["emb"][:, 0]
            recon = decoder(emb)                                # (B*T,3,H,W) in [0,1]
            target = to01(flat).to(device)
            loss = nn.functional.mse_loss(recon, target)
            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item() * flat.size(0); seen += flat.size(0)
        print(f"epoch {epoch}/{args.epochs}  recon_mse {running/seen:.5f}")
        torch.save({"state_dict": decoder.state_dict(),
                    "img_size": args.img_size, "latent_dim": args.latent_dim}, args.out)

    print(f"saved decoder -> {args.out}")


if __name__ == "__main__":
    main()
