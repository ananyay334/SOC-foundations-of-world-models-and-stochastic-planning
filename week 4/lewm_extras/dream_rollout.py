"""Produce decoded dream rollouts for a trained LeWM.

Given a real Push-T episode clip, we feed the first `history_size` frames as
context and let the world model imagine ("dream") future latents open-loop,
conditioned on the episode's real action sequence. We decode both the dreamed
latents and the ground-truth frames and lay them side by side.

Outputs a PNG grid (top row = ground truth, bottom row = decoded dream) and an
animated GIF.

Example:
    python dream_rollout.py --model pusht/lewm/weights_epoch_20.pt \
        --decoder decoder_pusht.pt --horizon 8 --episodes 3 --out dream.png
"""

import argparse

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch

from decoder import PixelDecoder
from lewm_common import (get_device, load_model, load_clip_dataset,
                         denorm_image, dream_latents)


def load_decoder(path, device):
    ckpt = torch.load(path, map_location="cpu")
    dec = PixelDecoder(latent_dim=ckpt["latent_dim"], img_size=ckpt["img_size"])
    dec.load_state_dict(ckpt["state_dict"])
    return dec.to(device).eval()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--dataset", default="pusht_expert_train.lance")
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--history-size", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=8, help="future steps to dream")
    ap.add_argument("--episodes", type=int, default=3, help="how many clips to render")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="dream.png")
    args = ap.parse_args()

    device = get_device()
    model = load_model(args.model, device)
    decoder = load_decoder(args.decoder, device)

    HS = args.history_size
    clip_len = HS + args.horizon
    ds = load_clip_dataset(args.dataset, clip_len, args.frameskip, args.img_size)

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(len(ds), size=args.episodes, replace=False)

    fig, axes = plt.subplots(2 * args.episodes, clip_len,
                             figsize=(1.4 * clip_len, 2.8 * args.episodes))
    if args.episodes == 1:
        axes = axes.reshape(2, clip_len)

    gif_frames = []
    for row, idx in enumerate(idxs):
        item = ds[int(idx)]
        pixels = item["pixels"]                       # (clip_len, 3, H, W) normalized
        actions = item["action"]                      # (clip_len, act_in)
        actions = torch.nan_to_num(actions, 0.0)

        gt = denorm_image(pixels)                     # (clip_len, H, W, 3)
        dreamed = dream_latents(model, pixels[:HS], actions,
                                history_size=HS, device=device)   # (T', D)
        dec_imgs = denorm_image(decoder(dreamed.to(device)))       # (T', H, W, 3)

        # align dreamed timeline to the clip (context frames shown as-is)
        for t in range(clip_len):
            ax_gt = axes[2 * row, t]; ax_dr = axes[2 * row + 1, t]
            ax_gt.imshow(gt[t]); ax_gt.axis("off")
            di = min(t, dec_imgs.shape[0] - 1)
            ax_dr.imshow(dec_imgs[di]); ax_dr.axis("off")
            if t < HS:
                ax_dr.set_title("ctx", fontsize=7)
        axes[2 * row, 0].set_ylabel("GT", fontsize=9)
        axes[2 * row + 1, 0].set_ylabel("dream", fontsize=9)

        # build a gif comparing gt vs dream over time for the first episode
        if row == 0:
            for t in range(clip_len):
                di = min(t, dec_imgs.shape[0] - 1)
                pair = np.concatenate([gt[t], dec_imgs[di]], axis=1)
                gif_frames.append((pair * 255).astype(np.uint8))

    fig.suptitle("LeWM decoded dream rollout — top: ground truth, bottom: imagined", fontsize=11)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved grid -> {args.out}")

    if gif_frames:
        gif_path = args.out.rsplit(".", 1)[0] + ".gif"
        imageio.mimsave(gif_path, gif_frames, duration=0.4)
        print(f"saved gif  -> {gif_path}")


if __name__ == "__main__":
    main()
