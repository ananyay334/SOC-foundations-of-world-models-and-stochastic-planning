"""t-SNE of the LeWM latent space on Push-T.

Encodes many frames sampled across episodes into LeWM latents, projects them to
2D with t-SNE, and colors points by physical quantities (block angle, agent
position, and episode progress). Clustering/structure by these quantities is
evidence the latent space encodes meaningful physical structure — the probing
result highlighted in the paper.

Example:
    python tsne_latents.py --model pusht/lewm/weights_epoch_20.pt \
        --num 3000 --img-size 112 --out tsne.png
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from lewm_common import get_device, load_model, load_clip_dataset, encode_frames


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="pusht_expert_train.lance")
    ap.add_argument("--img-size", type=int, default=112)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--num", type=int, default=3000, help="frames to embed")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="tsne.png")
    args = ap.parse_args()

    device = get_device()
    model = load_model(args.model, device)

    # clip_len=1 gives single frames; keep state for coloring.
    ds = load_clip_dataset(args.dataset, clip_len=1, frameskip=args.frameskip,
                           img_size=args.img_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    embs, states = [], []
    n = 0
    for batch in loader:
        pixels = batch["pixels"][:, 0]                    # (B,3,H,W)
        emb = encode_frames(model, pixels, device).cpu()
        embs.append(emb)
        if "state" in batch:
            states.append(batch["state"][:, 0].cpu())
        n += pixels.size(0)
        if n >= args.num:
            break

    Z = torch.cat(embs)[: args.num].numpy()
    state = torch.cat(states)[: args.num].numpy() if states else None
    print(f"embedded {Z.shape[0]} frames, dim {Z.shape[1]}")

    from sklearn.manifold import TSNE
    Y = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
             random_state=args.seed).fit_transform(Z)

    # PushT state = [agent_x, agent_y, block_x, block_y, block_angle, ...]
    color_specs = [("progress (sample order)", np.arange(len(Y)))]
    if state is not None and state.shape[1] >= 5:
        color_specs = [
            ("block angle", state[:, 4]),
            ("block x", state[:, 2]),
            ("agent x", state[:, 0]),
        ]

    fig, axes = plt.subplots(1, len(color_specs), figsize=(5.2 * len(color_specs), 4.6))
    if len(color_specs) == 1:
        axes = [axes]
    for ax, (name, c) in zip(axes, color_specs):
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=c, cmap="viridis", s=6, alpha=0.8)
        ax.set_title(f"LeWM latent t-SNE — colored by {name}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
