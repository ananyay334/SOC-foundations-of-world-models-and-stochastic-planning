"""w4 deliverables from the locally-trained LeWM: decoded dream rollouts + t-SNE.

Trains a small probe decoder on frozen LeWM latents (the model itself has no
decoder), then:
  * dream.png / dream.gif  — open-loop latent rollouts decoded to pixels,
  * tsne.png               — t-SNE of the latent space colored by physical state.

    python viz.py --model lewm_local.pt --data pusht_data.npz
"""

import os
# Avoid a macOS torch+sklearn OpenMP double-load segfault at the t-SNE step.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lewm_extras"))
from decoder import PixelDecoder            # reuse the w4 probe decoder
from lewm_local import LeWMLocal


def get_device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path, device):
    ck = torch.load(path, map_location="cpu"); c = ck["cfg"]
    m = LeWMLocal(action_dim=c["action_dim"], dim=c["dim"], history_size=c["history"])
    m.load_state_dict(ck["state_dict"])
    return m.to(device).eval().requires_grad_(False), c


def frames_chw(px):
    return torch.from_numpy(px).float().permute(0, 3, 1, 2) / 255.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lewm_local.pt")
    ap.add_argument("--data", default="pusht_data.npz")
    ap.add_argument("--dec-epochs", type=int, default=6)
    ap.add_argument("--img-size", type=int, default=64)
    args = ap.parse_args()

    device = get_device(); print(f"device={device}")
    model, cfg = load_model(args.model, device)
    D = cfg["dim"]; HS = cfg["history"]
    d = np.load(args.data)
    px_all, ac_all, ep_all, st_all = d["pixels"], d["actions"], d["ep_idx"], d["states"]

    # ---- 1. train a probe decoder on frozen latents ----
    dec = PixelDecoder(latent_dim=D, img_size=args.img_size).to(device)
    opt = torch.optim.AdamW(dec.parameters(), lr=2e-4)
    N = len(px_all); bs = 128
    for ep in range(1, args.dec_epochs + 1):
        perm = np.random.permutation(N); tot = 0.0
        for i in range(0, N - bs, bs):
            idx = perm[i:i + bs]
            imgs = frames_chw(px_all[idx]).to(device)
            with torch.no_grad():
                z = model.encode_pixels(imgs)
            rec = dec(z)
            loss = nn.functional.mse_loss(rec, imgs)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()
        print(f"decoder epoch {ep}/{args.dec_epochs}  mse {tot/(N//bs):.4f}")

    # ---- 2. decoded dream rollouts ----
    HZ = 8
    rng = np.random.default_rng(0)
    rows = 3
    fig, axes = plt.subplots(2 * rows, HS + HZ, figsize=(1.3 * (HS + HZ), 2.6 * rows))
    gif = []
    # pick episode starts with a full clip inside one episode
    starts = [i for i in range(N - (HS + HZ)) if ep_all[i] == ep_all[i + HS + HZ - 1]]
    for r, s in enumerate(rng.choice(starts, rows, replace=False)):
        clip = frames_chw(px_all[s:s + HS + HZ]).to(device)              # (T,3,H,W)
        acts = torch.from_numpy(ac_all[s:s + HS + HZ]).float().to(device) / 512.0
        with torch.no_grad():
            z0 = model.encode_pixels(clip[:HS].unsqueeze(0))[:, -1]      # latent after context
            # autoregressive dream of future latents
            emb = z0.unsqueeze(1).repeat(1, HS, 1)
            ae = model.action_encoder(acts.unsqueeze(0))
            dreamed = []
            for t in range(HS, HS + HZ):
                a = ae[:, t - HS + 1:t + 1]
                nxt = model.predict(emb[:, -HS:], a)[:, -1:]
                emb = torch.cat([emb, nxt], 1)
                dreamed.append(nxt[:, 0])
            dream_z = torch.stack([z0] + dreamed, 1)[0]                  # (HZ+1,D)
            dream_imgs = dec(dream_z).cpu()
        gt = clip.cpu()
        for t in range(HS + HZ):
            axes[2 * r, t].imshow(gt[t].permute(1, 2, 0).clamp(0, 1)); axes[2 * r, t].axis("off")
            di = min(max(t - HS + 1, 0), dream_imgs.shape[0] - 1)
            axes[2 * r + 1, t].imshow(dream_imgs[di].permute(1, 2, 0).clamp(0, 1)); axes[2 * r + 1, t].axis("off")
        axes[2 * r, 0].set_ylabel("GT", fontsize=9); axes[2 * r + 1, 0].set_ylabel("dream", fontsize=9)
        if r == 0:
            for t in range(HS + HZ):
                di = min(max(t - HS + 1, 0), dream_imgs.shape[0] - 1)
                pair = np.concatenate([gt[t].permute(1, 2, 0).numpy(),
                                       dream_imgs[di].permute(1, 2, 0).clamp(0, 1).numpy()], 1)
                gif.append((pair * 255).astype(np.uint8))
    fig.suptitle("Local LeWM decoded dream rollout — top: ground truth, bottom: imagined")
    fig.tight_layout(); fig.savefig("dream.png", dpi=130, bbox_inches="tight")
    imageio.mimsave("dream.gif", gif, duration=0.4)
    print("saved dream.png, dream.gif")

    # ---- 3. t-SNE of latents ----
    from sklearn.manifold import TSNE
    K = min(2500, N)
    idx = rng.choice(N, K, replace=False)
    with torch.no_grad():
        z = torch.cat([model.encode_pixels(frames_chw(px_all[idx[i:i+256]]).to(device)).cpu()
                       for i in range(0, K, 256)]).numpy()
    st = st_all[idx]; cov = d["coverage"][idx]
    Y = TSNE(n_components=2, perplexity=30, init="pca", random_state=0).fit_transform(z)
    specs = [("block angle", st[:, 4]), ("block x", st[:, 2]), ("coverage (→goal)", cov)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (nm, c) in zip(axes, specs):
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=c, cmap="viridis", s=6, alpha=0.8)
        ax.set_title(f"LeWM latent t-SNE — {nm}"); ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig("tsne.png", dpi=130, bbox_inches="tight")
    print("saved tsne.png")


if __name__ == "__main__":
    main()
