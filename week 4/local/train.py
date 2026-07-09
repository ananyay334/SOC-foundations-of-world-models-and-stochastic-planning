"""Train the compact local LeWM from scratch on the collected Push-T data.

Builds short temporal windows within each episode and optimizes the LeWM
objective (next-embedding MSE + SIGReg). Runs on MPS/CPU. Saves a checkpoint
and a loss-curve PNG.
"""

import argparse
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lewm_local import LeWMLocal


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


class WindowDataset(Dataset):
    """Sliding windows of length T that stay within one episode."""

    def __init__(self, npz_path, window):
        d = np.load(npz_path)
        self.px = d["pixels"]                      # (N,64,64,3) uint8
        self.ac = d["actions"]                     # (N,2)
        self.ep = d["ep_idx"]
        self.T = window
        self.starts = []
        for i in range(len(self.px) - window):
            if self.ep[i] == self.ep[i + window - 1]:
                self.starts.append(i)

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, k):
        i = self.starts[k]
        sl = slice(i, i + self.T)
        px = torch.from_numpy(self.px[sl]).float().permute(0, 3, 1, 2) / 255.0   # (T,3,H,W)
        ac = torch.from_numpy(self.ac[sl]) / 512.0                                # normalize to ~[0,1]
        return px, ac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="pusht_data.npz")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--history", type=int, default=3)
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--sigreg", type=float, default=0.09)
    ap.add_argument("--out", default="lewm_local.pt")
    args = ap.parse_args()

    device = get_device()
    print(f"device={device}")
    ds = WindowDataset(args.data, args.window)
    print(f"{len(ds)} training windows (T={args.window})")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    model = LeWMLocal(action_dim=2, dim=args.dim, history_size=args.history).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    hist = {"loss": [], "pred": [], "sigreg": []}
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        agg = {"loss": 0.0, "pred": 0.0, "sigreg": 0.0}
        n = 0
        for px, ac in loader:
            px, ac = px.to(device), ac.to(device)
            loss, logs = model.loss(px, ac, sigreg_weight=args.sigreg, num_preds=1)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            agg["loss"] += loss.item(); agg["pred"] += logs["pred"]; agg["sigreg"] += logs["sigreg"]; n += 1
        sched.step()
        for k in agg:
            hist[k].append(agg[k] / n)
        print(f"epoch {epoch:2d}/{args.epochs} | loss {agg['loss']/n:.4f} "
              f"| pred {agg['pred']/n:.4f} | sigreg {agg['sigreg']/n:.4f} | {time.time()-t0:.1f}s")
        torch.save({"state_dict": model.state_dict(),
                    "cfg": {"dim": args.dim, "history": args.history, "action_dim": 2}},
                   args.out)

    # loss curve
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(hist["loss"], label="total"); ax[0].plot(hist["pred"], label="pred MSE")
    ax[0].set_title("LeWM training loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["sigreg"], color="tab:red"); ax[1].set_title("SIGReg (Gaussianity)")
    ax[1].set_xlabel("epoch")
    fig.tight_layout(); fig.savefig("loss_curve.png", dpi=130)
    print(f"saved {args.out} and loss_curve.png")


if __name__ == "__main__":
    main()
