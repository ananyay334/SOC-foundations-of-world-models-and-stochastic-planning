"""Compact, local LeWM (JEPA world model from pixels) for an 8 GB M2.

Faithful to the LeWM recipe — a next-embedding prediction loss + the SIGReg
isotropic-Gaussian regularizer (its core contribution), autoregressive latent
predictor, no decoder — but sized to train from scratch on a laptop:
  * a small conv encoder at 64x64 instead of a 224px ViT,
  * a shallow AR predictor,
  * latent dim 128.

Reuses the paper's `SIGReg` and `ARPredictor` from ../le-wm/module.py so the
objective and dynamics model match the real thing.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

LEWM_REPO = Path(__file__).resolve().parents[1] / "le-wm"
sys.path.insert(0, str(LEWM_REPO))
from module import SIGReg, ARPredictor, MLP  # noqa: E402  (paper code)


class ConvEncoder(nn.Module):
    """64x64x3 -> latent vector of size `dim`."""

    def __init__(self, dim=128, ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1), nn.GroupNorm(8, ch), nn.SiLU(),        # 32
            nn.Conv2d(ch, ch * 2, 4, 2, 1), nn.GroupNorm(8, ch * 2), nn.SiLU(),  # 16
            nn.Conv2d(ch * 2, ch * 4, 4, 2, 1), nn.GroupNorm(8, ch * 4), nn.SiLU(),  # 8
            nn.Conv2d(ch * 4, ch * 4, 4, 2, 1), nn.GroupNorm(8, ch * 4), nn.SiLU(),  # 4
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        )
        self.fc = nn.Linear(ch * 4, dim)

    def forward(self, x):
        return self.fc(self.net(x))


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(action_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, a):
        return self.net(a.float())


class LeWMLocal(nn.Module):
    def __init__(self, action_dim=2, dim=128, history_size=3, pred_depth=3, pred_heads=4):
        super().__init__()
        self.dim = dim
        self.history_size = history_size
        self.encoder = ConvEncoder(dim)
        self.projector = MLP(input_dim=dim, output_dim=dim, hidden_dim=256,
                             norm_fn=lambda n: nn.BatchNorm1d(n))
        self.action_encoder = ActionEncoder(action_dim, dim)
        self.predictor = ARPredictor(num_frames=history_size, input_dim=dim, hidden_dim=dim,
                                     output_dim=dim, depth=pred_depth, heads=pred_heads,
                                     mlp_dim=256, dim_head=32, dropout=0.0)
        self.pred_proj = MLP(input_dim=dim, output_dim=dim, hidden_dim=256,
                             norm_fn=lambda n: nn.BatchNorm1d(n))
        self.sigreg = SIGReg(knots=17, num_proj=256)

    # -- core --
    def encode_pixels(self, pixels):
        """pixels (B, T, C, H, W) or (N, C, H, W) -> emb same leading dims, +dim."""
        squeeze = pixels.dim() == 4
        if squeeze:
            pixels = pixels.unsqueeze(1)
        b, t = pixels.shape[:2]
        x = pixels.reshape(b * t, *pixels.shape[2:]).float()
        z = self.projector(self.encoder(x))
        z = z.reshape(b, t, self.dim)
        return z[:, 0] if squeeze else z

    def predict(self, emb, act_emb):
        """emb (B,T,D), act_emb (B,T,D) -> next-emb preds (B,T,D)."""
        pred = self.predictor(emb, act_emb)
        b = emb.size(0)
        pred = self.pred_proj(pred.reshape(-1, self.dim)).reshape(b, -1, self.dim)
        return pred

    # -- training loss (LeWM: next-emb MSE + SIGReg) --
    def loss(self, pixels, actions, sigreg_weight=0.09, num_preds=1):
        emb = self.encode_pixels(pixels)                       # (B,T,D)
        act_emb = self.action_encoder(actions)                 # (B,T,D)
        ctx = self.history_size
        pred = self.predict(emb[:, :ctx], act_emb[:, :ctx])    # (B,ctx,D)
        tgt = emb[:, num_preds:num_preds + ctx]                # aligned next embs
        n = min(pred.size(1), tgt.size(1))
        pred_loss = (pred[:, :n] - tgt[:, :n].detach()).pow(2).mean()
        sig = self.sigreg(emb.transpose(0, 1))
        return pred_loss + sigreg_weight * sig, {"pred": pred_loss.item(), "sigreg": sig.item()}

    # -- planning: differentiable cost of action candidates vs a goal image --
    def rollout_latent(self, z0, actions):
        """Open-loop latent rollout.
        z0: (B, D) current latent ; actions: (B, T, A) -> final latent (B, D)."""
        B = z0.size(0)
        HS = self.history_size
        emb = z0.unsqueeze(1).repeat(1, HS, 1)                 # seed history with z0
        act_emb = self.action_encoder(actions)                # (B,T,D)
        T = actions.size(1)
        for t in range(T):
            a = act_emb[:, max(0, t - HS + 1):t + 1]
            pad = HS - a.size(1)
            if pad > 0:
                a = torch.cat([a[:, :1].expand(B, pad, self.dim), a], dim=1)
            nxt = self.predict(emb[:, -HS:], a)[:, -1:]        # (B,1,D)
            emb = torch.cat([emb, nxt], dim=1)
        return emb[:, -1]                                      # (B, D)

    def get_cost(self, info_dict, action_candidates):
        """info_dict: 'pixels' (B,C,H,W) current, 'goal' (B,C,H,W) goal image.
        action_candidates: (B, S, T, A) -> cost (B, S) (MSE to goal latent)."""
        device = next(self.parameters()).device
        pixels = info_dict["pixels"].to(device).float()
        goal = info_dict["goal"].to(device).float()
        if pixels.dim() == 5:      # (B,S,C,H,W) already expanded -> take one per env
            pixels = pixels[:, 0]
            goal = goal[:, 0]
        B, S, T, A = action_candidates.shape
        z0 = self.encode_pixels(pixels)                        # (B,D)
        zg = self.encode_pixels(goal)                          # (B,D)
        z0 = z0.unsqueeze(1).expand(B, S, self.dim).reshape(B * S, self.dim)
        acts = action_candidates.to(device).reshape(B * S, T, A)
        zT = self.rollout_latent(z0, acts)                     # (B*S, D)
        zg = zg.unsqueeze(1).expand(B, S, self.dim).reshape(B * S, self.dim)
        cost = (zT - zg).pow(2).mean(-1).reshape(B, S)
        return cost


if __name__ == "__main__":
    m = LeWMLocal()
    px = torch.randn(4, 4, 3, 64, 64)
    ac = torch.randn(4, 4, 2)
    loss, logs = m.loss(px, ac)
    print("params:", sum(p.numel() for p in m.parameters()) / 1e6, "M")
    print("loss:", float(loss), logs)
    info = {"pixels": torch.randn(3, 3, 64, 64), "goal": torch.randn(3, 3, 64, 64)}
    cands = torch.randn(3, 8, 5, 2)
    c = m.get_cost(info, cands)
    print("cost:", tuple(c.shape))
