"""Learned multi-modal pedestrian trajectory forecaster.

Architecture ("SocialMDN"):

* Ego encoder - GRU over the observed track, expressed relative to the last observed
  position and rotated into a heading-aligned frame (x axis = recent direction of motion).
  Per-step features: position, displacement, validity mask (short tracks are left-padded).
* Social encoder - MLP over up to ``MAX_NEIGHBORS`` neighbours (relative position and
  relative velocity in the ego frame) followed by single-head attention keyed on the ego
  state. Absent neighbours are masked out.
* Mixture decoder - K modes. Each mode predicts per-step displacements (cumulatively summed
  into a path) and per-step along/cross-track standard deviations that are constrained to be
  non-decreasing in time, so uncertainty grows with horizon by construction.
  A separate head predicts the mode logits.

Training objective: Gaussian-mixture negative log-likelihood of the full future trajectory
plus a winner-takes-all L2 term on the closest mode (encourages diverse, accurate modes).
Mode probabilities are post-hoc calibrated with temperature scaling on a validation split
(see ``training/calibrate.py``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .constants import MAX_NEIGHBORS, NUM_MODES, OBS_LEN, PRED_LEN


@dataclass
class ModelConfig:
    obs_len: int = OBS_LEN
    pred_len: int = PRED_LEN
    num_modes: int = NUM_MODES
    hidden: int = 96
    social: bool = True
    max_neighbors: int = MAX_NEIGHBORS
    min_sigma: float = 0.03
    dropout: float = 0.1

    def to_dict(self) -> dict:
        return asdict(self)


def heading_frame(obs: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (origin, rotation) that map world coords into the ego heading frame.

    obs: (B, T, 2) absolute positions, mask: (B, T) validity (left padded).
    origin is the last observed position. Rotation R (B, 2, 2) maps world -> ego: p_ego = R @ (p - o).
    Heading is estimated from the first valid observation to the last; if the agent has moved
    less than 0.05 units the identity rotation is used.
    """
    origin = obs[:, -1]
    first_idx = (mask > 0).float().argmax(dim=1)
    first = obs[torch.arange(obs.shape[0]), first_idx]
    d = origin - first
    norm = d.norm(dim=-1, keepdim=True)
    moving = norm > 0.05
    unit = torch.where(moving, d / norm.clamp_min(1e-6), torch.tensor([1.0, 0.0], device=obs.device))
    c, s = unit[:, 0], unit[:, 1]
    rot = torch.stack([torch.stack([c, s], -1), torch.stack([-s, c], -1)], -2)  # (B,2,2)
    return origin, rot


class SocialMDN(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        h = self.cfg.hidden
        self.in_proj = nn.Linear(5, h)
        self.gru = nn.GRU(h, h, batch_first=True)
        self.nbr_mlp = nn.Sequential(nn.Linear(5, h), nn.ReLU(), nn.Linear(h, h))
        self.q = nn.Linear(h, h)
        self.k = nn.Linear(h, h)
        self.v = nn.Linear(h, h)
        self.fuse = nn.Sequential(
            nn.Linear(2 * h, 2 * h), nn.ReLU(), nn.Dropout(self.cfg.dropout), nn.Linear(2 * h, 2 * h), nn.ReLU()
        )
        self.mode_emb = nn.Embedding(self.cfg.num_modes, 2 * h)
        self.decoder = nn.Sequential(
            nn.Linear(2 * h, 2 * h), nn.ReLU(), nn.Linear(2 * h, self.cfg.pred_len * 4)
        )
        self.logit_head = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, self.cfg.num_modes))

    def forward(
        self,
        obs: torch.Tensor,
        obs_mask: torch.Tensor,
        nbrs: torch.Tensor | None = None,
        nbr_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        obs: (B, T_obs, 2) absolute positions (left-padded; padded entries repeat first valid point)
        obs_mask: (B, T_obs) 1 for valid steps
        nbrs: (B, N, 4) neighbour [x, y, vx, vy] absolute position / velocity (per step) at last obs
        nbr_mask: (B, N)
        Returns world-frame mixture parameters:
          mu (B, K, T_pred, 2), cov (B, K, T_pred, 2, 2), logits (B, K), sigma_ego (B,K,T,2)
        """
        B = obs.shape[0]
        origin, rot = heading_frame(obs, obs_mask)
        rel = torch.einsum("bij,btj->bti", rot, obs - origin[:, None])
        disp = torch.zeros_like(rel)
        disp[:, 1:] = rel[:, 1:] - rel[:, :-1]
        m = obs_mask[..., None]
        disp = disp * m * torch.cat([torch.zeros_like(m[:, :1]), m[:, :-1]], 1)
        feats = torch.cat([rel * m, disp, m], -1)
        out, hN = self.gru(F.relu(self.in_proj(feats)))
        ego = hN[-1]  # (B,h)

        ctx = torch.zeros_like(ego)
        if self.cfg.social and nbrs is not None and nbr_mask is not None and nbrs.shape[1] > 0:
            ego_vel = disp[:, -1]  # per-step velocity in ego frame
            npos = torch.einsum("bij,bnj->bni", rot, nbrs[..., :2] - origin[:, None])
            nvel = torch.einsum("bij,bnj->bni", rot, nbrs[..., 2:4]) - ego_vel[:, None]
            dist = npos.norm(dim=-1, keepdim=True)
            nf = torch.cat([npos, nvel, dist], -1)
            ne = self.nbr_mlp(nf)
            q = self.q(ego)[:, None]  # (B,1,h)
            att = (q * self.k(ne)).sum(-1) / math.sqrt(ne.shape[-1])  # (B,N)
            att = att.masked_fill(nbr_mask <= 0, -1e9)
            w = torch.softmax(att, -1) * (nbr_mask.sum(-1, keepdim=True) > 0).float()
            ctx = (w[..., None] * self.v(ne)).sum(1)

        z = self.fuse(torch.cat([ego, ctx], -1))  # (B,2h)
        logits = self.logit_head(z)
        K, T = self.cfg.num_modes, self.cfg.pred_len
        zk = z[:, None] + self.mode_emb.weight[None]  # (B,K,2h)
        dec = self.decoder(zk).view(B, K, T, 4)
        mu_ego = torch.cumsum(dec[..., :2], dim=2)
        sig_ego = self.cfg.min_sigma + torch.cumsum(F.softplus(dec[..., 2:]) * 0.1, dim=2)  # non-decreasing
        # back to world frame: p = R^T p_ego + o ; cov = R^T diag(s^2) R
        rt = rot.transpose(1, 2)
        mu = torch.einsum("bij,bktj->bkti", rt, mu_ego) + origin[:, None, None]
        d = torch.diag_embed(sig_ego**2)  # (B,K,T,2,2)
        cov = rt[:, None, None] @ d @ rot[:, None, None]
        return {"mu": mu, "cov": cov, "logits": logits, "mu_ego": mu_ego, "sigma_ego": sig_ego, "rot": rot, "origin": origin}


def mixture_log_prob(out: dict[str, torch.Tensor], fut: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """log p(future | past) under the mixture; fut (B, T, 2) world coords. Returns (B,)."""
    rot, origin = out["rot"], out["origin"]
    fut_ego = torch.einsum("bij,btj->bti", rot, fut - origin[:, None])  # (B,T,2)
    diff = fut_ego[:, None] - out["mu_ego"]  # (B,K,T,2)
    s = out["sigma_ego"]
    lp = -0.5 * (diff / s) ** 2 - torch.log(s) - 0.5 * math.log(2 * math.pi)
    lp = lp.sum(dim=(-1, -2))  # (B,K)
    logpi = torch.log_softmax(out["logits"] / temperature, -1)
    return torch.logsumexp(logpi + lp, -1)


def loss_fn(out: dict[str, torch.Tensor], fut: torch.Tensor, wta_weight: float = 1.0) -> tuple[torch.Tensor, dict]:
    nll = -mixture_log_prob(out, fut).mean() / fut.shape[1]
    err = (out["mu"] - fut[:, None]).norm(dim=-1).mean(-1)  # (B,K) ADE per mode
    best = err.argmin(-1)
    wta = err.gather(1, best[:, None]).mean()
    cls = F.cross_entropy(out["logits"], best)
    loss = nll + wta_weight * wta + 0.5 * cls
    return loss, {"nll_per_step": nll.item(), "wta_ade": wta.item(), "mode_ce": cls.item()}


def save_checkpoint(path, model: SocialMDN, meta: dict) -> None:
    torch.save({"config": model.cfg.to_dict(), "state_dict": model.state_dict(), "meta": meta}, path)


def load_checkpoint(path, device: str | torch.device = "cpu") -> tuple[SocialMDN, dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    model = SocialMDN(ModelConfig(**ck["config"]))
    model.load_state_dict(ck["state_dict"])
    model.eval().to(device)
    return model, ck.get("meta", {})
