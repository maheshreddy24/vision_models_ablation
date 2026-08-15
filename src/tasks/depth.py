"""Monocular depth on NYU Depth v2 from frozen (masked) DINOv2 patch features.

Head: per-token Linear(D -> 1) predicting log-depth, scattered back onto the
patch grid via kept indices, bilinearly upsampled to image resolution.
Loss/metrics are computed on valid pixels; in `visible_only` mode (default)
they are additionally restricted to pixels of KEPT patches - the honest
measurement. `full_image` mode fills dropped positions with a learned
"unknown" embedding so the head must inpaint depth for unseen regions.
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data.nyu import NYUDepth
from ..models.masked_dinov2 import MaskedDinoV2
from .base import train_head

SILOG_LAMBDA = 0.85
TASK_NAME = "depth"


class LinearDepthHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.unknown = nn.Parameter(torch.zeros(dim))  # used only in full_image mode

    def forward(self, patch: torch.Tensor, kept_idx: torch.Tensor,
                grid: Tuple[int, int], full_image: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """-> (log-depth (B,1,gh,gw), visibility (B,1,gh,gw) bool)."""
        B, k, D = patch.shape
        gh, gw = grid
        n = gh * gw
        feat = patch.new_zeros(B, n, D)
        feat.scatter_(1, kept_idx.unsqueeze(-1).expand(-1, -1, D), patch)
        vis = torch.zeros(B, n, dtype=torch.bool, device=patch.device)
        vis.scatter_(1, kept_idx, True)
        if full_image:
            feat = torch.where(vis.unsqueeze(-1), feat,
                               self.unknown.view(1, 1, D).expand(B, n, D))
        pred = self.linear(feat).view(B, 1, gh, gw)
        return pred, vis.view(B, 1, gh, gw)


def _upsample(pred_grid: torch.Tensor, vis_grid: torch.Tensor,
              size: Tuple[int, int]) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = F.interpolate(pred_grid, size=size, mode="bilinear", align_corners=False)
    vis = F.interpolate(vis_grid.float(), size=size, mode="nearest").bool()
    return pred, vis


def silog_loss(pred_log: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    d = (pred_log - torch.log(target.clamp(min=1e-6)))[mask]
    if d.numel() == 0:
        return pred_log.sum() * 0.0
    return (d ** 2).mean() - SILOG_LAMBDA * d.mean() ** 2


@torch.no_grad()
def depth_metrics(pred: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> Dict[str, float]:
    p, t = pred[mask], target[mask]
    thresh = torch.maximum(p / t, t / p)
    return {
        "rmse": torch.sqrt(((p - t) ** 2).mean()).item(),
        "absrel": ((p - t).abs() / t).mean().item(),
        "delta1": (thresh < 1.25).float().mean().item(),
    }


def _loaders(cfg: dict) -> Tuple[DataLoader, DataLoader]:
    d = cfg["depth"]

    def make(split: str, shuffle: bool) -> DataLoader:
        ds = NYUDepth(cfg["paths"]["data"], split, d["resolution"], d["depth_clip"])
        return DataLoader(ds, batch_size=d["batch_size"], shuffle=shuffle,
                          num_workers=d["num_workers"])

    return make("train", True), make("test", False)


def _forward(encoder: MaskedDinoV2, head: LinearDepthHead, batch: tuple,
             keep_ratio: float, mask_seed: int, full_image: bool,
             device: torch.device):
    img, depth, ids = batch
    img, depth = img.to(device), depth.to(device)
    out = encoder(img, keep_ratio=keep_ratio, mask_seed=mask_seed, sample_ids=ids)
    pred_grid, vis_grid = head(out["patch"], out["kept_idx"], out["grid"], full_image)
    pred_log, vis = _upsample(pred_grid, vis_grid, depth.shape[-2:])
    valid = depth > 0
    mask = valid if full_image else (valid & vis)
    return pred_log, depth, mask


def run(encoder: MaskedDinoV2, cfg: dict, keep_ratio: float, mask_seed: int,
        device: torch.device, head: LinearDepthHead = None):
    """Train (unless a head is given) and evaluate. -> (metrics, head, n_epochs)."""
    d = cfg["depth"]
    full_image = d["eval_mode"] == "full_image"
    train_loader, test_loader = _loaders(cfg)
    n_epochs = 0

    if head is None:
        head = LinearDepthHead(encoder.embed_dim)

        def step(h, batch, dev):
            pred_log, depth, mask = _forward(encoder, h, batch, keep_ratio,
                                             mask_seed, full_image, dev)
            return silog_loss(pred_log, depth, mask)

        tag = f"depth kr={keep_ratio} seed={mask_seed}"
        head = train_head(head, train_loader, step, d["epochs"], d["lr"], device, tag)
        n_epochs = d["epochs"]

    head.to(device).eval()
    lo, hi = d["depth_clip"]
    agg = {"rmse": 0.0, "absrel": 0.0, "delta1": 0.0}
    n_batches = 0
    with torch.no_grad():
        for batch in test_loader:
            pred_log, depth, mask = _forward(encoder, head, batch, keep_ratio,
                                             mask_seed, full_image, device)
            pred = pred_log.exp().clamp(lo, hi)
            for name, value in depth_metrics(pred, depth, mask).items():
                agg[name] += value
            n_batches += 1
    metrics = {k: v / max(n_batches, 1) for k, v in agg.items()}
    return metrics, head, n_epochs
