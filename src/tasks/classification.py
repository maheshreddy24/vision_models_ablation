"""Linear probe on the CLS token, Oxford-IIIT Pets.

Since the backbone is frozen and masks are deterministic per
(keep_ratio, mask_seed, sample_id), CLS features are extracted once per run
configuration and cached to disk - head training then takes seconds.
"""
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from ..data.pets import NUM_CLASSES, OxfordPets
from ..models.masked_dinov2 import MaskedDinoV2
from .base import train_head

TASK_NAME = "classification"


def _extract_cls(encoder: MaskedDinoV2, cfg: dict, split: str, keep_ratio: float,
                 mask_seed: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    c = cfg["classification"]
    cache_dir = cfg["paths"]["cache"]
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"pets_{split}_kr{keep_ratio:.2f}_seed{mask_seed}.pt")
    if c["cache_features"] and os.path.exists(cache_path):
        blob = torch.load(cache_path)
        return blob["features"], blob["labels"]

    ds = OxfordPets(cfg["paths"]["data"], split, c["resolution"],
                    c["dataset"], c.get("fallback_dataset"))
    loader = DataLoader(ds, batch_size=c["batch_size"], shuffle=False,
                        num_workers=c["num_workers"])
    feats, labels = [], []
    with torch.no_grad():
        for img, y, ids in tqdm(loader, desc=f"[cls] features {split} kr={keep_ratio}",
                                leave=False):
            out = encoder(img.to(device), keep_ratio=keep_ratio,
                          mask_seed=mask_seed, sample_ids=ids)
            feats.append(out["cls"].cpu())
            labels.append(torch.as_tensor(y))
    features, labels = torch.cat(feats), torch.cat(labels)
    if c["cache_features"]:
        torch.save({"features": features, "labels": labels}, cache_path)
    return features, labels


def run(encoder: MaskedDinoV2, cfg: dict, keep_ratio: float, mask_seed: int,
        device: torch.device, head: nn.Linear = None):
    """Train (unless a head is given) and evaluate. -> (metrics, head, n_epochs)."""
    c = cfg["classification"]
    n_epochs = 0

    if head is None:
        f_tr, y_tr = _extract_cls(encoder, cfg, "train", keep_ratio, mask_seed, device)
        loader = DataLoader(TensorDataset(f_tr, y_tr),
                            batch_size=c["batch_size"], shuffle=True)
        head = nn.Linear(encoder.embed_dim, NUM_CLASSES)

        def step(h, batch, dev):
            f, y = batch
            return F.cross_entropy(h(f.to(dev)), y.to(dev))

        tag = f"cls kr={keep_ratio} seed={mask_seed}"
        head = train_head(head, loader, step, c["epochs"], c["lr"], device, tag)
        n_epochs = c["epochs"]

    head.to(device).eval()
    f_te, y_te = _extract_cls(encoder, cfg, "test", keep_ratio, mask_seed, device)
    with torch.no_grad():
        pred = head(f_te.to(device)).argmax(dim=1).cpu()
    metrics: Dict[str, float] = {"top1": (pred == y_te).float().mean().item()}
    return metrics, head, n_epochs
