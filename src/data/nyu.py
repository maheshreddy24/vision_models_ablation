"""NYU Depth v2 via the HuggingFace mirror `jagennath-hari/nyuv2`.

The original horatio.cs.nyu.edu server hosting nyu_depth_v2_labeled.mat is
unreliable/down, so we use a HF-native parquet mirror of the same 1449 labeled
pairs (MIT license, no remote code). Splits: 1014 train / 290 val / 145 test.
Depth is stored as a scaled 16-bit PNG; dividing by `depth_scale` from the
repo's scaling_factors.json recovers meters. Semantic + instance masks are
also included (future segmentation task, zero extra downloads).

Note: this split is NOT the official 795/654 Eigen split - fine for a
relative ablation, not comparable to published NYU numbers.
"""
import json
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

DATASET_ID = "jagennath-hari/nyuv2"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _cache_dir(data_root: str) -> str:
    return os.path.join(data_root, "nyu")


def _load_hf(data_root: str):
    from datasets import load_dataset
    return load_dataset(DATASET_ID, cache_dir=_cache_dir(data_root))


def _load_depth_scale(data_root: str) -> float:
    """depth_png / depth_scale = depth in meters (from scaling_factors.json)."""
    from huggingface_hub import snapshot_download
    local_dir = snapshot_download(
        repo_id=DATASET_ID, repo_type="dataset",
        allow_patterns="scaling_factors.json", cache_dir=_cache_dir(data_root))
    with open(os.path.join(local_dir, "scaling_factors.json")) as f:
        return float(json.load(f)["depth_scale"])


def download(data_root: str) -> None:
    _load_hf(data_root)
    _load_depth_scale(data_root)
    print("[nyu] HF mirror cached.")


class NYUDepth(Dataset):
    """Yields (image (3,H,W) ImageNet-normalized, depth (1,H,W) meters, sample_id).

    Invalid depth pixels (<= 0 or outside depth_clip) are set to 0 and must be
    masked out downstream. num_workers > 0 is fine (no shared file handles).
    """

    def __init__(self, data_root: str, split: str, resolution: Tuple[int, int],
                 depth_clip: Tuple[float, float]):
        assert split in ("train", "val", "test")
        self.ds = _load_hf(data_root)[split]
        self.depth_scale = _load_depth_scale(data_root)
        self.resolution = tuple(resolution)
        self.depth_clip = tuple(depth_clip)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        row = self.ds[i]

        img_t = TF.to_tensor(row["rgb"].convert("RGB"))
        img_t = TF.resize(img_t, list(self.resolution), antialias=True)
        img_t = TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)

        depth = np.asarray(row["depth"]).astype(np.float32) / self.depth_scale
        depth_t = torch.from_numpy(depth).unsqueeze(0)
        depth_t = TF.resize(depth_t, list(self.resolution), antialias=True)
        lo, hi = self.depth_clip
        depth_t = torch.where((depth_t >= lo) & (depth_t <= hi), depth_t,
                              torch.zeros_like(depth_t))
        return img_t, depth_t, i