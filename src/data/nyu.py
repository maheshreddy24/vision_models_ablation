"""NYU Depth v2 labeled subset: download + Dataset.

Single-file download (~2.8 GB): nyu_depth_v2_labeled.mat (HDF5), 1449 aligned
RGB (480x640) + depth (meters) pairs. Official split (splits.mat): 795 train /
654 test. The same .mat also contains semantic labels (future segmentation task).
"""
import os
import urllib.request
from typing import Tuple

import h5py
import numpy as np
import torch
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

LABELED_MAT_URL = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
SPLITS_MAT_URL = "http://horatio.cs.nyu.edu/mit/silberman/indoor_seg_sup/splits.mat"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _download(url: str, dst: str) -> None:
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        print(f"[nyu] {os.path.basename(dst)} already present, skipping.")
        return
    print(f"[nyu] downloading {url} -> {dst}")
    tmp = dst + ".part"

    def _progress(count: int, block: int, total: int) -> None:
        if total > 0 and count % 500 == 0:
            print(f"\r[nyu] {count * block / total * 100:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, tmp, reporthook=_progress)
    print()
    os.replace(tmp, dst)


def download(data_root: str) -> None:
    nyu_dir = os.path.join(data_root, "nyu")
    os.makedirs(nyu_dir, exist_ok=True)
    _download(LABELED_MAT_URL, os.path.join(nyu_dir, "nyu_depth_v2_labeled.mat"))
    _download(SPLITS_MAT_URL, os.path.join(nyu_dir, "splits.mat"))


class NYUDepth(Dataset):
    """Yields (image (3,H,W) ImageNet-normalized, depth (1,H,W) meters, sample_id).

    Depth is resized to the model input resolution; invalid pixels (<= 0 or
    outside depth_clip) are set to 0 and must be masked out downstream.
    Use num_workers=0: data is read from a single open h5 handle.
    """

    def __init__(self, data_root: str, split: str, resolution: Tuple[int, int],
                 depth_clip: Tuple[float, float]):
        assert split in ("train", "test")
        nyu_dir = os.path.join(data_root, "nyu")
        self.h5 = h5py.File(os.path.join(nyu_dir, "nyu_depth_v2_labeled.mat"), "r")
        splits = loadmat(os.path.join(nyu_dir, "splits.mat"))
        key = "trainNdxs" if split == "train" else "testNdxs"
        self.indices = splits[key].squeeze().astype(int) - 1  # 1-indexed in the .mat
        self.resolution = tuple(resolution)
        self.depth_clip = tuple(depth_clip)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        # Stored transposed: images (N, 3, W, H), depths (N, W, H).
        img = np.transpose(self.h5["images"][idx], (2, 1, 0))    # (480, 640, 3) uint8
        depth = np.transpose(self.h5["depths"][idx], (1, 0))     # (480, 640) float, meters

        img_t = torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255.0
        img_t = TF.resize(img_t, list(self.resolution), antialias=True)
        img_t = TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)

        depth_t = torch.from_numpy(depth.copy()).float().unsqueeze(0)
        depth_t = TF.resize(depth_t, list(self.resolution), antialias=True)
        lo, hi = self.depth_clip
        depth_t = torch.where((depth_t >= lo) & (depth_t <= hi), depth_t,
                              torch.zeros_like(depth_t))
        return img_t, depth_t, idx
