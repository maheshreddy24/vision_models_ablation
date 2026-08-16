"""Oxford-IIIT Pets (37 classes) via HuggingFace `datasets`."""
import os
from typing import Optional

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .nyu import IMAGENET_MEAN, IMAGENET_STD

NUM_CLASSES = 37


def _load_hf(dataset_id: str, fallback_id: Optional[str], cache_dir: str):
    from datasets import load_dataset
    try:
        return load_dataset(dataset_id, cache_dir=cache_dir)
    except Exception as e:  # dataset id moved/renamed
        if fallback_id is None:
            raise
        print(f"[pets] '{dataset_id}' failed ({e}); trying fallback '{fallback_id}'.")
        return load_dataset(fallback_id, cache_dir=cache_dir)


def download(data_root: str, dataset_id: str, fallback_id: Optional[str] = None) -> None:
    _load_hf(dataset_id, fallback_id, os.path.join(data_root, "pets"))
    print("[pets] dataset cached.")


class OxfordPets(Dataset):
    """Yields (image (3,R,R) ImageNet-normalized, label, sample_id)."""

    def __init__(self, data_root: str, split: str, resolution: int,
                 dataset_id: str, fallback_id: Optional[str] = None):
        assert split in ("train", "test")
        ds = _load_hf(dataset_id, fallback_id, os.path.join(data_root, "pets"))
        self.ds = ds[split if split in ds else {"train": "train", "test": "test"}[split]]
        self.resolution = resolution
        cols = self.ds.column_names
        self.image_col = "image" if "image" in cols else cols[0]
        self.label_col = "label" if "label" in cols else "labels"

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        row = self.ds[i]
        img = row[self.image_col].convert("RGB")
        img_t = TF.to_tensor(img)
        img_t = TF.resize(img_t, [self.resolution, self.resolution], antialias=True)
        img_t = TF.normalize(img_t, IMAGENET_MEAN, IMAGENET_STD)
        return img_t, int(row[self.label_col]), i
