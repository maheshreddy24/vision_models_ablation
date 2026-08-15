"""Shared utilities: config loading, seeding, resumable CSV result logging."""
import csv
import os
import random
import time
from typing import Dict, List, Set, Tuple

import numpy as np
import torch
import yaml

RESULT_FIELDS = [
    "task", "protocol", "keep_ratio", "mask_seed",
    "metric_name", "value", "n_epochs", "timestamp",
]

RunKey = Tuple[str, str, float, int]  # (task, protocol, keep_ratio, mask_seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(cfg: dict) -> torch.device:
    want = cfg.get("device", "cuda")
    if want == "cuda" and not torch.cuda.is_available():
        print("[utils] CUDA not available -> falling back to CPU.")
        return torch.device("cpu")
    return torch.device(want)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def append_results(csv_path: str, rows: List[Dict]) -> None:
    """Append result rows, writing a header if the file is new."""
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        for row in rows:
            row = {**row, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            writer.writerow(row)


def completed_runs(csv_path: str) -> Set[RunKey]:
    """Run keys already present in the CSV (used to resume an interrupted sweep)."""
    done: Set[RunKey] = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            done.add((row["task"], row["protocol"],
                      float(row["keep_ratio"]), int(row["mask_seed"])))
    return done


def metrics_to_rows(task: str, protocol: str, keep_ratio: float, mask_seed: int,
                    metrics: Dict[str, float], n_epochs: int) -> List[Dict]:
    return [
        {
            "task": task, "protocol": protocol, "keep_ratio": keep_ratio,
            "mask_seed": mask_seed, "metric_name": name, "value": float(value),
            "n_epochs": n_epochs, "timestamp": "",
        }
        for name, value in metrics.items()
    ]
