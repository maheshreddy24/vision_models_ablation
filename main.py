"""Unified runner for the DINOv2 patch-budget ablation.

    python main.py                                # everything: download -> sweep -> plot
    python main.py --stages sweep,plot            # skip downloads
    python main.py --tasks classification         # only the linear-probe task
    python main.py --config config.yaml
    python main.py --workers 6                    # sweep entries run in parallel processes

The sweep is resumable: completed (task, protocol, keep_ratio, mask_seed) runs
already in results/results.csv are skipped on re-run.

With --workers > 1, sweep entries (task, keep_ratio, mask_seed) are farmed out
to a process pool. Each worker process loads its own copy of the frozen
backbone once (via an initializer) and reuses it across every job it picks up.
Since the
 backbone never trains, this is safe and memory is the only real
constraint (ViT-S/14 is small: a handful of workers is cheap RAM-wise).
"""
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Optional

import torch

from src import plotting, utils
from src.data import nyu, pets
from src.models.masked_dinov2 import MaskedDinoV2, load_backbone
from src.tasks import classification, depth

TASK_MODULES = {"depth": depth, "classification": classification}

_worker: Dict[str, object] = {}  # per-process globals, populated by _init_worker


def stage_download(cfg: dict, tasks: list) -> None:
    data_root = cfg["paths"]["data"]
    if "depth" in tasks:
        nyu.download(data_root)
    if "classification" in tasks:
        c = cfg["classification"]
        pets.download(data_root, c["dataset"], c.get("fallback_dataset"))


def _budget_grid(cfg: dict):
    """(keep_ratio, mask_seed) pairs; at 1.0 the mask is a no-op -> one seed only."""
    for kr in cfg["budgets"]:
        seeds = cfg["mask_seeds"][:1] if kr >= 1.0 else cfg["mask_seeds"]
        for ms in seeds:
            yield float(kr), int(ms)


def _train_full_heads(cfg: dict, tasks: list, device) -> Dict[str, dict]:
    """protocol=reuse_full_head: train each task's head once at keep_ratio=1.0.

    Returns {task_name: cpu state_dict} so workers can rebuild the head
    without shipping the (already-loaded) encoder across the process boundary.
    """
    encoder = MaskedDinoV2(load_backbone(cfg["backbone"])).to(device)
    states = {}
    for task_name in tasks:
        module = TASK_MODULES[task_name]
        print(f"[sweep] {task_name}: training head once at keep_ratio=1.0")
        _, full_head, _ = module.run(encoder, cfg, 1.0, cfg["mask_seeds"][0], device)
        states[task_name] = {k: v.cpu() for k, v in full_head.state_dict().items()}
    return states


def _pending_jobs(cfg: dict, tasks: list, done: set) -> list:
    protocol = cfg["protocol"]
    jobs = []
    for task_name in tasks:
        for keep_ratio, mask_seed in _budget_grid(cfg):
            key = (task_name, protocol, keep_ratio, mask_seed)
            if key in done:
                print(f"[sweep] skip (done): {key}")
                continue
            jobs.append((task_name, keep_ratio, mask_seed))
    return jobs


def _run_one(module, encoder, cfg: dict, keep_ratio: float, mask_seed: int,
            device, head_state: Optional[dict]):
    utils.set_seed(cfg["seed"])
    head = None
    if head_state is not None:
        head = module.build_head(encoder.embed_dim).to(device)
        head.load_state_dict(head_state)
    return module.run(encoder, cfg, keep_ratio, mask_seed, device, head=head)


def _init_worker(backbone_name: str, device_str: str) -> None:
    """ProcessPoolExecutor initializer: load one backbone copy per worker, reused
    across every job that worker picks up (loading it per-job would dwarf the
    training cost)."""
    torch.set_num_threads(1)  # avoid oversubscribing CPUs across worker processes
    device = torch.device(device_str)
    _worker["encoder"] = MaskedDinoV2(load_backbone(backbone_name)).to(device)
    _worker["device"] = device


def _run_job(task_name: str, cfg: dict, keep_ratio: float, mask_seed: int,
            head_state: Optional[dict]):
    module = TASK_MODULES[task_name]
    metrics, _, n_epochs = _run_one(
        module, _worker["encoder"], cfg, keep_ratio, mask_seed, _worker["device"], head_state)
    return task_name, keep_ratio, mask_seed, metrics, n_epochs


def stage_sweep(cfg: dict, tasks: list, device, workers: int = 1) -> str:
    csv_path = os.path.join(cfg["paths"]["results"], "results.csv")
    done = utils.completed_runs(csv_path)
    protocol = cfg["protocol"]

    head_states: Dict[str, dict] = {}
    if protocol == "reuse_full_head":
        head_states = _train_full_heads(cfg, tasks, device)

    jobs = _pending_jobs(cfg, tasks, done)
    if not jobs:
        return csv_path

    if workers <= 1:
        encoder = MaskedDinoV2(load_backbone(cfg["backbone"])).to(device)
        for task_name, keep_ratio, mask_seed in jobs:
            print(f"[sweep] run: task={task_name} kr={keep_ratio} seed={mask_seed}")
            metrics, _, n_epochs = _run_one(
                TASK_MODULES[task_name], encoder, cfg, keep_ratio, mask_seed, device,
                head_states.get(task_name))
            utils.append_results(csv_path, utils.metrics_to_rows(
                task_name, protocol, keep_ratio, mask_seed, metrics, n_epochs))
            print(f"[sweep] {(task_name, protocol, keep_ratio, mask_seed)} -> {metrics}")
        return csv_path

    print(f"[sweep] running {len(jobs)} jobs across {workers} worker processes")
    load_backbone(cfg["backbone"])  # warm torch.hub cache once; avoids a download
                                     # race between worker processes below
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(cfg["backbone"], str(device))) as pool:
        futures = {
            pool.submit(_run_job, task_name, cfg, keep_ratio, mask_seed,
                        head_states.get(task_name)): (task_name, keep_ratio, mask_seed)
            for task_name, keep_ratio, mask_seed in jobs
        }
        for fut in as_completed(futures):
            task_name, keep_ratio, mask_seed = futures[fut]
            _, _, _, metrics, n_epochs = fut.result()
            key = (task_name, protocol, keep_ratio, mask_seed)
            utils.append_results(csv_path, utils.metrics_to_rows(
                task_name, protocol, keep_ratio, mask_seed, metrics, n_epochs))
            print(f"[sweep] {key} -> {metrics}")
    return csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stages", default="download,sweep,plot",
                        help="comma-separated subset of: download,sweep,plot")
    parser.add_argument("--tasks", default="depth,classification",
                        help="comma-separated subset of: depth,classification")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel worker processes for the sweep stage. "
                             "Each worker loads its own frozen backbone copy and stays "
                             "warm across every job it's assigned, so extra workers cost "
                             "one backbone's worth of RAM apiece and roughly divide sweep "
                             "wall-clock time by the worker count. Default 1 = sequential.")
    args = parser.parse_args()

    cfg = utils.load_config(args.config)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        assert t in TASK_MODULES, f"unknown task '{t}'"
    assert args.workers >= 1, "--workers must be >= 1"
    device = utils.resolve_device(cfg)
    os.makedirs(cfg["paths"]["results"], exist_ok=True)

    if "download" in stages:
        stage_download(cfg, tasks)
    csv_path = os.path.join(cfg["paths"]["results"], "results.csv")
    if "sweep" in stages:
        csv_path = stage_sweep(cfg, tasks, device, args.workers)
    if "plot" in stages:
        plotting.make_plots(csv_path, cfg["paths"]["results"], cfg["protocol"], cfg["backbone"])


if __name__ == "__main__":
    main()
