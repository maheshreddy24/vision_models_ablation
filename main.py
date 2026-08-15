"""Unified runner for the DINOv2-registers patch-budget ablation.

    python main.py                                # everything: download -> sweep -> plot
    python main.py --stages sweep,plot            # skip downloads
    python main.py --tasks classification         # only the linear-probe task
    python main.py --config config.yaml

The sweep is resumable: completed (task, protocol, keep_ratio, mask_seed) runs
already in results/results.csv are skipped on re-run.
"""
import argparse
import os

from src import plotting, utils
from src.data import nyu, pets
from src.models.masked_dinov2 import MaskedDinoV2, load_backbone
from src.tasks import classification, depth

TASK_MODULES = {"depth": depth, "classification": classification}


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


def stage_sweep(cfg: dict, tasks: list, device) -> str:
    csv_path = os.path.join(cfg["paths"]["results"], "results.csv")
    done = utils.completed_runs(csv_path)
    protocol = cfg["protocol"]

    encoder = MaskedDinoV2(load_backbone(cfg["backbone"])).to(device)

    for task_name in tasks:
        module = TASK_MODULES[task_name]
        full_head = None
        if protocol == "reuse_full_head":
            print(f"[sweep] {task_name}: training head once at keep_ratio=1.0")
            _, full_head, _ = module.run(encoder, cfg, 1.0, cfg["mask_seeds"][0], device)

        for keep_ratio, mask_seed in _budget_grid(cfg):
            key = (task_name, protocol, keep_ratio, mask_seed)
            if key in done:
                print(f"[sweep] skip (done): {key}")
                continue
            print(f"[sweep] run: task={task_name} kr={keep_ratio} seed={mask_seed}")
            utils.set_seed(cfg["seed"])
            metrics, _, n_epochs = module.run(
                encoder, cfg, keep_ratio, mask_seed, device, head=full_head)
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
    args = parser.parse_args()

    cfg = utils.load_config(args.config)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for t in tasks:
        assert t in TASK_MODULES, f"unknown task '{t}'"
    device = utils.resolve_device(cfg)
    os.makedirs(cfg["paths"]["results"], exist_ok=True)

    if "download" in stages:
        stage_download(cfg, tasks)
    csv_path = os.path.join(cfg["paths"]["results"], "results.csv")
    if "sweep" in stages:
        csv_path = stage_sweep(cfg, tasks, device)
    if "plot" in stages:
        plotting.make_plots(csv_path, cfg["paths"]["results"], cfg["protocol"])


if __name__ == "__main__":
    main()
