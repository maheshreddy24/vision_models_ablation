"""Plots: % patches kept vs performance (depth delta1 + Pets top-1, twin axes)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _agg(df: pd.DataFrame, task: str, metric: str) -> pd.DataFrame:
    sub = df[(df.task == task) & (df.metric_name == metric)]
    return (sub.groupby("keep_ratio")["value"]
            .agg(["mean", "std"]).fillna(0.0).reset_index()
            .sort_values("keep_ratio"))


def make_plots(csv_path: str, results_dir: str, protocol: str, backbone: str = "") -> None:
    df = pd.read_csv(csv_path)
    df = df[df.protocol == protocol]
    if df.empty:
        print(f"[plot] no rows for protocol '{protocol}' in {csv_path}; nothing to plot.")
        return
    os.makedirs(results_dir, exist_ok=True)

    depth = _agg(df, "depth", "delta1")
    cls = _agg(df, "classification", "top1")

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()

    if not depth.empty:
        x = depth.keep_ratio * 100
        ax1.plot(x, depth["mean"], "o-", color="tab:blue", label=r"Depth $\delta<1.25$")
        ax1.fill_between(x, depth["mean"] - depth["std"], depth["mean"] + depth["std"],
                         color="tab:blue", alpha=0.15)
        ax1.axhline(depth["mean"].iloc[-1], ls=":", color="tab:blue", alpha=0.5)
    if not cls.empty:
        x = cls.keep_ratio * 100
        ax2.plot(x, cls["mean"], "s--", color="tab:red", label="Pets top-1")
        ax2.fill_between(x, cls["mean"] - cls["std"], cls["mean"] + cls["std"],
                         color="tab:red", alpha=0.15)
        ax2.axhline(cls["mean"].iloc[-1], ls=":", color="tab:red", alpha=0.5)

    ax1.set_xlabel("% patches kept")
    ax1.set_ylabel(r"Depth $\delta<1.25$", color="tab:blue")
    ax2.set_ylabel("Pets top-1 accuracy", color="tab:red")
    title_backbone = backbone or "DINOv2-S/14"
    ax1.set_title(f"{title_backbone} patch-budget ablation ({protocol})")
    lines = ax1.get_lines()[:1] + ax2.get_lines()[:1]
    ax1.legend(lines, [ln.get_label() for ln in lines], loc="lower right")
    fig.tight_layout()
    out = os.path.join(results_dir, "patch_budget_ablation.png")
    fig.savefig(out, dpi=300)
    print(f"[plot] saved {out}")

    # Secondary figure: rmse + absrel (lower is better).
    fig2, ax = plt.subplots(figsize=(7, 4.5))
    for metric, color in [("rmse", "tab:green"), ("absrel", "tab:purple")]:
        agg = _agg(df, "depth", metric)
        if agg.empty:
            continue
        x = agg.keep_ratio * 100
        ax.plot(x, agg["mean"], "o-", color=color, label=metric.upper())
        ax.fill_between(x, agg["mean"] - agg["std"], agg["mean"] + agg["std"],
                        color=color, alpha=0.15)
    ax.set_xlabel("% patches kept")
    ax.set_ylabel("Error (lower is better)")
    ax.set_title("Depth error vs patch budget")
    ax.legend()
    fig2.tight_layout()
    out2 = os.path.join(results_dir, "depth_errors.png")
    fig2.savefig(out2, dpi=300)
    print(f"[plot] saved {out2}")
