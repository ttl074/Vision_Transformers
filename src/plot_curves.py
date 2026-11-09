"""
Plot training curves for the trained models from their CSV training logs.

Reads ``logs/<run name>_training_log.csv`` for each run (written by
src.train) and saves two figures to --fig_dir:

- ``training_curves.png``: train and validation loss and accuracy per
  epoch, all runs overlaid, with each run's best validation-loss epoch (the
  checkpoint evaluated on the test set) marked on the validation panels.
- ``lr_schedule.png``: the learning rate per epoch (linear warmup, then
  cosine decay), the same for every run.

Train accuracy is measured on MixUp/CutMix-mixed images against their
original labels, so it runs below validation accuracy.

Run from the repository root:
    python -m src.plot_curves
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

# Display name and color for each run; colors match the W&B charts.
RUNS = {
    "hybrid_seed2026": ("Hybrid (ResNet-56 + Transformer)", "tab:red"),
    "vit_seed2026": ("Plain ViT", "tab:blue"),
    "swin_seed2026": ("Swin", "tab:green"),
}

# (CSV column, panel title, y-axis label) for each panel, row by row.
PANELS = [
    ("train/loss", "Train loss", "Cross-entropy (label-smoothed)"),
    ("val/loss", "Validation loss", "Cross-entropy"),
    ("train/acc", "Train accuracy (mixed images)", "Top-1 accuracy"),
    ("val/acc", "Validation accuracy", "Top-1 accuracy"),
]


def load_logs(log_dir):
    """
    Read each run's training log.

    Runs whose log file is missing are skipped with a message, so the figure
    can be made before every model has finished training.

    Args:
        log_dir (str): Directory containing the ``*_training_log.csv`` files.

    Returns:
        dict[str, pd.DataFrame]: Training log of each available run, keyed by
        run name, in the order of ``RUNS``.
    """
    logs = {}
    for run in RUNS:
        path = os.path.join(log_dir, f"{run}_training_log.csv")
        if os.path.exists(path):
            logs[run] = pd.read_csv(path)
        else:
            print(f"Skipping {run}: no {path}")
    return logs


def plot_training_curves(logs, path):
    """
    Save a 2x2 grid of train/validation loss and accuracy curves.

    Each run is drawn in its own color. On the validation panels, a dot marks
    the epoch with the lowest validation loss, whose checkpoint (best.pt) was
    evaluated on the test set.

    Args:
        logs (dict[str, pd.DataFrame]): Training log of each run.
        path (str): Output image path.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, (column, title, ylabel) in zip(axes.flat, PANELS):
        for run, log in logs.items():
            label, color = RUNS[run]
            ax.plot(log["epoch"], log[column], color=color, linewidth=1.2, label=label)
            if column.startswith("val/"):
                best = log.loc[log["val/loss"].idxmin()]
                ax.scatter(best["epoch"], best[column], color=color, s=30, zorder=3)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    axes[0, 0].legend(title="Dots: best val-loss epoch", fontsize=9, title_fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_lr_schedule(logs, path):
    """
    Save the learning rate per epoch for one run.

    All runs share the same schedule, so one log is enough; the longest is
    used, since a run that stopped early ends before the schedule does.

    Args:
        logs (dict[str, pd.DataFrame]): Training log of each run.
        path (str): Output image path.
    """
    log = max(logs.values(), key=len)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(log["epoch"], log["lr"], color="black", linewidth=1.2)
    ax.set_title("Learning-rate schedule (all runs)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate at end of epoch")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Log and figure directories.
    """
    parser = argparse.ArgumentParser(description='Plot training curves from CSV logs')
    parser.add_argument('--log_dir', default='logs', help='Directory with the training logs')
    parser.add_argument('--fig_dir', default='figures', help='Directory for the figures')
    return parser.parse_args()


def main(args):
    """
    Make both figures from the training logs.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    logs = load_logs(args.log_dir)
    if not logs:
        raise SystemExit(f"No training logs found in {args.log_dir}")
    os.makedirs(args.fig_dir, exist_ok=True)
    plot_training_curves(logs, os.path.join(args.fig_dir, "training_curves.png"))
    plot_lr_schedule(logs, os.path.join(args.fig_dir, "lr_schedule.png"))


if __name__ == "__main__":
    main(parse_args())
