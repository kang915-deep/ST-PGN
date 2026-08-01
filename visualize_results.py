"""Create experiment plots from artifacts saved by train.py."""

from pathlib import Path

import numpy as np


def _load_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def save_experiment_plots(output_dir: str) -> None:
    """Save compact diagnostic plots for one ST-PGN run."""
    output = Path(output_dir)
    history_path = output / "history.npz"
    predictions_path = output / "predictions.npz"
    if not history_path.exists() or not predictions_path.exists():
        return
    try:
        plt = _load_matplotlib()
    except ImportError:
        print("matplotlib is not installed; skipped experiment plots")
        return

    history = np.load(history_path)
    data = np.load(predictions_path)
    epochs = history["epoch"]
    pred = data["prediction"]
    target = data["target"]
    residual = pred - target

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="train loss")
    axes[0].plot(epochs, history["val_rmse"], label="val RMSE")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("value")
    axes[0].set_title("Training and validation")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].scatter(target, pred, s=14, alpha=0.65)
    lower = float(min(target.min(), pred.min()))
    upper = float(max(target.max(), pred.max()))
    axes[1].plot([lower, upper], [lower, upper], "--", color="black")
    axes[1].set_xlabel("true RUL")
    axes[1].set_ylabel("predicted RUL")
    axes[1].set_title("Test prediction")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "training_and_prediction.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(residual, bins=20, alpha=0.8)
    axes[0].axvline(0.0, linestyle="--", color="black")
    axes[0].set_xlabel("prediction error")
    axes[0].set_ylabel("count")
    axes[0].set_title("Test residual distribution")
    units = data["unit"]
    cycles = data["cycle"]
    for unit in np.unique(units)[:6]:
        mask = units == unit
        order = np.argsort(cycles[mask])
        axes[1].plot(cycles[mask][order], target[mask][order], label=f"engine {unit}")
        axes[1].plot(cycles[mask][order], pred[mask][order], "--", alpha=0.8)
    axes[1].set_xlabel("cycle")
    axes[1].set_ylabel("RUL")
    axes[1].set_title("Engine degradation trajectories")
    axes[1].legend(fontsize=8, ncol=2)
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "residuals_and_trajectories.png", dpi=160)
    plt.close(fig)

    adjacency_path = output / "adjacency.npy"
    if adjacency_path.exists():
        adjacency = np.load(adjacency_path)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        image = ax.imshow(adjacency, aspect="auto", cmap="viridis")
        ax.set_xlabel("source sensor")
        ax.set_ylabel("target sensor")
        ax.set_title("Learned sensor adjacency")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(output / "sensor_adjacency.png", dpi=160)
        plt.close(fig)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir")
    save_experiment_plots(parser.parse_args().output_dir)
