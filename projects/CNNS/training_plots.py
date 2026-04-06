from pathlib import Path
import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_history(
    history,
    output_dir,
    prefix="lenet_digits",
    title_prefix="LeNet Digits",
):
    """Save loss and accuracy curves for the current training history."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    epochs = history["epochs"]

    loss_fig, loss_ax = plt.subplots(figsize=(8, 5))
    loss_ax.plot(epochs, history["train_loss"], marker="o", linewidth=2, label="Train Loss")
    loss_ax.plot(epochs, history["test_loss"], marker="s", linewidth=2, label="Test Loss")
    loss_ax.set_title(f"{title_prefix} Loss per Epoch")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_xticks(epochs)
    loss_ax.grid(True, linestyle="--", alpha=0.4)
    loss_ax.legend()
    loss_fig.tight_layout()
    loss_fig.savefig(output_path / f"{prefix}_loss.png", dpi=150)
    plt.close(loss_fig)

    acc_fig, acc_ax = plt.subplots(figsize=(8, 5))
    acc_ax.plot(epochs, history["train_acc"], marker="o", linewidth=2, label="Train Accuracy")
    acc_ax.plot(epochs, history["test_acc"], marker="s", linewidth=2, label="Test Accuracy")
    acc_ax.set_title(f"{title_prefix} Accuracy per Epoch")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_xticks(epochs)
    acc_ax.set_ylim(0.0, 1.0)
    acc_ax.grid(True, linestyle="--", alpha=0.4)
    acc_ax.legend()
    acc_fig.tight_layout()
    acc_fig.savefig(output_path / f"{prefix}_accuracy.png", dpi=150)
    plt.close(acc_fig)


def save_training_metrics(history, output_dir, prefix, summary=None):
    """Persist epoch metrics and optional benchmark summary data."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics_file = output_path / f"{prefix}_metrics.csv"
    with metrics_file.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "test_loss", "train_acc", "test_acc"])
        for idx, epoch in enumerate(history["epochs"]):
            writer.writerow(
                [
                    epoch,
                    history["train_loss"][idx],
                    history["test_loss"][idx],
                    history["train_acc"][idx],
                    history["test_acc"][idx],
                ]
            )

    if summary is not None:
        summary_file = output_path / f"{prefix}_summary.json"
        with summary_file.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
