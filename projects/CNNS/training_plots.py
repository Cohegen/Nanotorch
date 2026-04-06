from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_training_history(history, output_dir, prefix="lenet_digits"):
    """Save loss and accuracy curves for the current training history."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    epochs = history["epochs"]

    loss_fig, loss_ax = plt.subplots(figsize=(8, 5))
    loss_ax.plot(epochs, history["train_loss"], marker="o", linewidth=2, label="Train Loss")
    loss_ax.plot(epochs, history["test_loss"], marker="s", linewidth=2, label="Test Loss")
    loss_ax.set_title("LeNet Digits Loss per Epoch")
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
    acc_ax.set_title("LeNet Digits Accuracy per Epoch")
    acc_ax.set_xlabel("Epoch")
    acc_ax.set_ylabel("Accuracy")
    acc_ax.set_xticks(epochs)
    acc_ax.set_ylim(0.0, 1.0)
    acc_ax.grid(True, linestyle="--", alpha=0.4)
    acc_ax.legend()
    acc_fig.tight_layout()
    acc_fig.savefig(output_path / f"{prefix}_accuracy.png", dpi=150)
    plt.close(acc_fig)
