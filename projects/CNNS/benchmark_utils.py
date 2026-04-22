import time
import os
import sys 
from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import numpy as np

from Tensor import Tensor
from losses.losses import CrossEntropyLoss
from optimizers.optimizers import SGD
from training_plots import plot_training_history, save_training_metrics

from nanotorchvision.datasets import load_nanodigits, make_nanodigits_loader
from nanotorchvision.models import count_parameters


def train_epoch(model, dataloader, optimizer, loss_fn):
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in dataloader:
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        batch_labels = targets.data.astype(np.int64)
        batch_size_actual = len(batch_labels)
        total_loss += float(loss.data) * batch_size_actual
        total_correct += int(np.sum(np.argmax(logits.data, axis=1) == batch_labels))
        total_examples += batch_size_actual

    return total_loss / total_examples, total_correct / total_examples


def evaluate(model, dataloader):
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    loss_fn = CrossEntropyLoss()

    for inputs, targets in dataloader:
        logits = model(inputs)
        loss = loss_fn(logits, targets)

        batch_labels = targets.data.astype(np.int64)
        batch_size_actual = len(batch_labels)
        total_loss += float(loss.data) * batch_size_actual
        total_correct += int(np.sum(np.argmax(logits.data, axis=1) == batch_labels))
        total_examples += batch_size_actual

    return total_loss / total_examples, total_correct / total_examples


def smoke_test_model(model):
    """Sanity-check output shape and one backward pass."""
    x = Tensor(np.random.rand(4, 1, 8, 8).astype(np.float32))
    y = Tensor(np.array([0, 1, 2, 3], dtype=np.int64))

    logits = model(x)
    assert logits.shape == (4, 10), f"Expected logits shape (4, 10), got {logits.shape}"

    loss = CrossEntropyLoss()(logits, y)
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None, "Expected gradients on all trainable parameters"


def run_nanodigits_benchmark(
    model,
    model_name,
    title_prefix,
    epochs=10,
    batch_size=32,
    learning_rate=0.02,
    momentum=0.9,
    output_dir=None,
):
    """Train a NanoDigits classifier and save standard benchmark artifacts."""
    smoke_test_model(model)

    (train_images, train_labels), (test_images, test_labels) = load_nanodigits()
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent / "plots"
    else:
        output_dir = Path(output_dir)

    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    loss_fn = CrossEntropyLoss()
    history = {
        "epochs": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    print(f"Training {title_prefix} on NanoDigits")
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    print(
        f"Hyperparameters -> epochs={epochs}, batch_size={batch_size}, "
        f"lr={learning_rate}, momentum={momentum}"
    )
    print(f"Saving plots and metrics to: {output_dir}")

    training_start = time.perf_counter()

    for epoch in range(epochs):
        train_loader = make_nanodigits_loader(
            train_images,
            train_labels,
            batch_size=batch_size,
            shuffle=True,
            seed=7 + epoch,
        )
        test_loader = make_nanodigits_loader(
            test_images,
            test_labels,
            batch_size=batch_size,
            shuffle=False,
        )

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, loss_fn)
        test_loss, test_acc = evaluate(model, test_loader)

        history["epochs"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        plot_training_history(
            history,
            output_dir,
            prefix=model_name,
            title_prefix=title_prefix,
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}"
        )

    total_training_time_seconds = time.perf_counter() - training_start
    summary = {
        "model_name": model_name,
        "benchmark": title_prefix.replace(" ", ""),
        "dataset": "NanoDigits",
        "train_size": int(train_images.shape[0]),
        "test_size": int(test_images.shape[0]),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "momentum": float(momentum),
        "parameter_count": int(count_parameters(model)),
        "total_training_time_seconds": float(total_training_time_seconds),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_test_loss": float(history["test_loss"][-1]),
        "final_train_acc": float(history["train_acc"][-1]),
        "final_test_acc": float(history["test_acc"][-1]),
        "best_train_acc": float(max(history["train_acc"])),
        "best_test_acc": float(max(history["test_acc"])),
        "lowest_train_loss": float(min(history["train_loss"])),
        "lowest_test_loss": float(min(history["test_loss"])),
    }
    save_training_metrics(history, output_dir, prefix=model_name, summary=summary)
    return model, history, summary
