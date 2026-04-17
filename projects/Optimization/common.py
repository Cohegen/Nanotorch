import json
import random
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Tensor import Tensor
from activations.activations import ReLU
from dataloader.dataloader import Dataloader, TensorDataset
from layers.layers import Linear, Sequential
from losses.losses import CrossEntropyLoss
from nanotorchvision.datasets import load_nanodigits
from optimizers.optimizers import SGD


DEFAULT_SEED = 7


def seed_everything(seed=DEFAULT_SEED):
    np.random.seed(seed)
    random.seed(seed)


def build_digits_mlp(hidden_dims=(128, 64)):
    layers = []
    in_features = 64
    for hidden_dim in hidden_dims:
        layers.append(Linear(in_features, hidden_dim))
        layers.append(ReLU())
        in_features = hidden_dim
    layers.append(Linear(in_features, 10))
    return Sequential(*layers)


def load_flat_nanodigits():
    (train_images, train_labels), (test_images, test_labels) = load_nanodigits()
    train_x = train_images.reshape(train_images.shape[0], -1).astype(np.float32)
    test_x = test_images.reshape(test_images.shape[0], -1).astype(np.float32)
    train_y = train_labels.astype(np.int64)
    test_y = test_labels.astype(np.int64)
    return (train_x, train_y), (test_x, test_y)


def make_flat_loader(features, labels, batch_size=32, shuffle=False, seed=DEFAULT_SEED):
    if seed is not None:
        np.random.seed(seed)
    dataset = TensorDataset(Tensor(features), Tensor(labels))
    return Dataloader(dataset, batch_size=batch_size, shuffle=shuffle)


def evaluate_classifier(model, dataloader):
    if hasattr(model, "eval"):
        model.eval()

    loss_fn = CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in dataloader:
        logits = model(inputs)
        loss = loss_fn(logits, targets)

        labels = targets.data.astype(np.int64)
        batch_size_actual = len(labels)
        total_loss += float(loss.data) * batch_size_actual
        total_correct += int(np.sum(np.argmax(logits.data, axis=1) == labels))
        total_examples += batch_size_actual

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "examples": total_examples,
    }


def train_classifier(
    model,
    train_features,
    train_labels,
    test_features,
    test_labels,
    epochs=12,
    batch_size=32,
    learning_rate=0.05,
    momentum=0.9,
    seed=DEFAULT_SEED,
):
    seed_everything(seed)
    if hasattr(model, "train"):
        model.train()

    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    loss_fn = CrossEntropyLoss()
    history = []

    for epoch in range(epochs):
        if hasattr(model, "train"):
            model.train()

        train_loader = make_flat_loader(
            train_features,
            train_labels,
            batch_size=batch_size,
            shuffle=True,
            seed=seed + epoch,
        )

        running_loss = 0.0
        running_correct = 0
        total_examples = 0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            loss.backward()
            optimizer.step()

            labels = targets.data.astype(np.int64)
            batch_size_actual = len(labels)
            running_loss += float(loss.data) * batch_size_actual
            running_correct += int(np.sum(np.argmax(logits.data, axis=1) == labels))
            total_examples += batch_size_actual

        train_metrics = {
            "loss": running_loss / max(total_examples, 1),
            "accuracy": running_correct / max(total_examples, 1),
        }
        test_loader = make_flat_loader(
            test_features,
            test_labels,
            batch_size=batch_size,
            shuffle=False,
            seed=None,
        )
        test_metrics = evaluate_classifier(model, test_loader)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
            }
        )

    return history


def make_calibration_samples(features, max_samples=32):
    samples = []
    for row in features[:max_samples]:
        samples.append(Tensor(row.reshape(1, -1).astype(np.float32)))
    return samples


def next_batch(loader):
    for batch in loader:
        return batch
    raise RuntimeError("Expected loader to yield at least one batch")


def save_json(path, payload):
    def _to_jsonable(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_to_jsonable)
