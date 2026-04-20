import argparse
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Tensor import Tensor
from activations.activations import ReLU
from convolution.convolutions import BatchNorm2d, Conv2d, MaxPool2d
from dataloader.dataloader import Compose, RandomCrop, RandomHorizontalFlip
from layers.layers import Linear
from losses.losses import CrossEntropyLoss
from optimizers.optimizers import Adam
from projects.data_manager import DatasetManager
from training_plots import plot_training_history, save_training_metrics


def batch_iterator(images, labels, batch_size=32, shuffle=True, seed=7, transform=None):
    """Yield mini-batches shaped for Conv2d: (batch, channels, height, width)."""
    indices = np.arange(len(images))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_images = images[batch_idx].copy()
        batch_labels = labels[batch_idx]

        if transform is not None:
            batch_images = np.stack([transform(image).data for image in batch_images], axis=0)

        yield batch_images.astype(np.float32), batch_labels.astype(np.int64)


train_transforms = Compose(
    [
        RandomHorizontalFlip(p=0.5),
        RandomCrop(32, padding=4),
    ]
)


class LeNetCIFAR:
    """
    A LeNet-style CNN for CIFAR-10 32x32 RGB images.

    Architecture:
        Input:  3 x 32 x 32
        Conv:   3 -> 16, 5x5
        Pool:   2x2
        Conv:   16 -> 32, 5x5
        Pool:   2x2
        FC:     32*5*5 -> 120 -> 84 -> 10
    """

    def __init__(self, num_classes=10):
        self.conv1 = Conv2d(in_channels=3, out_channels=16, kernel_size=5)
        self.bn1 = BatchNorm2d(16)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.bn2 = BatchNorm2d(32)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        self.relu = ReLU()
        self.flattened_size = 32 * 5 * 5
        self.fc1 = Linear(self.flattened_size, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, num_classes)

        self._mark_linear_params_trainable()

    def _mark_linear_params_trainable(self):
        for param in self.fc1.parameters() + self.fc2.parameters() + self.fc3.parameters():
            param.requires_grad = True
            param.grad = None

    def train(self):
        self.bn1.training = True
        self.bn2.training = True
        return self

    def eval(self):
        self.bn1.eval()
        self.bn2.eval()
        return self

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def parameters(self):
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.bn1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.bn2.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.fc3.parameters())
        return params

    def __call__(self, x):
        return self.forward(x)


def train_epoch(
    model,
    images,
    labels,
    optimizer,
    loss_fn,
    batch_size=32,
    seed=7,
    transform=None,
):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_images, batch_labels in batch_iterator(
        images,
        labels,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        transform=transform,
    ):
        inputs = Tensor(batch_images)
        targets = Tensor(batch_labels)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size_actual = len(batch_labels)
        total_loss += float(loss.data) * batch_size_actual
        total_correct += int(np.sum(np.argmax(logits.data, axis=1) == batch_labels))
        total_examples += batch_size_actual

    return total_loss / total_examples, total_correct / total_examples


def evaluate(model, images, labels, batch_size=64):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    loss_fn = CrossEntropyLoss()

    for batch_images, batch_labels in batch_iterator(
        images,
        labels,
        batch_size=batch_size,
        shuffle=False,
    ):
        logits = model(Tensor(batch_images))
        loss = loss_fn(logits, Tensor(batch_labels))

        batch_size_actual = len(batch_labels)
        total_loss += float(loss.data) * batch_size_actual
        total_correct += int(np.sum(np.argmax(logits.data, axis=1) == batch_labels))
        total_examples += batch_size_actual

    return total_loss / total_examples, total_correct / total_examples


def main(
    epochs=50,
    batch_size=8,
    learning_rate=0.001,
    quick_test=False,
    train_limit=500,
    test_limit=200,
):
    data_manager = DatasetManager()
    (train_images, train_labels), (test_images, test_labels) = data_manager.get_cifar10()

    if train_limit is not None:
        train_images = train_images[:train_limit]
        train_labels = train_labels[:train_limit]
    if test_limit is not None:
        test_images = test_images[:test_limit]
        test_labels = test_labels[:test_limit]

    if quick_test:
        train_images = train_images[:500]
        train_labels = train_labels[:500]
        test_images = test_images[:100]
        test_labels = test_labels[:100]

    plot_dir = Path(__file__).resolve().parent / "plots"
    model = LeNetCIFAR(num_classes=10)
    optimizer = Adam(model.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss()
    history = {
        "epochs": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    print("Training LeNetCIFAR on CIFAR-10")
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    print(
        f"Hyperparameters -> epochs={epochs}, batch_size={batch_size}, "
        f"lr={learning_rate}, quick_test={quick_test}, "
        f"train_limit={train_limit}, test_limit={test_limit}"
    )
    print(f"Saving plots and metrics to: {plot_dir}")

    training_start = time.perf_counter()

    for epoch in range(epochs):
        epoch_start = time.perf_counter()
        train_loss, train_acc = train_epoch(
            model,
            train_images,
            train_labels,
            optimizer,
            loss_fn,
            batch_size=batch_size,
            seed=7 + epoch,
            transform=train_transforms,
        )
        test_loss, test_acc = evaluate(model, test_images, test_labels, batch_size=batch_size)
        epoch_time_seconds = time.perf_counter() - epoch_start

        history["epochs"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)

        plot_training_history(
            history,
            plot_dir,
            prefix="lenet_cifar",
            title_prefix="LeNet CIFAR",
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.3f} | "
            f"epoch_time={epoch_time_seconds:.2f}s"
        )

    total_training_time_seconds = time.perf_counter() - training_start
    summary = {
        "model_name": "lenet_cifar",
        "benchmark": "LeNetCIFAR",
        "dataset": "CIFAR-10",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "quick_test": quick_test,
        "train_limit": len(train_images),
        "test_limit": len(test_images),
        "total_training_time_seconds": total_training_time_seconds,
        "final_train_loss": history["train_loss"][-1],
        "final_test_loss": history["test_loss"][-1],
        "final_train_acc": history["train_acc"][-1],
        "final_test_acc": history["test_acc"][-1],
    }
    save_training_metrics(history, plot_dir, prefix="lenet_cifar", summary=summary)

    print(f"Total training time: {total_training_time_seconds:.2f}s")
    print(f"Final test accuracy: {history['test_acc'][-1]:.3f}")

    return model, history, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--train-limit", type=int, default=500)
    parser.add_argument("--test-limit", type=int, default=200)
    args = parser.parse_args()

    main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        quick_test=args.quick_test,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )
