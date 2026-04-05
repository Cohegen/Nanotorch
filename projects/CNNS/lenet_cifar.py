import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Tensor import Tensor
from activations.activations import ReLU
from convolution.convolutions import Conv2d, MaxPool2d
from layers.layers import Linear
from losses.losses import CrossEntropyLoss
from optimizers.optimizers import SGD
from projects.data_manager import DatasetManager


def load_cifar10(max_train=500, max_test=100, seed=7):
    """Load a small CPU-friendly CIFAR-10 subset from the local dataset manager."""
    manager = DatasetManager(auto_confirm=True)
    (train_images, train_labels), (test_images, test_labels) = manager.get_cifar10()

    rng = np.random.default_rng(seed)

    if max_train is not None and max_train < len(train_images):
        train_idx = rng.choice(len(train_images), size=max_train, replace=False)
        train_images = train_images[train_idx]
        train_labels = train_labels[train_idx]

    if max_test is not None and max_test < len(test_images):
        test_idx = rng.choice(len(test_images), size=max_test, replace=False)
        test_images = test_images[test_idx]
        test_labels = test_labels[test_idx]

    return (train_images, train_labels), (test_images, test_labels)


def batch_iterator(images, labels, batch_size=16, shuffle=True, seed=7):
    """Yield mini-batches shaped as (batch, 3, 32, 32)."""
    indices = np.arange(len(images))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        yield images[batch_idx], labels[batch_idx]


class LeNetCIFAR:
    """
    LeNet-style CNN adapted for CIFAR-10.

    Architecture:
        Input:  3 x 32 x 32
        Conv:   3 -> 6, 5x5
        Pool:   2x2
        Conv:   6 -> 16, 5x5
        Pool:   2x2
        FC:     400 -> 120 -> 84 -> 10
    """

    def __init__(self, num_classes=10):
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=5, padding=0)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(in_channels=6, out_channels=16, kernel_size=5, padding=0)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        self.relu = ReLU()
        self.flattened_size = 16 * 5 * 5
        self.fc1 = Linear(self.flattened_size, 120)
        self.fc2 = Linear(120, 84)
        self.fc3 = Linear(84, num_classes)

        self._mark_linear_params_trainable()

    def _mark_linear_params_trainable(self):
        for param in self.fc1.parameters() + self.fc2.parameters() + self.fc3.parameters():
            param.requires_grad = True
            param.grad = None

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
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
        params.extend(self.conv2.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        params.extend(self.fc3.parameters())
        return params

    def predict(self, images):
        logits = self.forward(Tensor(images))
        return np.argmax(logits.data, axis=1)

    def __call__(self, x):
        return self.forward(x)


def train_epoch(model, images, labels, optimizer, loss_fn, batch_size=16, seed=7):
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_images, batch_labels in batch_iterator(
        images, labels, batch_size=batch_size, shuffle=True, seed=seed
    ):
        inputs = Tensor(batch_images.astype(np.float32))
        targets = Tensor(batch_labels)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        count = len(batch_labels)
        total_loss += float(loss.data) * count
        total_correct += int(np.sum(np.argmax(logits.data, axis=1) == batch_labels))
        total_examples += count

    return total_loss / total_examples, total_correct / total_examples


def evaluate(model, images, labels, batch_size=16):
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    loss_fn = CrossEntropyLoss()

    for batch_images, batch_labels in batch_iterator(
        images, labels, batch_size=batch_size, shuffle=False
    ):
        logits = model(Tensor(batch_images.astype(np.float32)))
        loss = loss_fn(logits, Tensor(batch_labels))

        count = len(batch_labels)
        total_loss += float(loss.data) * count
        total_correct += int(np.sum(np.argmax(logits.data, axis=1) == batch_labels))
        total_examples += count

    return total_loss / total_examples, total_correct / total_examples


def smoke_test():
    """Check forward/backward shapes for CIFAR-sized inputs."""
    model = LeNetCIFAR(num_classes=10)
    x = Tensor(np.random.rand(4, 3, 32, 32).astype(np.float32))
    y = Tensor(np.array([0, 1, 2, 3], dtype=np.int64))

    logits = model(x)
    assert logits.shape == (4, 10), f"Expected logits shape (4, 10), got {logits.shape}"

    loss = CrossEntropyLoss()(logits, y)
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None, "Expected gradients on all trainable parameters"


def main(
    epochs=2,
    batch_size=16,
    learning_rate=0.02,
    momentum=0.9,
    max_train=500,
    max_test=100,
):
    smoke_test()

    (train_images, train_labels), (test_images, test_labels) = load_cifar10(
        max_train=max_train,
        max_test=max_test,
    )

    model = LeNetCIFAR(num_classes=10)
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    loss_fn = CrossEntropyLoss()

    print("Training LeNetCIFAR on CIFAR-10")
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    print(
        f"Hyperparameters -> epochs={epochs}, batch_size={batch_size}, "
        f"lr={learning_rate}, momentum={momentum}, "
        f"max_train={max_train}, max_test={max_test}"
    )

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model,
            train_images,
            train_labels,
            optimizer,
            loss_fn,
            batch_size=batch_size,
            seed=7 + epoch,
        )
        test_loss, test_acc = evaluate(model, test_images, test_labels, batch_size=batch_size)

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}"
        )

    return model


if __name__ == "__main__":
    main()
