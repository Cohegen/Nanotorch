import sys
import pickle
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Tensor import Tensor
from activations.activations import ReLU
from convolution.convolutions import Conv2d, MaxPool2d
from dataloader.dataloader import Dataloader, TensorDataset
from layers.layers import Linear
from losses.losses import CrossEntropyLoss
from optimizers.optimizers import SGD
from training_plots import plot_training_history, save_training_metrics


def load_nanodigits(data_dir=None):
    """Loading the local NanoDigits train/test split."""
    if data_dir is None:
        data_dir = PROJECT_ROOT / "datasets" / "nanodigits"
    else:
        data_dir = Path(data_dir)

    with open(data_dir / "train.pkl", "rb") as handle:
        train = pickle.load(handle)
    with open(data_dir / "test.pkl", "rb") as handle:
        test = pickle.load(handle)

    train_images = np.asarray(train["images"], dtype=np.float32)
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    test_images = np.asarray(test["images"], dtype=np.float32)
    test_labels = np.asarray(test["labels"], dtype=np.int64)

    return (train_images, train_labels), (test_images, test_labels)


def make_loader(images, labels, batch_size=32, shuffle=False, seed=None):
    """Creating a repo-native DataLoader for NanoDigits CNN batches."""
    if seed is not None:
        np.random.seed(seed)

    image_tensor = Tensor(images[:, None, :, :].astype(np.float32))
    label_tensor = Tensor(labels.astype(np.int64))
    dataset = TensorDataset(image_tensor, label_tensor)
    return Dataloader(dataset, batch_size=batch_size, shuffle=shuffle)


class ResidualBlock:
    """A small basic residual block without batch normalization."""

    def __init__(self, channels):
        self.conv1 = Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.conv2 = Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1)
        self.relu = ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        return out

    def parameters(self):
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        return params

    def __call__(self, x):
        return self.forward(x)


class MiniResNetDigits:
    """
    A CPU-friendly mini ResNet for 8x8 grayscale digit images.

    Architecture:
        Input:   1 x 8 x 8
        Stem:    Conv 1 -> 8
        Stage 1: ResidualBlock(8)
        Down:    MaxPool 2x2
        Stage 2: Conv 8 -> 16
        Stage 3: ResidualBlock(16)
        Head:    FC 16*4*4 -> 32 -> 10
    """

    def __init__(self, num_classes=10):
        self.stem = Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.block1 = ResidualBlock(channels=8)
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.transition = Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.block2 = ResidualBlock(channels=16)

        self.relu = ReLU()
        self.flattened_size = 16 * 4 * 4
        self.fc1 = Linear(self.flattened_size, 32)
        self.fc2 = Linear(32, num_classes)

        self._mark_linear_params_trainable()

    def _mark_linear_params_trainable(self):
        """Linear parameters need explicit grad tracking in this codebase."""
        for param in self.fc1.parameters() + self.fc2.parameters():
            param.requires_grad = True
            param.grad = None

    def forward(self, x):
        x = self.stem(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.pool(x)
        x = self.transition(x)
        x = self.relu(x)
        x = self.block2(x)

        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def parameters(self):
        params = []
        params.extend(self.stem.parameters())
        params.extend(self.block1.parameters())
        params.extend(self.transition.parameters())
        params.extend(self.block2.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params

    def predict(self, images):
        logits = self.forward(Tensor(images[:, None, :, :]))
        return np.argmax(logits.data, axis=1)

    def __call__(self, x):
        return self.forward(x)


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


def smoke_test():
    """Sanity-check shapes and one backward pass."""
    model = MiniResNetDigits(num_classes=10)
    x = Tensor(np.random.rand(4, 1, 8, 8).astype(np.float32))
    y = Tensor(np.array([0, 1, 2, 3], dtype=np.int64))

    logits = model(x)
    assert logits.shape == (4, 10), f"Expected logits shape (4, 10), got {logits.shape}"

    loss = CrossEntropyLoss()(logits, y)
    loss.backward()

    for param in model.parameters():
        assert param.grad is not None, "Expected gradients on all trainable parameters"


def main(epochs=10, batch_size=32, learning_rate=0.02, momentum=0.9):
    smoke_test()

    (train_images, train_labels), (test_images, test_labels) = load_nanodigits()
    plot_dir = Path(__file__).resolve().parent / "plots"

    model = MiniResNetDigits(num_classes=10)
    optimizer = SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    loss_fn = CrossEntropyLoss()
    history = {
        "epochs": [],
        "train_loss": [],
        "test_loss": [],
        "train_acc": [],
        "test_acc": [],
    }

    print("Training MiniResNetDigits on NanoDigits")
    print(f"Train: {train_images.shape}, Test: {test_images.shape}")
    print(
        f"Hyperparameters -> epochs={epochs}, batch_size={batch_size}, "
        f"lr={learning_rate}, momentum={momentum}"
    )
    print(f"Saving plots to: {plot_dir}")

    training_start = time.perf_counter()

    for epoch in range(epochs):
        train_loader = make_loader(
            train_images,
            train_labels,
            batch_size=batch_size,
            shuffle=True,
            seed=7 + epoch,
        )
        test_loader = make_loader(
            test_images,
            test_labels,
            batch_size=batch_size,
            shuffle=False,
        )
        train_loss, train_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
        )
        test_loss, test_acc = evaluate(model, test_loader)

        history["epochs"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        plot_training_history(
            history,
            plot_dir,
            prefix="mini_resnet_digits",
            title_prefix="Mini ResNet Digits",
        )

        print(
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}"
        )

    total_training_time_seconds = time.perf_counter() - training_start
    summary = {
        "model_name": "mini_resnet_digits",
        "benchmark": "MiniResNetDigits",
        "dataset": "NanoDigits",
        "train_size": int(train_images.shape[0]),
        "test_size": int(test_images.shape[0]),
        "batch_size": int(batch_size),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "momentum": float(momentum),
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
    save_training_metrics(history, plot_dir, prefix="mini_resnet_digits", summary=summary)

    return model


if __name__ == "__main__":
    main()
