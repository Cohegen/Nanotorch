import sys
import pickle
from pathlib import Path
import time

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
from training_plots import plot_training_history


def load_nanodigits(data_dir=None):
    """Load the local NanoDigits train/test split."""
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


def batch_iterator(images, labels, batch_size=32, shuffle=True, seed=7):
    """Yield mini-batches shaped for Conv2d: (batch, channels, height, width)."""
    indices = np.arange(len(images))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        batch_images = images[batch_idx][:, None, :, :]
        batch_labels = labels[batch_idx]
        yield batch_images, batch_labels


class LeNetDigitsNaive:
    """
    A small LeNet-style CNN using NAIVE (Nested Loops) convolutions.
    """

    def __init__(self, num_classes=10):
        # Explicitly using naive method for benchmarking
        self.conv1 = Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1, method='naive')
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1, method='naive')
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        self.relu = ReLU()
        self.flattened_size = 16 * 2 * 2
        self.fc1 = Linear(self.flattened_size, 64)
        self.fc2 = Linear(64, 32)
        self.fc3 = Linear(32, num_classes)

        self._mark_linear_params_trainable()

    def _mark_linear_params_trainable(self):
        """Linear parameters need explicit grad tracking in this codebase."""
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

    def __call__(self, x):
        return self.forward(x)


def train_epoch(model, images, labels, optimizer, loss_fn, batch_size=32, seed=7):
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_images, batch_labels in batch_iterator(
        images, labels, batch_size=batch_size, shuffle=True, seed=seed
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
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    loss_fn = CrossEntropyLoss()

    for batch_images, batch_labels in batch_iterator(
        images, labels, batch_size=batch_size, shuffle=False
    ):
        logits = model(Tensor(batch_images))
        loss = loss_fn(logits, Tensor(batch_labels))

        batch_size_actual = len(batch_labels)
        total_loss += float(loss.data) * batch_size_actual
        total_correct += int(np.sum(np.argmax(logits.data, axis=1) == batch_labels))
        total_examples += batch_size_actual

    return total_loss / total_examples, total_correct / total_examples


def main(epochs=2, batch_size=16, learning_rate=0.03):
    (train_images, train_labels), (test_images, test_labels) = load_nanodigits()
    
    # Using only a subset for naive benchmarking because it is SLOW
    train_images = train_images[:100]
    train_labels = train_labels[:100]

    model = LeNetDigitsNaive(num_classes=10)
    optimizer = SGD(model.parameters(), lr=learning_rate)
    loss_fn = CrossEntropyLoss()

    print("--- Training LeNetDigits (NAIVE LOOPS) ---")
    print(f"Training on {len(train_images)} samples for {epochs} epochs...")

    start_time = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        train_loss, train_acc = train_epoch(model, train_images, train_labels, optimizer, loss_fn, batch_size=batch_size)
        duration = time.time() - epoch_start
        print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Time: {duration:.2f}s")
    
    total_duration = time.time() - start_time
    print(f"Total Naive Training Time: {total_duration:.2f}s")

if __name__ == "__main__":
    main()
