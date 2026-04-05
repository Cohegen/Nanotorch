import sys
import os
import numpy as np
from numpy.ma import flatten_structured_array

from projects import data_manager
from projects.CNNS.lenet_digits import project_root
rng = np.random.default_rng(7)
import argparse
import time 

##adding project to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

#importing NanoTorch components
from Tensor import Tensor
from layers.layers import Linear
from activations.activations import ReLU,Softmax
from convolution.convolutions import Conv2d,MaxPool2d,BatchNorm2d
from optimizers.optimizers import Adam
from dataloader.dataloader import DataLoader,Dataset,RandomCrop,RandomHorizontalFlip,Compose

#importing dataset manager
from data_manager import DatasetManager



class CIFARDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        img = self.data[idx]

        if self.transform is not None:
            img = self.transform(img)

        if not isinstance(img, Tensor):
            img = Tensor(img)

        label = Tensor(self.labels[idx])

        return img, label

    def __len__(self):
        return len(self.data)

    def get_num_classes(self):
        return 10


# Augmentations
train_transforms = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomCrop(32, padding=4)
])


def flatten(x):
    if hasattr(x, "data"):
        batch_size = x.data.shape[0]
        return Tensor(x.data.reshape(batch_size, -1))
    else:
        raise ValueError("Expected Tensor with .data attribute")


class CIFARCNN:
    def __init__(self):
        self.conv1 = Conv2d(3, 32, (3, 3))
        self.bn1 = BatchNorm2d(32)

        self.conv2 = Conv2d(32, 64, (3, 3))
        self.bn2 = BatchNorm2d(64)

        self.pool = MaxPool2d(2, 2)
        self.relu = ReLU()

        self.fc1 = Linear(64 * 6 * 6, 256)
        self.fc2 = Linear(256, 10)

        self._training = True

    def train(self):
        self._training = True
        self.bn1.train()
        self.bn2.train()
        return self

    def eval(self):
        self._training = False
        self.bn1.eval()
        self.bn2.eval()
        return self

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))

        x = flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return [
            self.conv1.weight, self.conv1.bias,
            self.bn1.gamma, self.bn1.beta,
            self.conv2.weight, self.conv2.bias,
            self.bn2.gamma, self.bn2.beta,
            self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias
        ]


def train_cifar_cnn(model, train_loader, epochs, learning_rate=0.001):
    print(f"Dataset: {len(train_loader.dataset)}")
    print(f"Batch size: {train_loader.batch_size}")

    model.train()
    optimizer = Adam(model.parameters(), learning_rate=learning_rate)

    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        epoch_loss = 0
        correct = 0
        total = 0
        batch_count = 0

        for batch_idx, (batch_data, batch_labels) in enumerate(train_loader):
            if batch_idx >= 100:
                break

            outputs = model(batch_data)

            batch_size = len(batch_labels.data)
            num_classes = 10

            targets = np.zeros((batch_size, num_classes))
            for i in range(batch_size):
                targets[i, int(batch_labels.data[i])] = 1.0

            outputs_np = np.array(outputs.data)

            exp_outputs = np.exp(outputs_np - np.max(outputs_np, axis=1, keepdims=True))
            softmax = exp_outputs / (np.sum(exp_outputs, axis=1, keepdims=True) + 1e-8)

            loss_value = -np.mean(np.sum(targets * np.log(softmax + 1e-8), axis=1))
            loss = Tensor([loss_value])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions = np.argmax(outputs_np, axis=1)
            correct += np.sum(predictions == batch_labels.data.flatten())
            total += batch_size

            epoch_loss += loss_value
            batch_count += 1

            if (batch_idx + 1) % 20 == 0:
                acc = 100 * correct / max(1, total)
                print(f"Batch {batch_idx+1}: Loss={loss_value:.4f}, Acc={acc:.1f}%")

        avg_loss = epoch_loss / max(1, batch_count)
        acc = 100 * correct / max(1, total)

        print(f"Epoch done → Loss={avg_loss:.4f}, Acc={acc:.1f}%")

    return model


def test_cifar_cnn(model, test_loader, class_names):
    model.eval()

    correct = 0
    total = 0
    class_correct = np.zeros(10)
    class_total = np.zeros(10)

    for batch_idx, (batch_data, batch_labels) in enumerate(test_loader):
        if batch_idx >= 20:
            break

        outputs = model(batch_data)
        outputs_np = np.array(outputs.data)

        preds = np.argmax(outputs_np, axis=1)
        labels = batch_labels.data.flatten()

        correct += np.sum(preds == labels)
        total += len(labels)

        for i in range(len(labels)):
            label = int(labels[i])
            class_total[label] += 1
            if preds[i] == label:
                class_correct[label] += 1

    accuracy = 100 * correct / max(1, total)
    print(f"\nOverall Accuracy: {accuracy:.2f}%")

    return accuracy


def main():
    import argparse
    import time
    import numpy as np

    parser = argparse.ArgumentParser()

    parser.add_argument('--test-only', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)  # ✅ 10 epochs here
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--quick-test', action='store_true')

    args = parser.parse_args()

    class_names = ['plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    data_manager = DatasetManager()
    (train_data, train_labels), (test_data, test_labels) = data_manager.get_cifar10()

    if args.quick_test:
        train_data = train_data[:500]
        train_labels = train_labels[:500]
        test_data = test_data[:100]
        test_labels = test_labels[:100]

    train_dataset = CIFARDataset(train_data, train_labels, transform=train_transforms)
    test_dataset = CIFARDataset(test_data, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)

    model = CIFARCNN()

    if args.test_only:
        rng = np.random.default_rng()
        x = Tensor(rng.standard_normal((1, 3, 32, 32)).astype(np.float32))
        out = model(x)
        print("Forward pass OK:", out.data.shape)
        return

    print("\nStarting training...\n")

    start = time.time()

    model = train_cifar_cnn(
        model,
        train_loader,
        epochs=args.epochs
    )

    print(f"\nTraining time: {time.time() - start:.2f}s")

    print("\nEvaluating...\n")
    acc = test_cifar_cnn(model, test_loader, class_names)

    print(f"\nFinal Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    main()




