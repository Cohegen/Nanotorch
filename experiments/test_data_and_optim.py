from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nanotorch as nt
from nanotorch.optim import SGD
from nanotorch.utils.data import Compose, DataLoader, RandomCrop, RandomHorizontalFlip, TensorDataset


def assert_allclose(actual, expected, message, atol=1e-6):
    if not np.allclose(actual, expected, atol=atol):
        raise AssertionError(f"{message}\nactual={actual}\nexpected={expected}")


def main():
    features = nt.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    labels = nt.tensor([0, 1, 0])
    dataset = TensorDataset(features, labels)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batches = list(loader)

    assert len(dataset) == 3
    assert len(batches) == 2
    assert batches[0][0].shape == (2, 2)
    assert_allclose(batches[0][0].data, np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), "dataloader first feature batch is wrong")
    assert_allclose(batches[0][1].data, np.array([0.0, 1.0], dtype=np.float32), "dataloader first label batch is wrong")

    image = nt.tensor([[1.0, 2.0], [3.0, 4.0]])
    flip = RandomHorizontalFlip(p=1.0)
    flipped = flip(image)
    assert_allclose(flipped.data, np.array([[2.0, 1.0], [4.0, 3.0]], dtype=np.float32), "horizontal flip failed")

    np.random.seed(0)
    crop = RandomCrop(size=2, padding=1)
    cropped = crop(image)
    assert cropped.shape == (2, 2)

    pipeline = Compose([RandomHorizontalFlip(p=1.0)])
    piped = pipeline(image)
    assert_allclose(piped.data, flipped.data, "compose pipeline changed transform output")

    parameter = nt.tensor([1.0, -2.0], requires_grad=True)
    parameter.grad = np.array([0.5, -0.25], dtype=np.float32)
    optimizer = SGD([parameter], lr=0.1)
    optimizer.step()
    assert_allclose(parameter.data, np.array([0.95, -1.975], dtype=np.float32), "sgd step failed")

    print("test_data_and_optim: PASS")


if __name__ == "__main__":
    main()
