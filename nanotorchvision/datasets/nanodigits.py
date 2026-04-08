import pickle
from pathlib import Path

import numpy as np

from Tensor import Tensor
from dataloader.dataloader import Dataloader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


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


def make_nanodigits_loader(images, labels, batch_size=32, shuffle=False, seed=None):
    """Create a NanoTorch DataLoader for 8x8 grayscale digits."""
    if seed is not None:
        np.random.seed(seed)

    image_tensor = Tensor(images[:, None, :, :].astype(np.float32))
    label_tensor = Tensor(labels.astype(np.int64))
    dataset = TensorDataset(image_tensor, label_tensor)
    return Dataloader(dataset, batch_size=batch_size, shuffle=shuffle)
