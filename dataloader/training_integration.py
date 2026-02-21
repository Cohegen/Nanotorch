import time

import numpy as np
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tensor import Tensor


from dataloader import Dataloader, TensorDataset



def test_training_integration():
    """ Test DataLoader integration with training workflow."""
    print(" Integration Test: Training Workflow...")

    # Create a realistic dataset
    num_samples = 1000
    num_features = 20
    num_classes = 5

    # Synthetic classification data
    features = Tensor(np.random.randn(num_samples, num_features))
    labels = Tensor(np.random.randint(0, num_classes, num_samples))

    dataset = TensorDataset(features, labels)

    # Create train/val splits
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Manual split (in production, you'd use proper splitting utilities)
    train_indices = list(range(train_size))
    val_indices = list(range(train_size, len(dataset)))

    # Create subset datasets
    train_samples = [dataset[i] for i in train_indices]
    val_samples = [dataset[i] for i in val_indices]

    # Convert back to tensors for TensorDataset
    train_features = Tensor(np.stack([sample[0].data for sample in train_samples]))
    train_labels = Tensor(np.stack([sample[1].data for sample in train_samples]))
    val_features = Tensor(np.stack([sample[0].data for sample in val_samples]))
    val_labels = Tensor(np.stack([sample[1].data for sample in val_samples]))

    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)

    # Create DataLoaders
    batch_size = 32
    train_loader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = Dataloader(val_dataset, batch_size=batch_size, shuffle=False)

    print(" Dataset splits:")
    print(f"  Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"  Validation: {len(val_dataset)} samples, {len(val_loader)} batches")

    # Simulate training loop
    print("\n Simulated Training Loop:")

    epoch_samples = 0
    batch_count = 0

    for batch_idx, (batch_features, batch_labels) in enumerate(train_loader):
        batch_count += 1
        epoch_samples += len(batch_features.data)

        # Simulate forward pass (just check shapes)
        assert batch_features.data.shape[0] <= batch_size, "Batch size exceeded"
        assert batch_features.data.shape[1] == num_features, "Wrong feature count"
        assert len(batch_labels.data) == len(batch_features.data), "Mismatched batch sizes"

        if batch_idx < 3:  # Show first few batches
            print(f"  Batch {batch_idx + 1}: {batch_features.data.shape[0]} samples")

    print(f"  Total: {batch_count} batches, {epoch_samples} samples processed")

    # Validate that all samples were seen
    assert epoch_samples == len(train_dataset), f"Expected {len(train_dataset)}, processed {epoch_samples}"

    print(" Training integration works correctly!")

if __name__ == "__main__":
    test_training_integration()
