import time

import numpy as np
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tensor import Tensor


from dataloader import Dataloader, TensorDataset


def analyze_dataloader_performance():
    print("Analyzing Dataloader performance")

    # Creating test dataset of varying size
    sizes = [1000, 5000, 10000]
    batch_sizes = [16, 64, 256]

    print("\nBatch size vs Loading time")

    for size in sizes:
        # Creating synthetic dataset
        features = Tensor(np.random.rand(size, 100))  # 100 features
        labels = Tensor(np.random.randint(0, 10, size))  # class indices
        dataset = TensorDataset(features, labels)

        print(f"\nDataset size: {size} samples")

        for batch_size in batch_sizes:
            # Time data loading
            loader = Dataloader(dataset, batch_size=batch_size, shuffle=False)

            start_time = time.time()
            batch_count = 0
            for _batch in loader:
                batch_count += 1
            elapsed = time.time() - start_time

            throughput = size / elapsed if elapsed > 0 else float("inf")
            print(
                f" Batchsize {batch_size:3d}: {elapsed:.3f}s ({throughput:,.0f} samples/sec)"
            )

    # Analyzing shuffle overhead
    print("\nShuffle Overhead Analysis:")

    dataset_size = 10000
    features = Tensor(np.random.randn(dataset_size, 50))
    labels = Tensor(np.random.randint(0, 5, dataset_size))
    dataset = TensorDataset(features, labels)

    batch_size = 64

    # No shuffle
    loader_no_shuffle = Dataloader(dataset, batch_size=batch_size, shuffle=False)
    start_time = time.time()
    list(loader_no_shuffle)
    time_no_shuffle = time.time() - start_time

    # With shuffle
    loader_shuffle = Dataloader(dataset, batch_size=batch_size, shuffle=True)
    start_time = time.time()
    list(loader_shuffle)
    time_shuffle = time.time() - start_time

    shuffle_overhead = (
        ((time_shuffle - time_no_shuffle) / time_no_shuffle) * 100
        if time_no_shuffle > 0
        else float("inf")
    )

    print(f"  No shuffle: {time_no_shuffle:.3f}s")
    print(f"  With shuffle: {time_shuffle:.3f}s")
    if np.isfinite(shuffle_overhead):
        print(f"  Shuffle overhead: {shuffle_overhead:.1f}%")
    else:
        print("  Shuffle overhead: n/a (no-shuffle time was ~0)")


def analyze_memory_usage():
    print("\nAnalyzing Memory Usage Patterns...")

    # Memory usage estimation
    def estimate_memory_mb(batch_size: int, feature_size: int, dtype_bytes: int = 4) -> float:
        """Estimate memory usage for a batch (MB)."""
        return (batch_size * feature_size * dtype_bytes) / (1024 * 1024)

    print("\nMemory Usage by Batch Configuration:")

    feature_sizes = [784, 3072, 50176]  # MNIST, CIFAR-10, ImageNet-ish
    feature_names = [
        "MNIST (28×28)",
        "CIFAR-10 (32×32×3)",
        "ImageNet (224×224×1)",
    ]
    batch_sizes = [1, 32, 128, 512]

    for feature_size, name in zip(feature_sizes, feature_names):
        print(f"\n{name}:")
        for batch_size in batch_sizes:
            memory_mb = estimate_memory_mb(batch_size, feature_size)
            print(f"  Batch {batch_size:3d}: {memory_mb:6.1f} MB")

    print("\nMemory Trade-offs:")
    print(" - Larger batches: more memory, better GPU utilization")
    print(" - Smaller batches: less memory, noisier gradients")
    print(" - Sweet spot: usually 32-128 depending on model size")

    # Demonstrate actual memory usage with our tensors
    print("\nActual Tensor Memory Usage:")

    tensor_small = Tensor(np.random.randn(32, 784))  # small batch
    tensor_large = Tensor(np.random.randn(512, 784))  # large batch

    # Measure numpy data bytes
    small_bytes = tensor_small.data.nbytes
    large_bytes = tensor_large.data.nbytes

    # Measure python object overhead (rough)
    small_total = tensor_small.data.__sizeof__() + tensor_small.__sizeof__()
    large_total = tensor_large.data.__sizeof__() + tensor_large.__sizeof__()

    print("  Small batch (32×784):")
    print(f"    - Data only: {small_bytes / 1024:.1f} KB")
    print(f"    - With object overhead: {small_total / 1024:.1f} KB")
    print("  Large batch (512×784):")
    print(f"    - Data only: {large_bytes / 1024:.1f} KB")
    print(f"    - With object overhead: {large_total / 1024:.1f} KB")
    print(f"  Ratio: {large_bytes / small_bytes:.1f}× (data scales linearly)")


def analyze_collation_overhead():
    """Analyze the cost of collating samples into batches."""
    print("\nAnalyzing Collation Overhead...")

    # Test different batch sizes to see collation cost
    dataset_size = 1000
    feature_size = 100
    features = Tensor(np.random.randn(dataset_size, feature_size))
    labels = Tensor(np.random.randint(0, 10, dataset_size))
    dataset = TensorDataset(features, labels)

    print("\nCollation Time by Batch Size:")

    for batch_size in [8, 32, 128, 512]:
        loader = Dataloader(dataset, batch_size=batch_size, shuffle=False)

        start_time = time.time()
        for _batch in loader:
            pass
        total_time = time.time() - start_time

        batches = len(loader)
        time_per_batch_ms = (total_time / batches) * 1000 if batches > 0 else float("inf")

        print(
            f" Batch size {batch_size:3d}: {time_per_batch_ms:.2f}ms per batch ({batches} batches total)"
        )

    print("\nCollation Insights:")
    print(" - Larger batches take longer to collate (more np.stack work)")
    print(" - But fewer large batches can be more efficient than many small ones")
    print(" - Optimal: balance batch size and iteration overhead")


if __name__ == "__main__":
    analyze_dataloader_performance()
    analyze_memory_usage()
    analyze_collation_overhead()