import sys
import os
import time
import numpy as np
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nanotorch as nt
from nanotorchvision.models import AlexNetTinyDigits
from losses.losses import CrossEntropyLoss
from optimizers.optimizers import SGD

def benchmark_model_speed():
    # Setup parameters
    batch_size = 4  # Small batch because naive is very slow
    num_steps = 5   # Number of iterations to average
    
    # Create dummy data (8x8 grayscale digits)
    x_data = np.random.randn(batch_size, 1, 8, 8).astype(np.float32)
    y_data = np.random.randint(0, 10, (batch_size,)).astype(np.int64)
    
    inputs = nt.Tensor(x_data, requires_grad=True)
    targets = nt.Tensor(y_data)
    loss_fn = CrossEntropyLoss()

    print(f"--- CNN Performance Comparison ---")
    print(f"Model: AlexNetTinyDigits")
    print(f"Input Shape: {inputs.shape}")
    print(f"Iterations: {num_steps}")
    print("-" * 35)

    # 1. Benchmark Naive
    model_naive = AlexNetTinyDigits(num_classes=10, method='naive')
    optimizer_naive = SGD(model_naive.parameters(), lr=0.01)
    
    print("\nRunning Naive (Nested Loops)...")
    naive_times = []
    for i in range(num_steps):
        start = time.time()
        
        # Full training step
        optimizer_naive.zero_grad()
        outputs = model_naive(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer_naive.step()
        
        step_time = time.time() - start
        naive_times.append(step_time)
        print(f"  Step {i+1}: {step_time:.4f}s")

    avg_naive = sum(naive_times) / num_steps

    # 2. Benchmark Optimized
    model_opt = AlexNetTinyDigits(num_classes=10, method='im2col')
    optimizer_opt = SGD(model_opt.parameters(), lr=0.01)
    
    print("\nRunning Optimized (im2col)...")
    opt_times = []
    for i in range(num_steps):
        start = time.time()
        
        # Full training step
        optimizer_opt.zero_grad()
        outputs = model_opt(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer_opt.step()
        
        step_time = time.time() - start
        opt_times.append(step_time)
        print(f"  Step {i+1}: {step_time:.4f}s")

    avg_opt = sum(opt_times) / num_steps

    # Results
    print("\n" + "="*35)
    print(f"RESULTS (Average per step):")
    print(f"Naive Method:     {avg_naive:.4f}s")
    print(f"Optimized Method: {avg_opt:.4f}s")
    print("-" * 35)
    print(f"SPEEDUP: {avg_naive / avg_opt:.2f}x FASTER")
    print("="*35)

if __name__ == "__main__":
    benchmark_model_speed()
