import sys
import os
import time
import numpy as np

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import nanotorch as nt
from convolution.convolutions import Conv2d

def benchmark():
    # Setup parameters
    batch_size = 8
    in_channels = 3
    out_channels = 16
    img_size = 32
    kernel_size = 3
    
    x = nt.tensor(np.random.randn(batch_size, in_channels, img_size, img_size), requires_grad=True)
    
    print(f"Benchmarking Conv2d: Input {x.shape}, Output Channels {out_channels}")
    
    # 1. Benchmark Naive (Nested Loops)
    conv_naive = Conv2d(in_channels, out_channels, kernel_size, method='naive')
    # Copy weights to ensure fair comparison
    conv_im2col = Conv2d(in_channels, out_channels, kernel_size, method='im2col')
    conv_im2col.weight.data = conv_naive.weight.data.copy()
    if conv_naive.bias is not None:
        conv_im2col.bias.data = conv_naive.bias.data.copy()

    print("\n--- Method: Naive (Nested Loops) ---")
    start = time.time()
    out_naive = conv_naive(x)
    fwd_time = time.time() - start
    print(f"Forward Pass: {fwd_time:.4f} seconds")
    
    start = time.time()
    out_naive.backward(np.ones_like(out_naive.data))
    bwd_time = time.time() - start
    print(f"Backward Pass: {bwd_time:.4f} seconds")
    total_naive = fwd_time + bwd_time

    # 2. Benchmark im2col
    print("\n--- Method: im2col (Optimized) ---")
    start = time.time()
    out_optimized = conv_im2col(x)
    fwd_time = time.time() - start
    print(f"Forward Pass: {fwd_time:.4f} seconds")
    
    # Reset grad
    x.grad = None
    conv_im2col.weight.grad = None
    
    start = time.time()
    out_optimized.backward(np.ones_like(out_optimized.data))
    bwd_time = time.time() - start
    print(f"Backward Pass: {bwd_time:.4f} seconds")
    total_optimized = fwd_time + bwd_time
    
    # Verify correctness
    diff = np.abs(out_naive.data - out_optimized.data).max()
    print(f"\nMax difference between outputs: {diff:.2e}")
    
    print(f"\nSpeedup: {total_naive / total_optimized:.2f}x faster!")

if __name__ == "__main__":
    benchmark()
