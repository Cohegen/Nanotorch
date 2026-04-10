import os 

import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from acceleration import fused_gelu,unfused_gelu,vectorized_matmul,DEFAULT_TILING_ITERATIONS, DEFAULT_WARMUP_ITERATIONS, tiled_matmul
import numpy as np


def memory_efficiency_analysis():
    """
    This function analyzes memory allocation patterns for different operations.
    """
    import tracemalloc

    sizes = [100,500,1000]

    print("\n Memory Allocation Analysis:")
    print("┌─────────┬──────────────┬──────────────┬──────────────┐")
    print("│  Size   │ Vectorized   │ Unfused GELU │ Fused GELU   │")
    print("│         │ Matmul (MB)  │ (MB)         │ (MB)         │")
    print("├─────────┼──────────────┼──────────────┼──────────────┤")

    for size in sizes:
        x = Tensor(np.random.randn(size,size).astype(np.float32))
        y = Tensor(np.random.randn(size,size).astype(np.float32))

        # Measure vectorized matmul
        tracemalloc.start()
        _ = vectorized_matmul(x, y)
        _, matmul_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Measure unfused GELU
        tracemalloc.start()
        _ = unfused_gelu(x)
        _, unfused_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        # Measure fused GELU
        tracemalloc.start()
        _ = fused_gelu(x)
        _, fused_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        print(f"│ {size:6d}  │ {matmul_peak/1e6:10.2f}   │ {unfused_peak/1e6:10.2f}   │ {fused_peak/1e6:8.2f}   │")
        print("└─────────┴──────────────┴──────────────┴──────────────┘")

    print("\n Key insights:")
    print("   • Vectorized matmul: ~3× input size (2 inputs + 1 output)")
    print("   • Unfused GELU: ~8-10× input size (many intermediate tensors)")
    print("   • Fused GELU: ~2× input size (1 input + 1 output only)")
    print("   • Fusion reduces memory allocations by 4-5×")
    print(" Memory efficiency critical for large batch sizes and limited GPU memory")

if __name__ == "__main__":
    memory_efficiency_analysis()
