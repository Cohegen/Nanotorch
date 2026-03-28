import os 

import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from acceleration import DEFAULT_TILING_ITERATIONS, DEFAULT_WARMUP_ITERATIONS, vectorized_matmul,unfused_gelu,fused_gelu
import numpy as np

def testing_fusion_speedrun():
    """
    This function measures performance impact of kernel fusion

    """

    #creating moderatelylarge tensor for meaningful timing
    size = 2000
    x = Tensor(np.random.randn(size,size).astype(np.float32))
    warmup_iterations = DEFAULT_WARMUP_ITERATIONS
    timing_iterations= DEFAULT_TILING_ITERATIONS

    #warmingup both implmentations 
    for _ in range(warmup_iterations):
        _ = unfused_gelu(x)
        _ = fused_gelu(x)

    #timing unfused version
    import time
    start = time.time()
    for _ in range(timing_iterations):
        result_unfused = unfused_gelu(x)
    unfused_time = time.time() - start 

    #time fused version 
    start = time.time()
    for _ in range(timing_iterations):
        result_fused = fused_gelu(x)
    fused_time = time.time() -start

    #verifying for numerical correctness
    assert np.allclose(result_unfused.data,result_fused.data,atol=1e-6), \
        "Fused and unfused implementations must be numerically equivalent"

    #calculating performance metrics
    speedup = unfused_time /fused_time if fused_time > 0 else 1.0
    unfused_per_elem = (unfused_time/timing_iterations) / (size *size)*1e9 #ns per element
    fused_per_elem = (fused_time/timing_iterations) / (size *size) *1e9

    print(f"Kernel Fusion Performance Analysis:")
    print(f"   Tensor size: {size}×{size} = {size*size:,} elements")
    print(f"   Unfused time: {unfused_time/timing_iterations*1000:.2f} ms")
    print(f"   Fused time:   {fused_time/timing_iterations*1000:.2f} ms")
    print(f"   Speedup: {speedup:.2f}× faster")
    print(f"   Per-element: {unfused_per_elem:.1f} ns → {fused_per_elem:.1f} ns")

    #memory bandwidth estimate
    bytes_per_elem = 4 #float32
    unfused_memory_ops=7 #7 intermidiate arrays
    fused_memory_ops = 2 #reading input, write output

    unfused_bandwidth =  (unfused_memory_ops * size * size * bytes_per_elem) / (unfused_time / timing_iterations) / 1e9
    fused_bandwidth = (fused_memory_ops * size * size * bytes_per_elem) / (fused_time / timing_iterations) / 1e9

    if speedup > 1.5:
        print(" Excellent! Kernel fusion providing significant speedup")
    elif speedup > 1.1:
        print(" Good! Kernel fusion providing measurable benefit")
    else:
        print("  Limited speedup - may be compute-bound or small tensor size")

    print(" Fusion performance analysis completed!")

if __name__ == "__main__":
    testing_fusion_speedrun()

