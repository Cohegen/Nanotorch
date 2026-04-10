import os 

import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from acceleration import fused_gelu,vectorized_matmul,DEFAULT_TILING_ITERATIONS, DEFAULT_WARMUP_ITERATIONS, tiled_matmul
import numpy as np

def arithmetic_intensity_analysis():
    """
    This function demonstrates the roofline model with different operations
    """

    size = 1024
    iterations = 10

    operations = []

    #creating test data
    x = Tensor(np.random.randn(size,size).astype(np.float32))
    y = Tensor(np.random.randn(size,size).astype(np.float32))

    print("\n Arithmetic Intensity Analysis:")
    print("┌─────────────────────┬─────────┬─────────────┬─────────────┬─────────────┐")
    print("│ Operation           │ AI      │ Time (ms)   │ GFLOPS      │ GB/s        │")
    print("│                     │(FLOPs/B)│             │             │             │")
    print("├─────────────────────┼─────────┼─────────────┼─────────────┼─────────────┤")

    #1. Element-wise addition (very low arithmetic intensity)
    import time 
    start = time.time()
    for _ in range(iterations):
        _ = Tensor(x.data + y.data)
    add_time = (time.time() - start) / iterations

    add_flops = size*size #one addition per element
    add_bytes = 3*size*size*4 #read x, read y, write result
    add_ai  = add_flops /add_bytes
    add_gflops =  add_flops / (add_time * 1e9)
    add_bandwidth = add_bytes / (add_time * 1e9)

    print(f"│ Element-wise Add    │ {add_ai:6.3f}  │ {add_time*1000:9.2f}   │ {add_gflops:9.1f}   │ {add_bandwidth:9.1f}   │")

    #2. Element-wise multiply (still low,but slightly higher)
    start = time.time()
    for _ in range(iterations):
        _ =Tensor(x.data*y.data)
    mul_time = (time.time())

    mul_flops = size*size
    mul_bytes = 3*size*size*4
    mul_ai = mul_flops /mul_bytes
    mul_gflops = mul_flops / (mul_time * 1e9)
    mul_bandwidth = mul_bytes / (mul_time * 1e9)

    print(f"│ Element-wise Mult   │ {mul_ai:6.3f}  │ {mul_time*1000:9.2f}   │ {mul_gflops:9.1f}   │ {mul_bandwidth:9.1f}   │")

    #3. GELU (medium arithmetic intensity)
    start = time.time()
    for _ in range(iterations):
        _ = fused_gelu(x)
    gelu_time = (time.time() - start) /iterations

    gelu_flops = size*size*8 #approximate : x**3,add,mul,tanh
    gelu_bytes = 2*size*size*size*4 #read x,write result
    gelu_ai = gelu_flops /gelu_bytes 
    gelu_gflops = gelu_flops / (gelu_time *1e9)
    gelu_bandwidth = gelu_bytes / (gelu_time * 1e9)

    print(f"│ Fused GELU          │ {gelu_ai:6.3f}  │ {gelu_time*1000:9.2f}   │ {gelu_gflops:9.1f}   │ {gelu_bandwidth:9.1f}   │")

    #4. Matrix multiplication (high arithmetic intensity)
    start = time.time()
    for _ in range(iterations):
        _ = vectorized_matmul(x,y)
    matmul_time = (time.time()- start) /iterations

    matmul_flops = 2*size**3 #2N**3 FLOPs
    matmul_bytes = 3*size*size*4 #3 matrices
    matmul_ai = matmul_flops / matmul_bytes
    matmul_gflops = matmul_flops / (matmul_time * 1e9)
    matmul_bandwidth = matmul_bytes / (matmul_time * 1e9)

    print(f"│ Matrix Multiply     │ {matmul_ai:6.3f}  │ {matmul_time*1000:9.2f}   │ {matmul_gflops:9.1f}   │ {matmul_bandwidth:9.1f}   │")
    print("└─────────────────────┴─────────┴─────────────┴─────────────┴─────────────┘")

    print(f"\n Roofline Model Insights:")
    print(f"    Low AI (< 1): Memory bound - limited by bandwidth")
    print(f"    Med AI (1-10): Transitional - depends on implementation")
    print(f"    High AI (> 10): Compute bound - limited by ALU throughput")
    print(f"    Matrix multiplication ({matmul_ai:.1f} AI) is ideal for GPUs/TPUs")
    print(f"    Element-wise ops ({add_ai:.3f} AI) need memory optimization")
    print(" Design algorithms with high arithmetic intensity for performance")

if __name__ == "__main__":
    arithmetic_intensity_analysis()