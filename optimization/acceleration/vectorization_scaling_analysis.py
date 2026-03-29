import os 

import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from acceleration import vectorized_matmul,DEFAULT_TILING_ITERATIONS, DEFAULT_WARMUP_ITERATIONS, tiled_matmul
import numpy as np

def vectorization_scaling_analysis():
    """
    This function intends to perform some analysis across different tensor sizes with
    respect to the acceleration techniques we have implemented.
    """

    #test sizes spanning different cache regimes
    sizes = [64,128,256,512,1024,2048]

    print("\n Vectorization Scaling Analysis:")
    print("┌─────────┬─────────────┬─────────────┬─────────────┬─────────────┐")
    print("│  Size   │ Time (ms)   │ GFLOPS      │ Bandwidth   │ Efficiency  │")
    print("│         │             │             │ (GB/s)      │ (% of peak) │")
    print("├─────────┼─────────────┼─────────────┼─────────────┼─────────────┤")

    for size in sizes:
        #creating test matrices
        a = Tensor(np.random.rand(size,size).astype(np.float32))
        b = Tensor(np.random.randn(size,size).astype(np.float32))

        #warm up
        for _ in range(2):
            _ = vectorized_matmul(a,b)

        #time elapsed for vectorized matmuls
        iterations = max(1,100//(size //64)) #fewer iterations for large sizes
        import time 
        start = time.time()
        for _ in range(iterations):
            result = vectorized_matmul(a,b)
        elapsed = (time.time()-start) /iterations

        #calculating performance metrics
        flops = 2*size**3 # 2N**3 FLOPS for matrix multiplication
        gflops = flops / (elapsed * 1e9)

        bytes_accessed = 3*size*size*4 #3 matricesx size**2x4bytes
        bandwidth = bytes_accessed / (elapsed *1e9)

         # Estimating efficiency (rough baseline: modern CPU ~100-500 GFLOPS peak)
        estimated_peak_gflops = 200  # Conservative estimate
        efficiency = min(100, gflops / estimated_peak_gflops * 100)

        print(f"│ {size:6d}  │ {elapsed*1000:9.2f}   │ {gflops:9.1f}   │ {bandwidth:9.1f}   │ {efficiency:9.1f}   │")
    
    print("└─────────┴─────────────┴─────────────┴─────────────┴─────────────┘")

if __name__ == "__main__":
    vectorization_scaling_analysis()