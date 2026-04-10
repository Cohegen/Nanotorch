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
from optimization.profiling.profiling import Profiler

def acceleration_with_profiler():
    """
    This function measures Acceleration gains using Profiler 
    """
    print("Measuring Acceleration Gains with Profiler")
    print("="*70)

    profiler = Profiler()

    #creating two simple models (one slow (loop based) and the other fast (vectorized))
    class SlowLinear:
        """
        Linear Layer using explicit loops (slow)
        """
        def __init__(self,in_features,out_features):
            self.weight = Tensor(np.random.randn(in_features,out_features).astype(np.float32))
            self.name = "slow_linear"

        def forward(self,x):
            #explicitly looping implementation
            batch_size = x.shape[0]
            out_features = self.weight.shape[1]
            result = np.zeros((batch_size,out_features),dtype=np.float32)

            for i in range(batch_size):
                for j in range(out_features):
                    for k in range(x.shape[1]):
                        result[i,j] += x.data[i,k] * self.weight.data[k,j]

            return Tensor(result)

    class FastLinear:
        """
        Linear layer using vectorized matmuls (fast)
        """
        def __init__(self,in_features,out_features):
            self.weight = Tensor(np.random.randn(in_features,out_features).astype(np.float32)*0.01)
            self.name = "fast_linear"

        def forward(self,x):
            #vectorized implementation
            return vectorized_matmul(x,self.weight)

    in_features,out_features = 128,64
    batch_size = 32

    #creating models
    slow_model =SlowLinear(in_features,out_features)
    fast_model = FastLinear(in_features,out_features)

    #creating an input tensor
    input_tensor = Tensor(np.random.randn(batch_size,in_features).astype(np.float32))

    print("\nBefore:Slow Model")
    print("-"*70)

    #measuring slow model
    slow_latency = profiler.measure_latency(slow_model,input_tensor,warmup=3,iterations=10)
    slow_flops = profiler.count_flops(slow_model,(batch_size,in_features))

    print(f"   Latency: {slow_latency:.2f} ms")
    print(f"   FLOPs: {slow_flops:,}")
    print(f"   Throughput: {slow_flops / (slow_latency / 1000) / 1e9:.2f} GFLOP/s")

    print("\n AFTER: Vectorized implementation")
    print("-" * 70)

    #measuring fast model
    fast_latency = profiler.measure_latency(fast_model,input_tensor,warmup=3,iterations=10)
    fast_flops = profiler.count_flops(fast_model,(batch_size,in_features))

    print(f"   Latency: {fast_latency:.2f} ms")
    print(f"   FLOPs: {fast_flops:,}")
    print(f"   Throughput: {fast_flops / (fast_latency / 1000) / 1e9:.2f} GFLOP/s")

    print("\n ACCELERATION GAINS")
    print("=" * 70)
    speedup = slow_latency / fast_latency
    print(f"   Speedup: {speedup:.1f}x faster")
    print(f"   Time saved: {slow_latency - fast_latency:.2f} ms per inference")
    print(f"   Throughput improvement: {speedup:.1f}x more inferences/second")

    print("\n Key Insight:")
    print(f"   Vectorization with numpy.matmul leverages optimized BLAS libraries")
    print(f"   that use SIMD instructions and cache-friendly memory access patterns.")
    print(f"   This is why {speedup:.0f}x speedups are possible with the same FLOPs!")
    print("\n This is the power of acceleration: same math, different execution!")

if __name__ =="__main__":
    acceleration_with_profiler()