from multiprocessing import Value
import time
import os
import sys
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
from activations.activations import ReLU, Softmax
from layers.layers import XAVIER_SCALE_FACTOR, Layer,Linear,Dropout

def analyzing_layer_performance():
    """Analyzing layer Computational Complexity..."""
    print("Analyzing layer Computational Complexity....")

    #test forward pass FLOPs
    batch_sizes = [1,32,128,512]
    layer = Linear(784,256)

    print("\nLinear Layer FLOPs Analysis:")
    print("Batch size -> Matrix Multiply FLOPs -> Bias Add FLOPs -> Total FLOPs")

    for batch_size in batch_sizes:
        #matmuls: (batch,in) @ (in,out) = batch*in*out FLOPs
        matmul_flops = batch_size* 784 * 256
        #bias addition: batch * out FLOPs
        bias_flops = batch_size * 256
        total_flops = matmul_flops + bias_flops

        print(f"{batch_size:10d} -> {matmul_flops:15,} -> {bias_flops:13,}-> {total_flops:11,}")

    #adding timing measurements
    print("\nLinear Layer Timing Analysis: ")
    print("Batch size-> Time(ms) -> Thoughput (samples/sec)")

    for batch_size in batch_sizes:
        x = Tensor(np.random.randn(batch_size,784))

        #warm up
        for _ in range(10):
            _ =layer.forward(x)

        #time multiple iterations
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            _ = layer.forward(x)
        elapsed = time.perf_counter() - start

        time_per_forward =(elapsed /iterations) * 1000 #converting to nanoseconds
        throughput = (batch_size * iterations) /elapsed

        print(f"{batch_size:10d} -> {time_per_forward:8.3f} ms -> {throughput:12,.0f} samples/sec")


if __name__ == "__main__":
    analyzing_layer_performance()  