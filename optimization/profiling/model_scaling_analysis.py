import os
import sys 
import numpy as np


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear


def model_scaling_analysis():
    """
    Analysis of how model performance scales with size
    """

    profiler = Profiler()
    results = []

    #test different model sizes 
    sizes = [64,128,256,512]

    print("\nModel Scaling Analysis:")
    print("Size\tParams\t\tFLOPs\t\tLatency(ms)\tMemory(MB)\tGFLOP/s")
    print("-" * 80)

    for size in sizes:
        #create models of different sizes for comparison
        test_model = Linear(size,size)
        input_shape = (32,size) #batch of 32
        dummy_input = Tensor(np.random.randn(*input_shape))

        #simulate linear layer characteristics
        linear_params = size * size + size # W + b
        linear_flops = size * size * 2 #matmul

        #measure actual performance
        latency = profiler.measure_latency(test_model,dummy_input,warmup=3,iterations=10)
        memory = profiler.measure_memory(test_model,input_shape)

        gflops_per_second = (linear_flops/ 1e9) / (latency / 1000)

        results.append({
            'size':size,
            'parameters':linear_params,
            'flops':linear_params,
            'latency_ms':latency,
            'memory_mb':memory['peak_memory_mb'],
            'gflops_per_second':gflops_per_second
        })

        print(f"{size}\t{linear_params:,}\t\t{linear_flops:,}\t\t"
              f"{latency:.2f}\t\t{memory['peak_memory_mb']:.2f}\t\t"
              f"{gflops_per_second:.2f}")

    print("\n Scaling Analysis Insights:")

    #memory scaling
    memory_growth   = results[-1]['memory_mb'] / max(results[0]['memory_mb'],0.001)
    print(f"Memory grows {memory_growth:.1f}× from {sizes[0]} to {sizes[-1]} size")

    #compute scaling
    compute_growth = results[-1]['gflops_per_second'] / max(results[0]['gflops_per_second'], 0.001)
    print(f"Compute efficiency changes {compute_growth:.1f}× with size")

    #Performance characteristics
    avg_efficiency = np.mean([r['gflops_per_second'] for r in results])
    if avg_efficiency < 10:  # Arbitrary threshold for "low" efficiency
        print(" Low compute efficiency suggests memory-bound workload")
    else:
        print(" High compute efficiency suggests compute-bound workload")

def batch_size_effects_analysis():
    """
    Analyze how batch size affects performance and efficiency
    """

    profiler = Profiler()
    batch_sizes = [1,8,32,128]
    feature_size = 256

    print("\nBatch Size Effects Analysis:")
    print("Batch\tLatency(ms)\tThroughput(samples/s)\tMemory(MB)\tMemory Efficiency")
    print("-" * 85)

    for batch_size in batch_sizes:
        test_model = Linear(feature_size,feature_size)
        input_shape = (batch_size,feature_size)
        dummy_input = Tensor(np.random.randn(*input_shape))

        #measure performance
        latency = profiler.measure_latency(test_model,dummy_input,warmup=3,iterations=10)
        memory = profiler.measure_memory(test_model,input_shape)

        #calculate throughput
        samples_per_second = (batch_size * 1000) / latency #samples/second

        #calculate efficiency (samples per unit memory)
        efficiency = samples_per_second /max(memory['peak_memory_mb'],0.001)

        print(f"{batch_size}\t{latency:.2f}\t\t{samples_per_second:.0f}\t\t\t"
              f"{memory['peak_memory_mb']:.2f}\t\t{efficiency:.1f}")

    print("\n Batch Size Insights:")
    print("Larger batches typically improve throughput but increase memory usage")

if __name__ == "__main__":
    model_scaling_analysis()
    batch_size_effects_analysis()