from multiprocessing import reduction
import os
import sys
from turtle import backward 
import numpy as np


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear


def testing_module():
    """
    This function intends to the profiling module
    """
    print("Running Module Intergration Test")
    print("="*50)

    # Testing realistic usage patterns
    print(" Integration Test: Complete Profiling Workflow...")

    #creating a profiler
    profiler = Profiler()

    #creating mode and data 
    test_model = Linear(16,32)
    test_input = Tensor(np.random.randn(8,16))

    #run complete profiling workflow
    print("1.Measuring model characteristics...")
    params = profiler.count_parameters(test_model)
    flops = profiler.count_flops(test_model,test_input.shape)
    memory = profiler.measure_memory(test_model,test_input.shape)
    latency = profiler.measure_latency(test_model,test_input,warmup=2,iterations=5)

    print(f"   Parameters: {params}")
    print(f"   FLOPs: {flops}")
    print(f"   Memory: {memory['peak_memory_mb']:.2f} MB")
    print(f"   Latency: {latency:.2f} ms")

    #testing advanced profiling
    print("2. Running advanced profiling...")
    forward_profile = profiler.profile_forward_pass(test_model,test_input)
    backward_profile = profiler.profile_backward_pass(test_model,test_input)

    assert 'gflops_per_second' in forward_profile
    assert 'total_latency_ms' in backward_profile
    print(f"   Forward GFLOP/s: {forward_profile['gflops_per_second']:.2f}")
    print(f"   Training latency: {backward_profile['total_latency_ms']:.2f} ms")

     # Testing bottleneck analysis
    print("3. Analyzing performance bottlenecks...")
    bottleneck = forward_profile['bottleneck']
    efficiency = forward_profile['computational_efficiency']
    print(f"   Bottleneck: {bottleneck}")
    print(f"   Compute efficiency: {efficiency:.3f}")

    # Validate end-to-end workflow
    assert params >= 0, "Parameter count should be non-negative"
    assert flops >= 0, "FLOP count should be non-negative"
    assert memory['peak_memory_mb'] >= 0, "Memory usage should be non-negative"
    assert latency >= 0, "Latency should be non-negative"
    assert forward_profile['gflops_per_second'] >= 0, "GFLOP/s should be non-negative"
    assert backward_profile['total_latency_ms'] >= 0, "Total latency should be non-negative"
    assert bottleneck in ['memory', 'compute'], "Bottleneck should be memory or compute"
    assert 0 <= efficiency <= 1, "Efficiency should be between 0 and 1"

    print(" End-to-end profiling workflow works!")

    #testing production-like scenario
    print("4. Testing production profiling scenario...")

    #simulating larger model analysis
    large_model = Linear(512, 256)
    large_input = Tensor(np.random.randn(32, 512))  # Larger model input
    large_profile = profiler.profile_forward_pass(large_model, large_input)

    # Verify profile contains optimization insights
    assert 'bottleneck' in large_profile, "Profile should identify bottlenecks"
    assert 'memory_bandwidth_mbs' in large_profile, "Profile should measure memory bandwidth"

    print(f"   Large model analysis: {large_profile['bottleneck']} bottleneck")
    print(f"   Memory bandwidth: {large_profile['memory_bandwidth_mbs']:.1f} MB/s")

    print(" Production profiling scenario works!")

    print("\n" + "=" * 50)

# Run comprehensive module test
if __name__ == "__main__":
    testing_module()