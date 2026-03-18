import os
import sys
from turtle import backward 
import numpy as np


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear

def advanced_profiling():
    """
    This function is intended to validate our advanced profiling functions
    in profiling.py to provide comprehensive analysis.

    It tests the forward and backward pass proiling completeness
    """

    #create profilier and testing model
    profiler = Profiler()
    test_model = Linear(8,4)
    test_input = Tensor(np.random.randn(1, 8))
    
    #testing forward pass profiling
    forward_profile = profiler.profile_forward_pass(test_model,test_input)

    #validate forward profile structure
    required_forward_keys = [
        'parameters','flops','latency_ms','gflops_per_second',
        'memory_bandwidth_mbs','bottleneck'
    ]

    for key in required_forward_keys:
        assert key in forward_profile,f"Missing key: {key}"

    assert forward_profile['parameters'] >= 0
    assert forward_profile['flops'] >= 0
    assert forward_profile['latency_ms'] >= 0
    assert forward_profile['gflops_per_second'] >=0

    print(f"Forward profiling: {forward_profile['gflops_per_second']:.2f} GFLOP/s")

    #testing backward pass profiling
    backward_profile = profiler.profile_backward_pass(test_model,test_input)

    #validate backward profile structure 
    required_backward_keys = [
        'forward_flops','backward_flops','total_flops',
        'total_latency_ms','total_memory_mb','optimizer_memory_estimates'

    ]

    for key in required_backward_keys:
        assert key in backward_profile,f"Missing key: {key}"

    #validate relationships
    assert backward_profile['total_flops'] >= backward_profile['forward_flops']
    assert backward_profile['total_latency_ms'] >= backward_profile['forward_latency_ms']
    assert 'sgd' in backward_profile['optimizer_memory_estimates']
    assert 'adam' in backward_profile['optimizer_memory_estimates']

    #check backward pass estimates are reasonable
    assert backward_profile['backward_flops'] >= backward_profile['forward_flops'],\
        "Backward pass should have atleast as many FLOPs as forward"
    assert backward_profile['gradient_memory_mb'] >=0,\
        "Gradient memory should be non-negative"

    print(f"Backward profiling: {backward_profile['total_latency_ms']:.2f} ms total")
    print(f"Memory breakdown: {backward_profile['total_memory_mb']:.2f} MB training")
    print("Advanced profiling functions work correctly")

if __name__ == "__main__":
    advanced_profiling()
