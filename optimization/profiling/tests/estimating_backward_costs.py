import os
import sys 
import numpy as np


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear


def estimating_backward_costs():
    """
    This test validates the helper function that estimates backward pass FLOPs and 
    latency from forward measurements.
    """

    profiler = Profiler()

    #testing known forward values -> 2x backward
    costs = profiler._estimate_backward_costs(forward_flops=1000,forward_latency_ms=5.0)
    assert costs['backward_flops'] == 2000,f"Expected 2000, got {costs['backward_flops']}"
    assert costs['backward_latency_ms'] == 10.0, f"Expected10.0, got {costs['backward_latency_ms']}"
    print(f"1000 forward FLOPs -> {costs['backward_flops']} backward FLOPs")

    #testing zero forward -> zero backward 
    costs_zero = profiler._estimate_backward_costs(forward_flops=0,forward_latency_ms=0.0)
    assert costs_zero['backward_flops'] == 0
    assert costs_zero['backward_latency_ms'] == 0.0
    print("Zero forward -> zero backward")

    print("_estimate_backward_cost works correctly")

if __name__ == "__main__":
    estimating_backward_costs()