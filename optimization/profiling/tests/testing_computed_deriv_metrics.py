import os
import sys 
import numpy as np


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear

def testing_computer_derived_metrics():
    """
    This function intends to test the _compute_derived_metrics helper function

    """

    profiler = Profiler()

    #testing whether known values -> known throughput 
    #1e9 FLOPs in 100ms (1 second) = 1.0 GFLOP/s
    metrics = profiler._compute_derived_metrics(
        flops=1_000_000_000,latency_ms = 1000.0,peak_memory_mb=100.0
    )

    assert abs(metrics['gflops_per_second']-1.0) < 0.01, \
        f"Expected 1.0 GFLOP/s, got {metrics['gflops_per_second']}"

    print(f"1B FLOPs / 1s = {metrics['gflops_per_second']:.1f} GFLOP/s")

    #testing memory bandwidth calculation
    #100 MB in 1 second = 100 MB/s
    assert abs(metrics['memory_bandwidth_mbs'] - 100.0) < 0.1, \
        f"Expected 100 MB/s, got {metrics['memory_bandwidth_mbs']}"
    print(f" Memory bandwidth: {metrics['memory_bandwidth_mbs']:.1f} MB/s")


    #testing Efficiency bounded by [0,1]
    assert 0 <= metrics['computational_efficiency'] <=1.0,\
        f"Efficiency out of bounds: {metrics['computational_efficiency']}"
    print(f"Efficient: {metrics['computational_efficiency']:.3f}")

    print("_compute_derived_metrics works correctly")

if __name__ == "__main__":
    testing_computer_derived_metrics()