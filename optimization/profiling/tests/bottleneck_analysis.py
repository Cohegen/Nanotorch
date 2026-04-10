import os
import sys 
import numpy as np


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear

def bottleneck_analysis():

    profiler = Profiler()

    #testing Memory-bound (how high bandwidth is relative to compute)
    result = profiler._analyze_bottleneck(gflops_per_second=1.0,memory_bandwidth_mbs=10000.0)
    assert result['is_memory_bound'] is True, "High bandwidth should be memory-bound"
    assert result['bottleneck'] == 'memory'
    print('High badnwidth -> memory-bound')

    #testing Compute-bound (low bandwidth relative to compute)
    result = profiler._analyze_bottleneck(gflops_per_second=10.0,memory_bandwidth_mbs=500.0)
    assert result['is_memory_bound'] != result['is_compute_bound'],\
        "Memory-bound and compute-bound should be mutually exclusive"
    print(f"Mutually exclusive: bottleneck = {result['bottleneck']}")

    print("_analyze_bottleneck works correctly!")


if __name__ == "__main__":
    bottleneck_analysis()