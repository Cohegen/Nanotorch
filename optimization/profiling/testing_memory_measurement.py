import os
import sys 
import numpy as np
import tracemalloc

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear

def testing_memory_measurement():
    """
    This function intends to validate memory tracking works correctly and provides useful metrics

    Here we test whether the measure_memory works correctly
    """

    profiler  = Profiler()

    #testing basic memory measurement 
    test_tensor = Tensor(np.random.randn(10,20))
    test_model = Linear(20,10)
    memory_stats = profiler.measure_memory(test_model,(10,20))

    #validating dictionary structure 
    required_keys = ['parameter_memory_mb','activation_memory_mb','peak_memory_mb','memory_efficiency']
    for key in required_keys:
        assert key in memory_stats, f"Missing key: {key}"

    #validating non-negative values
    for key in required_keys:
        assert memory_stats[key] >= 0,f"{key} should be non-negative, got {memory_stats[key]}"

    print(f"Basic measurement: {memory_stats['peak_memory_mb']:.3f} MB peak")


    #testing memory scaling with size 
    small_model = Linear(5,5)
    large_model = Linear(50,50)

    small_memory = profiler.measure_memory(small_model,(5,5))
    large_memory = profiler.measure_memory(large_model,(50,50))

    #large tensor should use more activation memory 
    assert large_memory['activation_memory_mb'] >= small_memory['activation_memory_mb'],\
        "Large tensor should use more activation memory"

    #testing Efficiency bounds
    assert 0<= memory_stats['memory_efficiency'] <=1.0,\
        f"Memory efficiency should be between 0 and 1, got {memory_stats['memory_efficiency']}"

    print(f"Efficiency:{memory_stats['memory_efficiency']:.3f} (0-1) range")

    print("Memory meaurement works correctly")

if __name__ == "__main__":
    testing_memory_measurement() 
