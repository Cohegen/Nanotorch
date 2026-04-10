import os
import sys 
import numpy as np

"""
The tracelloc libray tracks memory allocation during model execution.
"""
import tracemalloc

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear

def testing_calculate_parameter_memory():
    """
    This function intends to test the _calculate_parameter_memory helper function
    """

    profiler = Profiler()

    #testing with known parameter counting -> known memory 
    class KnownModel:
        def __init__(self):
            #1024*1024 = 1,048,576 parameters = exactly 4MB atFP32
            self.weight = Tensor(np.random.randn(1024,1024))

    model = KnownModel()
    memory_mb = profiler._calculate_parameter_memory(model)
    expected_mb = (1024*1024*4) / (1024*1024) #4.0 MB
    assert abs(memory_mb - expected_mb) < 0.01, f"Expected {expected_mb} MB, got {memory_mb}"
    print(f"1M params = {memory_mb:.1f} MB")

    #testing Zero parameter model
    class EmptyModel:
        pass 

    empty_mb = profiler._calculate_parameter_memory(EmptyModel())
    assert empty_mb == 0.0, f"Expected 0.0 MB, got {empty_mb}"
    print("Empty model = 0.0 MB")
    
    print("_calculate_parameter_memory works correctly")

if __name__ == "__main__":
    testing_calculate_parameter_memory()
