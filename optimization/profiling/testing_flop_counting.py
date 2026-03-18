import os
import sys 
import numpy as np


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear

def testing_flop_counting():
    """
    This function intends to test count_flops method
    """
    profiler = Profiler()

    #testing Simple tensor operation
    test_tensor = Tensor(np.random.rand(4,8))
    flops= profiler.count_flops(test_tensor,(4,8))
    expected_flops = 4*8 # 1 FLOP per element for generic operation
    assert flops == expected_flops,f"Expected {expected_flops} FLOPs, got {flops}"
    print(f"Tensor operation:{flops} FLOPs")

    #testing a simulated Linear layer 
    class DummyLinear:
        def __init__(self,in_features,out_features):
            self.weight = Tensor(np.random.randn(in_features,out_features))
            self.__class__.__name__ = 'Linear'

    dummy_linear = DummyLinear(128,64)
    flops = profiler.count_flops(dummy_linear,(1,128))
    expected_flops = 128*64*2 #matmul FLOPs
    assert flops == expected_flops,f"Expected {expected_flops} FLOPs, got {flops}"
    print(f"Linear layer: {flops} FLOPs")

    #testing batch size independece
    flops_batch1 = profiler.count_flops(dummy_linear,(1,128))
    flops_batch32 = profiler.count_flops(dummy_linear,(32,128))
    assert flops_batch1 == flops_batch32,"FLOPs should be independent of batch size"
    print(f"Batch independence: {flops_batch1} FLOPs (same for batch 1 and 32)")

    print("FLOP counting works correclty")

if __name__ == "__main__":
    testing_flop_counting()