import os
import sys 
import numpy as np


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear

def testing_count_linear_flops():
    """
    This function validates whether the helper function _count_linear_flops
    computes FLOPS for a single Linear layer.
    """

    profiler = Profiler()

    #creating a dummy linear layer 
    class DummyLinear:
        def __init__(self,in_f,out_f):
            self.weight = Tensor(np.random.randn(in_f,out_f))
            self.__class__.__name__ = 'Linear'

    #testing known dimensions
    layer = DummyLinear(128,64)
    flops = profiler._count_linear_flops(layer,(1,128))
    assert flops == 128*64*2,f"Expected {128*64*2}, got {flops}"
    print(f"Linear(128,64):{flops} FLOPs")

    #testing square layer 
    layer_sq =DummyLinear(256,256)
    flops_sq = profiler._count_linear_flops(layer_sq,(1,256))
    assert flops_sq == 256*256*2,f"Expected {256*256*2},got {flops_sq}"
    print(f"Linear(256,256): {flops_sq} FLOPs")

    #testing batch indepedence (uses last dim only)
    flops_b1 = profiler._count_linear_flops(layer,(1,128))
    flops_b32 = profiler._count_linear_flops(layer,(32,128))
    assert flops_b1 == flops_b32,"Flops should be batch-independent"
    print("Batch-independent FLOPs confirmed")
    print("_count_linear_flops works correctly")

if __name__ == "__main__":
    testing_count_linear_flops()