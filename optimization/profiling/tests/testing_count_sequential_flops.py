import os
import sys 
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear

def testing_count_sequential_flops():
    """
    This function intends to validate the helper that sums FLOPs across layers in a Sequential model

    """
    profiler = Profiler()

    #creating a dummy sequential model with two Linear Layers
    class DummyLinear:
        def __init__(self,in_f,out_f):
            self.weight = Tensor(np.random.randn(in_f,out_f))
            self.__class__.__name__ = 'Linear'

    class DummySequential:
        def __init__(self,*layer_list):
            self.layers = list(layer_list)

    model = DummySequential(DummyLinear(128,64),DummyLinear(64,10))
    total_flops = profiler._count_sequential_flops(model,(1,128))

    expected = (128*64*2) + (64*10*2)
    assert total_flops == expected,f"Expected {expected}, got {total_flops}"
    print(f"Sequential(128->64->10): {total_flops} FLOPs")

    #single layer sequnetial
    model_single = DummySequential(DummyLinear(32,16))
    flops_single = profiler._count_sequential_flops(model_single,(1,32))
    assert flops_single == 32 * 16 * 2, f"Expected {32*16*2}, got {flops_single}"
    print(f"Single-layer sequential: {flops_single} FLOPs")

    print("   _count_sequential_flops works correctly")

if __name__ == "__main__":
    testing_count_sequential_flops()
        