import os
import sys 
import numpy as np


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear

def testing_count_conv_flops():
    """
    this function intends to test whether the _count_conv_flops helper function
    computes FLOPs for a Conv2d layer
    """
    profiler = Profiler()

    #creating dummy Conv2d layer
    class DummyConv:
        def __init__(self,in_c,out_c,k,s=1):
            self.in_channels = in_c 
            self.out_channels = out_c 
            self.kernel_size = k
            self.stride = s 
            self.__class__.__name__ = 'Conv2d'

    #testing Siple 3x3 conv and stride 1
    conv = DummyConv(3,16,3,1)
    flops = profiler._count_conv_flops(conv,(1,3,32,32))
    expected = 32*32*3*3*3*16*2
    assert flops == expected,f"Expected {expected}, got {flops}"
    print(f"Conv2d(3,16,3):{flops} FLOPs")

    #testing with Stride 2 which halves output spatila dims
    conv_s2 = DummyConv(3,64,7,2)
    flops_s2 = profiler._count_conv_flops(conv_s2,(1,3,224,224))
    out_h,out_w = 224//2 , 224// 2
    expected_s2 = out_h * out_w*7*7*3*64*2
    assert flops_s2 == expected_s2,f"Expected {expected_s2},got {flops_s2}"
    print(f"Conv2d(3,64,7,stride=2):{flops_s2} FLOPs")

    #testing whether attributes returns 0
    class Incomplete:
        pass

    assert profiler._count_conv_flops(Incomplete(),(1,3,32,32)) == 0
    print("Missing attributes returns 0")

    print("_count_conv_flops works correctly")

if __name__ == "__main__":
    testing_count_conv_flops()