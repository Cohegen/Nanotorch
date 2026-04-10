import os
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
from convolutions import AvgPool2d

def testing_avgpool2d_output_shape():
    """
    This function is intended to find out if
    `_compute_pool_output_shape` correctly computes the
    spatial dimensions after average pooling

    """
    #standard 2x2 pooling, this halves dimensions
    pool = AvgPool2d(kernel_size=2,stride=2)
    oh,ow = pool._compute_pool_output_shape(8,8)
    assert (oh,ow) == (4,4),f"2X2 stride 2: expected (4,4), got ({oh},{ow})"

    #Non-square input
    oh,ow = pool._compute_pool_output_shape(16,8)
    assert (oh,ow) == (8,4),f"Non-square (8,4),got ({oh},{ow})"

    #overlapping pooling: kernel=3,stride =1
    pool_overlap = AvgPool2d(kernel_size=3,stride=1)
    oh,ow = pool_overlap._compute_pool_output_shape(5,5)
    assert (oh,ow) == (3,3), f"Overlapping:expected(3,3),got({oh},{ow})"

    print("AvgPool2d output shape computation works correctly!")

if __name__ == "__main__":
    testing_avgpool2d_output_shape()