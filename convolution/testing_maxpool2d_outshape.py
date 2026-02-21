import os 
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
from convolutions import Conv2d, MaxPool2d

def test_maxpool2d_output_shape():
    """
    This function intends to test MaxPool2d.__compute_pool_output_shape
    method

    """

    #Standard 2x2 pooling with stride 2 i.e halving dimensions
    pool = MaxPool2d(kernel_size=2,stride=2)
    oh,ow = pool._compute_pool_output_shape(8,8)
    assert (oh,ow) == (4,4),f"2x2 stride 2:expected (4,4) got ({oh},{ow})"

    #Non-square input
    oh,ow = pool._compute_pool_output_shape(16,8)
    assert (oh,ow) == (8,4),f"Non-square expected (8,4), got ({oh},{ow})"

    #Overlapping pooling : kernel=3,stride=1
    pool_overlap = MaxPool2d(kernel_size=3,stride=1)
    oh,ow = pool_overlap._compute_pool_output_shape(5,5)
    assert (oh,ow) == (3,3),f"Overlapping:expected (3,3),got ({oh},{ow})"

    #Large kernel
    pool_large = MaxPool2d(kernel_size=4,stride=4)
    oh,ow = pool_large._compute_pool_output_shape(16,16)
    assert (oh,ow) == (4,4),f"4x4 stride:expected (4,4), got ({oh},{ow})"

    print("MaxPool2d computation works correctly")

if __name__ == "__main__":
    test_maxpool2d_output_shape()