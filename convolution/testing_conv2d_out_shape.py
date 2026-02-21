import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
from convolutions import Conv2d

def testing_conv2d_output_shape():
    """Tests Conv2d._compute_output_shape for various configurations"""
    print("Testing Conv2d output shape computation")

    #same padding output == input
    conv_same = Conv2d(3,16,kernel_size=3,padding=1,stride=1)
    oh,ow = conv_same._compute_output_shape(32,32)
    assert (oh,ow) == (32,32),f"Same padding:expected (32,32), got ({oh},{ow})"
    
    #No padding:output shrinks by (kernel-1)
    conv_no_pad = Conv2d(3,16,kernel_size=3,padding=0,stride=1)
    oh,ow = conv_no_pad._compute_output_shape(32,32)
    assert (oh,ow) == (30,30),f"Nopadding:expected(30,30), got ({oh},{ow})"

    #Stride 2:output roughly halves
    conv_stride = Conv2d(3,16,kernel_size=3,padding=0,stride=2)
    oh,ow = conv_stride._compute_output_shape(32,32)
    assert (oh,ow) == (15,15),f"Stride 2 :expected(15,15),got ({oh},{ow})"

    #Non-square input 
    conv_rect =Conv2d(1,8,kernel_size=3,padding=1,stride=1)
    oh,ow = conv_rect._compute_output_shape(28,14)
    assert (oh,ow) == (28,14),f"Rectangular: expected (28,14),got({oh},{ow})"

    #Larger kernel
    conv_5x5 = Conv2d(3,16,kernel_size=5,padding=0,stride=1)
    oh,ow = conv_5x5._compute_output_shape(32,32)
    assert (oh,ow) == (28,28),f"5x5 kernel: (28,28),got ({oh},{ow})"

    print("Conv2d output shape computation works correctly!")

if __name__ =="__main__":
    testing_conv2d_output_shape()

