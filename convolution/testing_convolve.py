import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
from convolutions import Conv2d
import numpy as np 

def testing_convolve_loops():
    """
    This function is intended to validate the core
    sliding window computatation in `_convolve_loops`
    """
    print("This function is testing Conv2d convolution loops...")

    #create a Conv2d with knwn weight(1 input channel,1 output channel,2x2 kernel)
    conv = Conv2d(in_channels=1,out_channels=1,kernel_size=2,bias=False)
    #Set weights to known values: [[1,0],[0,1]] (indentiy-like kernel)
    conv.weight = Tensor(np.array([[[[1.0, 0.0],
                                      [0.0, 1.0]]]]), requires_grad=True)

     # Input: 1 batch, 1 channel, 3x3
    # [[1, 2, 3],
    #  [4, 5, 6],
    #  [7, 8, 9]]


    padded = np.array([[[[1.0,2.0,3.0],
                            [4.0,5.0,6.0],
                            [7.0,8.0,9.0]
                            ]]])

    # Output should be 2x2 (no padding):
    # pos(0,0): 1*1 + 2*0 + 4*0 + 5*1 = 6
    # pos(0,1): 2*1 + 3*0 + 5*0 + 6*1 = 8
    # pos(1,0): 4*1 + 5*0 + 7*0 + 8*1 = 12
    # pos(1,1): 5*1 + 6*0 + 8*0 + 9*1 = 14

    output = conv._convolve_loops(padded,batch_size=1,out_h=2,out_w=2)

    expected = np.array([[[[6.0,8.0],
                            [12.0,14.0]
    
                        ]]])

    assert np.allclose(output,expected),f"Expected:\n{expected}\nGot:\n{output}"

    #Test with multiple output channels
    conv2 = Conv2d(in_channels=1,out_channels=2,kernel_size=2,bias=False)
    #Channel 0 all ones kernel,Channle 1 all twos kernels
    conv2.weight = Tensor(np.array([[[[1.0, 1.0], [1.0, 1.0]]],
                                     [[[2.0, 2.0], [2.0, 2.0]]]]), requires_grad=True)

    output2 = conv2._convolve_loops(padded, batch_size=1, out_h=2, out_w=2)

    # Channel 0 (all-ones kernel): sum of each 2x2 window
    # pos(0,0): 1+2+4+5=12, pos(0,1): 2+3+5+6=16
    # pos(1,0): 4+5+7+8=24, pos(1,1): 5+6+8+9=28
    expected_ch0 = np.array([[12.0, 16.0], [24.0, 28.0]])
    expected_ch1 = expected_ch0 * 2  # All-twos kernel = 2x all-ones
    assert np.allclose(output2[0, 0], expected_ch0), f"Channel 0 mismatch"
    assert np.allclose(output2[0, 1], expected_ch1), f"Channel 1 mismatch"

    print("Conv2d convolution loops work correctly!")


if __name__ =="__main__":
    testing_convolve_loops()