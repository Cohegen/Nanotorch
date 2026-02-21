import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
from convolutions import Conv2d
import numpy as np 

def testing_conv2d_padding():
    """
    This functions tests the `_apply_padding` function
    to ensure it correctly zero-pads the spatial dimensions
    while leaving batch and channel dimensions untouched.
    """

    #No padding 
    conv_no_pad = Conv2d(1,1,kernel_size=3,padding=0)
    x = np.ones((1,1,4,4))
    result = conv_no_pad._apply_padding(x)
    assert result.shape== (1,1,4,4),f"No-pad shape:expected (1,1,4,4), got {result.shape}"
    assert np.array_equal(result,x),"No-pad should return input unchanged"

    #Padding=1:Adds 1 pixel border of zeros
    conv_pad1 = Conv2d(1,1,kernel_size=3,padding=1)
    x = np.ones((1,1,3,3))
    result = conv_pad1._apply_padding(x)
    assert result.shape == (1,1,5,5),f"Pad-1 shape:expected (1,1,5,5),got{result.shape}"

    #Checking that borders are zero
    assert np.all(result[:, :, 0, :] == 0), "Top border should be zeros"
    assert np.all(result[:, :, -1, :] == 0), "Bottom border should be zeros"
    assert np.all(result[:, :, :, 0] == 0), "Left border should be zeros"
    assert np.all(result[:, :, :, -1] == 0), "Right border should be zeros"

    #checking that center is preserved
    assert np.all(result[:,:,1:4,1:4]==1),"Center should be preserved"

    #padding=2 adds 2 pixel border
    conv_pad2 = Conv2d(1,1,kernel_size=5,padding=2)
    x = np.ones((2,3,4,4))
    result = conv_pad2._apply_padding(x)
    assert result.shape == (2,3,8,8),f"Pad-2 shape:expected(2,3,8,8), got {result.shape}"

    #Batch and channel dims unchanged
    assert result.shape[0] == 2,"Batch dim should be unchanged"
    assert result.shape[1]== 3,"Channle dim should be unchanged"

    print("Conv2d padding works correctly")

if __name__ == "__main__":
    testing_conv2d_padding()