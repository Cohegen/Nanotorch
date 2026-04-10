import numpy as np
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from activations.activations import GELU, TOLERANCE
from Tensor import Tensor

def testing_gelu():
    """A function which tests GELU activation function"""
    print("Testing GELU activation")

    gelu = GELU()

    #Test zero 
    x = Tensor([0.0])
    result = gelu.forward(x)
    assert np.allclose(result.data,[0.0],atol=TOLERANCE),f"GELU(0) should be a 0, got{result.data} instead"

    #testing on postive values
    x = Tensor([1.0])
    result = gelu.forward(x)
    assert result.data[0] > 0.8, f"GELU(1) should be=0.84,got {result.data[0]} instead. "

    #test negative values
    x = Tensor([-1.0])
    result = gelu.forward(x)
    assert result.data[0] < 0 and result.data[0] > -0.2, f"GELU(-1) should be=-0.16, got {result.data[0]} "

    #testing smoothness property i.e no sharp corners like RELU
    x = Tensor([0.001,0.0,0.001])
    result = gelu.forward(x)
    #values should be close to each other
    diff1 = abs(result.data[1]-result.data[0])
    diff2 = abs(result.data[2]-result.data[1])
    assert diff1 < 0.01 and diff2 < 0.01, "GELU shoud be smooth and around zero"

    print("GELU works correctly.")

if __name__ == "__main__":
    testing_gelu()
