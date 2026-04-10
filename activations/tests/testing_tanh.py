import numpy as np
import os 
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Tensor import Tensor
from activations import Tanh, TOLERANCE


def testing_tanh():
    """Fuction which test Tanh activation function"""
    print("Testing Tanh activation function")

    tanh = Tanh()

    #testing zero
    x = Tensor([0.0])
    result = tanh.forward(x)
    assert np.allclose(result.data,[0.0]), f"tanh(0) should be 0, got {result.data}"

    #testing range property all outputs are
    #within the range of (-1,1)
    x = Tensor([-10,-1,0,0,1,10])
    result = tanh.forward(x)
    assert np.all(result.data >= -1) and np.all(result.data <= 1), "All tanh outputs should be in [-1,1]"

    #testing symmetry tanh(-x) = -tanh(x)
    x = Tensor([2.0])
    pos_result = tanh.forward(x)
    x_neg = Tensor([-2.0])
    neg_result = tanh.forward(x_neg)
    assert np.allclose(pos_result.data,-neg_result.data)

    #test extreme values
    x = Tensor([-1000,1000])
    result = tanh.forward(x)
    assert np.allclose(result.data[0],-1,atol=TOLERANCE)
    assert np.allclose(result.data[1],1,atol=TOLERANCE)

    print("Tanh works correctly!")

if __name__ == "__main__":
    testing_tanh()
