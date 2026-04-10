import numpy as np
from activations import TOLERANCE, ReLU
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Tensor import Tensor


def testing_relu():
    """A dummy function which tests the ReLu function we have created"""
    relu = ReLU()

    print("Testing ReLU activation function")
    #testing a tensor mixed with postive and negative numbers    
    x = Tensor([-1,-2,-3,10,9,5])
    result = relu.forward(x)
    expected = [0,0,0,10,9,5]
    assert np.allclose(result.data,expected), f"RelU failed, expected{expected}, got{result.data}"

    ##testing on purely negative numbers
    x = Tensor([-1,-2,-3,-4,-5])
    result = relu.forward(x)
    assert np.allclose(result.data,[0,0,0,0,0]), "ReLU should zero out all elements of the tensor because they are negative" 

    ##testing on postive tensor
    x = Tensor([1,3,5,7])
    result = relu.forward(x)
    assert np.allclose(result.data,[1,3,5,7]),"Tensor should retain the positive integers"

    ##testing sparsity property
    x = Tensor([-1,-2,-3,1])
    result = relu.forward(x)
    zeros =  np.sum(result.data==0)
    assert zeros == 3 , f"ReLU creates sparsity, got{zeros} out of 4"

    print("ReLU works correctly")

if __name__ == "__main__":
    testing_relu()

