import os
import sys
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
from convolutions import BatchNorm2d

def testing_batchnorm2d_validate_input():
    """
    This function is intended for testing the _validate_inpyt
    method in the BatchNorm class
    """
    bn = BatchNorm2d(num_features=16)

    #valid 4D input should not raise any error
    x_valid = Tensor(np.random.randn(2,16,8,8))
    bn._validate_input(x_valid) #should pass silently

    # testing with 3D input
    ## a 3D input should raise an error
    try:
        bn._validate_input(Tensor(np.random.randn(16,8,8)))
        assert False,"Should have raised ValueError for 3D input"
    except ValueError as e:
        assert "3D" in str(e), f"Error should mention 3D, got:{e}"

    #testing in with wrong channel
    ##wrong channel should raise an error
    try:
        bn._validate_input(Tensor(np.random.randn(2,8,4,4)))
        assert False, "Should have raised ValueError for wrong channels"
    except ValueError as e:
        assert "mismatch" in str(e), f"Error should mention mismatch, got {e}"

    
    print("BatchNorm2d._validate_input works correctly")

if __name__ == "__main__":
    testing_batchnorm2d_validate_input()