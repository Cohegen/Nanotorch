import os
import sys
from unittest import result


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses import log_softmax
from Tensor import Tensor
from activations.activations import ReLU
from layers.layers import Linear
import numpy as np


def testing_log_softmax():
    print("Testing log-softmax")

    #testing basic functionality
    x = Tensor([[1.0,2.0,3.0],[0.1,0.2,0.9]])
    result = log_softmax(x,dim=-1)

    #verifying shape preservation
    assert result.shape == x.shape, f"Shape mismatch: expected {x.shape},got{result.shape}"

     # Verify log-softmax properties: exp(log_softmax) should sum to 1
    softmax_result = np.exp(result.data)
    row_sums = np.sum(softmax_result, axis=-1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), f"Softmax doesn't sum to 1: {row_sums}"

    #testing numerical stabiity with large values
    large_x = Tensor([[100.0,101.0,102.0]])
    large_result = log_softmax(large_x,dim=-1)
    assert not np.any(np.isnan(large_result.data)), "Nan  values in the result with large inputs"
    assert not np.any(np.isinf(large_result.data)),"Inf values in the result with large inputs"

    print("Log softmax works correctly with numerical stability!")

if __name__ == "__main__":
    testing_log_softmax()