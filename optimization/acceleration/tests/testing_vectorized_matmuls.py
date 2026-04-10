import os 

import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from acceleration import vectorized_matmul
import numpy as np

def testing_vectorized_matmul():
    """
    This function tests whether vectorized_matmul works correctly

    """
    #testing basic 2D multiplication
    a = Tensor([[1,2],[3,4]])
    b = Tensor([[5,6],[7,8]])
    result = vectorized_matmul(a,b)

    expected = np.array([[19,22],[43,50]])
    assert np.allclose(result.data,expected),f"Basic matmul failed: expected {expected}, got{result.data}"

    #testing batch multiplication (3D tensor)
    batch_size,m,k,n = 2,3,4,5
    a_batch = Tensor(np.random.randn(batch_size,m,k))
    b_batch = Tensor(np.random.randn(batch_size,k,n))
    result_batch = vectorized_matmul(a_batch,b_batch)

    assert result_batch.shape== (batch_size,m,n),f"Wrong batch shape: {result_batch.shape}"

    #testing broadcasting (different batch dimensions)
    a_single = Tensor(np.random.randn(m,k))
    b_batch = Tensor(np.random.randn(batch_size,k,n))
    result_broadcast = vectorized_matmul(a_single,b_batch)

    assert result_broadcast.shape == (batch_size,m,n), f"BroadCasting failed: {result_broadcast.shape}"

    #testing error cases
    try:
        vectorized_matmul(Tensor([1,2,3]),Tensor([4,5])) #1D tensors
        assert False, "Should reject 1D tensors"
    except ValueError as e:
        assert "2D+" in str(e)

    try:
         vectorized_matmul(Tensor([[1, 2]]), Tensor([[1], [2], [3]])) #shape mismatch
         assert False, "Should reject incompatible shapes"
    except ValueError as e:
        assert "shape mismatch" in str(e).lower()

    print("vectorzed_matmuls works correctly")

if __name__ == "__main__":
    testing_vectorized_matmul()
