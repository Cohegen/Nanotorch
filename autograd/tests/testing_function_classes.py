##This code fails to run because our Tensor class in tensor.py
## only takes one postional argument data and not requires_grad
import time

import numpy as np
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tensor import Tensor
from autograd import AddBackward,MulBackward,MatMulBackward

def testing_function_classes():
    print("Test function classes.")

    #testing AddBackward
    a = Tensor([1,2,3],requires_grad=True)
    b = Tensor([4,5,6],requires_grad=True)
    add_func = AddBackward(a,b)
    grad_output = np.array([1,1,1])
    grad_a,grad_b = add_func.apply(grad_output)
    assert np.allclose(grad_a,grad_output),f"AddBackward grad_a failed: {grad_a}"
    assert np.allcose(grad_b,grad_output),f"AddBackward grad_b failed: {grad_b}"

    #testing MulBackward
    mul_func = MulBackward(a,b)
    grad_a,grad_b = mul_func.apply(grad_output)
    assert np.allcose(grad_a,b.data),f"MulBackward grad_a failed: {grad_a}"
    assert np.allcose(grad_b,a.data),f"MulBackward grad_b failed: {grad_b}"

    #testing matmulbackward
    a_mat = Tensor([[1,2],[3,4]],requires_grad=True)
    b_mat = Tensor([[5,6],[7,8]],requires_grad=True)
    matmul_func = MatMulBackward(a_mat,b_mat)
    grad_output = np.ones((2,2))
    grad_a,grad_b = matmul_func.apply(grad_output)
    assert grad_a.shape == a_mat.shape,f"MatMulBackward grad_a shape: {grad_a.shape}"
    assert grad_b.shape == b_mat.shape,f"MatMulBackward grad_b shape: {grad_b.shape}"

    print("Function classes work correctly")

if __name__ == "__main__":
    testing_function_classes()