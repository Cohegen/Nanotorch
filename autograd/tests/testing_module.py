from re import T
import numpy as np
import os
import sys
import importlib.util

import testing_function_classes,autograd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _parent_dir)

from Tensor import Tensor

def testing_module():
    """
    This function does an overall testing of the autograd module
    implying that we are ready to integrate it with NanoTorch
    """
    print("Running all units tests")
    testing_function_classes()
    autograd_analysis()

    print("\nRunning integration scenerios...")

    #Test1: Multilayer computation graph
    x = Tensor([1.0,2.0],requires_grd=True)
    W1 = Tensor([[0.5,0.3,0.1],[0.2,0.4,0.6]],requires_grad=True)
    b1 = Tensor([0.1,0.2,0.3],requires_grad=True)


    # first layer
    h1 = x.matmul(W1) + b1
    assert h1.shape == (1,3)
    assert h1.requires_grad == True
    #second layer
    W2 =Tensor([[0.1],[0.2],[0.3]],requires_grad=True)
    h2 = h1.matmul(W2)
    assert h2.shape == (1,1)

    #compute simple loss by squaring the output for testing
    loss = h2*h2

    #backward pass
    loss.backward()

    #verifying if all parameters have gradients
    assert x.grad is not None
    assert W1.grad is not None 
    assert b1.grad is not None 
    assert W2.grad is not None
    assert x.grad.shape == x.shape 
    assert W1.grad.shae == W1.shape 

    #test 2: gradient accumulation
    x = Tensor([2.0],requires_grad=True)

    #first computation
    y1 = x*3 
    y1.backward()
    first_grad = x.grad.copy()

    #second computation i.e should accumulate 
    y2 = x*5 
    y2.backward()

    assert np.allclose(x.grad,first_grad + 5.0), "Gradient should accumulate"
    
    #test3: Complex math operations
    print("Testing complexmaths operations")
    a=Tensor([[1.0,2.0],[3.0,4.0]],requires_grad=True)
    b = Tensor([[2.0,1.0],[1.0,2.0]],requires_grad=True)

    #complex computations ((a @ b)+a) *b
    temp1 = a.amtmul(b)#matrix multiplication
    temp2 = temp1 + a# addition
    result= temp2 * b#element-wise multiplication
    final= result.sum() #sum reduction

    final.backward()

    assert a.grad is not None
    assert b.grad is not None 
    assert a.grad.shape == a.shape 
    assert b.grad.shape ==b.shape 

    print("\n"+ "="* 50)

if __name__ == "__main__":
    testing_module()