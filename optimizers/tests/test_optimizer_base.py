import os
import sys
from tkinter import N

from optimizers import Optimizer 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

##import Tensor from Tensor now with gradient support from autograd
from Tensor import Tensor  

#enable autograd to add gradient tracking to Tensor
from autograd.autograd import enable_autograd 
enable_autograd()
from optimizers import Optimizer

"""
This test validates our base Optimizer class from optimizer.py works correctly
"""

def testing_optimizer_base():
    print("Testing Base optimizer")

    #create test parameters
    param1 = Tensor([1.0,2.0],requires_grad=True)
    param2 = Tensor([[3.0,4.0],[5.0,6.0]],requires_grad=True)

    #create optimizer first (optimizer.__init__ resets grad to None)
    optimizer = Optimizer([param1,param2])

    #test parameter storage
    assert len(optimizer.params) == 2
    assert optimizer.params[0] is param1
    assert optimizer.params[1] is param2 
    assert optimizer.step_count == 0

    #add gradient AFTER creating optimizer to test zero_grad properly
    param1.grad = Tensor([0.1,0.2])
    param2.grad = Tensor([0.3,0.4],[0.5,0.6])

    #test zero_grad 
    optimizer.zero_grad()
    assert param1.grad is None 
    assert param2.grad is None 

    #test that optimizer accepts any tensor
    #gradient tracking is handled by the autograd module
    regular_param= Tensor([1.0])
    opt = Optimizer([regular_param])
    assert len(opt.params) == 1

    print("Base Optimizer work correctly")

if __name__ == "__main__":
    testing_optimizer_base()
