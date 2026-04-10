import os
import sys
from tkinter import N, NO
import numpy as np

from optimizers import Optimizer 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

##import Tensor from Tensor now with gradient support from autograd
from Tensor import Tensor  

#enable autograd to add gradient tracking to Tensor
from autograd.autograd import enable_autograd 
enable_autograd()
from optimizers import Optimizer,AdamW,Adam

def testing_adamw_optimizer():
    print("Testing AdamW Optimizer")

    #testing AdamW vs Adam 
    #creating identical parameters for comparision
    param_adam = Tensor([1.0,2.0],requires_grad=True)
    param_adamw = Tensor([1.0,2.0],requires_grad=True)

    #create optimizers with same settings
    adam= Adam([param_adam],lr=0.01,weight_decay=0.01)
    adamw= AdamW([param_adamw],lr=0.01,weight_decay=0.01)

    param_adam.grad = Tensor([0.1,0.2])
    param_adamw.grad = Tensor([0.1,0.2])

    #take one step
    adam.step()
    adamw.step()

    assert not np.allclose(param_adam.data,param_adamw.data,rtol=1e-6)

    #testing AdamW basic functionality
    param = Tensor([1.0,2.0],requires_grad=True)
    optimizer = AdamW([param],lr=0.01,weight_decay=0.01)
    #set gradient AFTER changed
    param.grad = Tensor([0.1,0.2])
    original_data = param.data.copy()

    optimizer.step()

    #parameter should have changed
    assert not np.array_equal(param.data,original_data)
    assert optimizer.step_count == 1

    #testing that moment buffers are created
    assert optimizer.m_buffers[0] is not None
    assert optimizer.v_buffers[0] is not None 

    #testing zero weight decay behaves like Adam
    param1 = Tensor([1.0,2.0],requires_grad=True)
    param2 = Tensor([1.0,2.0],requires_grad=True)

    adam_no_wd = Adam([param1],lr=0.01,weight_decay=0.0)
    adamw_no_wd = Adam([param2],lr=0.01,weight_decay=0.0)

    #set gradients AFTER creating optimizers
    param1.grad = Tensor([0.1,0.2])
    param2.grad = Tensor([0.1,0.2])

    adam_no_wd.step()
    adamw_no_wd.step()

    assert np.allclose(param1.data,param2.data,rtol=1e-10)

    print("AdamW optimizer works correctly")

if __name__== "__main__":
    testing_adamw_optimizer()
