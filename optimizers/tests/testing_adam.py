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
from optimizers import Optimizer,Adam 

def testing_adam():

    print("Testing Adam Optimizer")

    #test basic Adam Functionality
    param=Tensor([1.0,2.0],requires_grad=True)
    optimizer = Adam([param],lr=0.01,betas=(0.9,0.999),eps=1e-8)

    #set gradient AFTER creating optimizer (optimizer.__init__ resets grad to None)
    param.grad = Tensor([0.1,0.2])
    original_data = param.data.copy()

    #first step
    optimizer.step()

    #manually compute expected values
    grad = np.array([0.1,0.2])

    #first moment: m = 0.9 *0 + 0.1*grad = 0.1*grad
    m = 0.1* grad 

    #second moment: v = 0.999 * 0 + 0.001 * grad^2 = 0.001* grad^2
    v = 0.001* (grad ** 2)

    # bias correction
    bias_correction1 = 1 - 0.9 ** 1
    bias_correction2 = 1 - 0.999 ** 1

    m_hat = m / bias_correction1
    v_hat = v /bias_correction2

    #update
    expected= original_data - 0.01* m_hat / (np.sqrt(v_hat)+ 1e-8)

    assert np.allclose(param.data,expected,rtol=1e-6)
    assert optimizer.step_count == 1

    #test second step to verify moment accumulation
    param.grad = Tensor([0.1,0.2])
    optimizer.step()

    #should have update moments
    assert optimizer.m_buffers[0] is not None 
    assert optimizer.v_buffers[0] is not None 
    assert optimizer.step_count == 2

    #test with weight decay
    param2 = Tensor([1.0,2.0],requires_grad=True)
    optimizer_wd = Adam([param2],lr=0.01,weight_decay=0.01)

    #set gradient AFTER creating optimizer
    param2.grad =Tensor([0.1,0.2])
    optimizer_wd.step()

    #weight decay should modify the effective gradient
    # grad_with_decay = [0.1,0.2] * [1.0,2.0] = [0.11,0.22]
    # the exact computation is complex
    assert not np.array_equal(param2.data,np.array([1.0,2.0]))

    print("Adam optimizer works")

if __name__ == "__main__":
    testing_adam()