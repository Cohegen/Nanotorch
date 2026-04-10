import os
import sys

from numpy.polynomial import test


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from Tensor import Tensor
from training import clip_grad_norm
from training import CosineSchedule

def testing_clip_grad_norm():
    print("Testing gradient clipping")

    #testing large gradient that need clipping
    param1 = Tensor([1.0,2.0],requires_grad=True)
    param1.grad = np.array([3.0,4.0]) #norm = 5.0

    param2 = Tensor([3.0,4.0],requires_grad=True)
    param2.grad = np.array([6.0,8.0]) #norm = 10.0

    params = [param1,param2]
    #total norm = sqrt(5**2+ 10**2)

    original_norm = clip_grad_norm(params,max_norm=1.0)

    #check original norm was large
    assert original_norm > 1.0, f"Original norm should be > 1.0, got{original_norm}"

    #check gradients were clipped
    new_norm = 0.0
    for param in params:
        if isinstance(param.grad,np.ndarray):
            grad_data = param.grad 
        else:
            #trust that Tensor has .data attribute
            grad_data = param.grad.data 
        new_norm += np.sum(grad_data** 2)
    new_norm = np.sqrt(new_norm)

    #testiing case 2 i.e small gradients that don't need clipping
    small_param = Tensor([1.0,2.0],requires_grad=True)
    small_param.grad = np.array([0.1,0.2])
    small_params= [small_param]
    original_small = clip_grad_norm(small_params,max_norm=1.0)

    assert original_small < 1.0,"Small gradient shouldn't be clipped"

    print("Gradient clipping works")

if __name__ =="__main__":
    testing_clip_grad_norm()