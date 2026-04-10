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
from optimizers import Optimizer,SGD,AdamW,Adam

def analyze_optim_memory_usage():
    print("Analyzing Optimizer Usage...")

    """
    SGD:2x parameter memory i.e momentum buffer
    Adam/AdamW are 3x parameter memory because of two moment buffers
    Memory scales linearly with model size
    Trade-off is that more memory for better convergence
    """

    #creating a test paramters of different sizes
    param_sizes = [1000,10000,100000]

    print("Optimizer Memory Analysis (per parameter tensor):")
    print("="*60)
    print(f"{'Size':<10} {'SGD':<10} {'Adam':10} {'AdamW':<10} {'Ratio':<10}")
    print("-"*60)

    for size in param_sizes:
        #create paramter
        param = Tensor(np.random.randn(size),requires_grad=True)

        #SGDmemory (paramter + momentum buffer)
        sgd = SGD([param],momentum=0.9)
        #Set gradient AFTER creating optimizer
        param.grad = Tensor(np.random.random(size))
        sgd.step() #intialize buffers
        sgd_memory = size * 2 # param + momentum buffers

        #Adam memory (parameter + 2 moment buffers)
        param_adam = Tensor(np.random.randn(size),requires_grad=True)
        adam = Adam([param_adam])
        #Set gradients After creating optimizer
        adam.step()
        adam_memory = size * 3

        #AdamW memory 
        adamw_memory =adam_memory

        #memory ratio i.e Adam/SGD
        ratio = adam_memory /sgd_memory

        print(f"{size:<10} {sgd_memory:<10} {adam_memory:<10} {adamw_memory:<10} {ratio:.1f}x")


if __name__ == "__main__":
    analyze_optim_memory_usage()