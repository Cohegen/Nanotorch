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

def analyze_optim_convergence_behavior():
    print("Analyzeing Optimizer convergence behavior")
    """
    Here we simulate optimization of a quadratic function f(x) = 0.5**x^2
    we expect the optimal solution to be as follows x* = 0 and gradient = x

    SGD has steady progress but can be slow
    SGD+Momentum has faster convergence, less oscillation
    Adam- Adaptive rates help with different parameter scales
    AdamW it's similat to Adam with regularization effects.
    """
    def quadratic_loss(x):
        """
        Simple quadratic function for optimization testing
        """
        return 0.5 * (x**2).sum()

    def compute_grad(x):
        """Gradient of quadratic function: df/dx = x."""
        return x.copy()

    #starting point
    x_start = np.array([5.0,-3.0,2.0])

    #testing different optimizers
    optimizers_to_test = [
        ("SGD",SGD,{"lr":0.1}),
        ("SGD+Momentum",SGD,{"lr":0.1,"momentum":0.9}),
        ("Adam",Adam, {"lr":0.1}),
        ("AdamW",AdamW,{"lr":0.1,"weight_decay":0.01})
    ]

    print("Convergence Analysis of f(x) = 0.5* x**2")
    print("="*70)
    print(f"{'Optimizer':15} {'Step 0':<12} {'Step 5':<12} {'Step 10':<12} {'Final Loss':<12}")

    for name,optimizer_class,kwargs in optimizers_to_test:
        #resetting parameters 
        param = Tensor(x_start.copy(),requires_grad=True)
        optimizer = optimizer_class([param],**kwargs)

        losses = []

        #running optimization for 10 steps
        for step in range(11):
            #compute loss and gradient
            loss = quadratic_loss(param.data)
            param.grad = Tensor(compute_grad(param.data))

            losses.append(loss)

            #updating parameters
            if step < 10:
                optimizer.step()
                optimizer.zero_grad()

             # Format results
        step0 = f"{losses[0]:.6f}"
        step5 = f"{losses[5]:.6f}"
        step10 = f"{losses[10]:.6f}"
        final = f"{losses[10]:.6f}"

        print(f"{name:<15} {step0:<12} {step5:<12} {step10:<12} {final:<12}")

if __name__ == "__main__":
    analyze_optim_convergence_behavior()