import os
import sys
from tkinter import N
import numpy as np

from optimizers import Optimizer 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

##import Tensor from Tensor now with gradient support from autograd
from Tensor import Tensor  

#enable autograd to add gradient tracking to Tensor
from autograd.autograd import enable_autograd 
enable_autograd()
from optimizers import Optimizer,SGD

def testing_sgd():
    print("Testing SGD optimizer")

    #Test basic SGD without momentum
    param = Tensor([1.0,2.0],requires_grad=True)
    optimizer = SGD([param],lr=0.1)
    #set gradient AFTER creating optimizer (optimizer.__init__ resets grad to None)
    param.grad = Tensor([0.1,0.2])
    original_data = param.data.copy()

    optimizer.step()

    #expected param: param -lr * grad = [1.0,2.0]- 0.1*[0.1,0.2] = [0.99,1.98]
    expected = original_data - 0.1 * np.array([0.1,0.2])
    assert np.allclose(param.data,expected)
    assert optimizer.step_count == 1

    #test SGD with momentum
    param2 = Tensor([1.0,2.0],requires_grad=True)
    optimizer_momentum = SGD([param2],lr=0.1,momentum=0.9)

    #set gradient AFTER creating optimizer 
    param2.grad = Tensor([0.1,0.2])

    #First step: v = 0.9 * 0 + [0.1,0.2] = [0.1,0.2]
    optimizer_momentum.step()
    expected_first = np.array([1.0,2.0]) - 0.1 * np.array([0.1,0.2])
    assert np.allclose(param2.data,expected_first)

    #second step with same gradient
    param2.grad = Tensor([0.1,0.2])
    optimizer_momentum.step()

    #v = 0.9* [0.1,0.2] + [0.1,0.2] = [0.19,0.38]
    expected_momentum = np.array([0.19,0.38])
    expected_second = expected_first - 0.1 * expected_momentum 
    assert np.allclose(param2.data,expected_second,rtol=1e-5)

    # Test weight decay 
    param3 = Tensor([1.0,2.0],requires_grad=True)
    optimizer_wd = SGD([param3],lr=0.1,weight_decay=0.01)
    #set gradient AFTER creating optimizer
    param3.grad = Tensor([0.1,0.2])

    #grad_with_decay = [0.1,0.2] + 0.01 * [1.0,2.0] = [0.11,0.22]
    expected_wd = np.array([1.0,2.0]) - 0.1 * np.array([0.11,0.22])
    assert np.allclose(param3.data,expected_wd)


if __name__ == "__main__":
    testing_sgd()
   