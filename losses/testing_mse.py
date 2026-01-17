import os
import sys
from unittest import result




sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses import MSELoss,EPILSON
from Tensor import Tensor
from activations.activations import ReLU
from layers.layers import Linear
import numpy as np

def testing_mse_loss():
    print("Testing MSE loss...")

    loss_fn = MSELoss()

    #Testing perfect predictions i.e loss should be 0
    predictions = Tensor([1.0,2.0,3.0])
    targets = Tensor([1.0,2.0,3.0])
    perfect_loss = loss_fn.forward(predictions,targets)
    assert np.allclose(perfect_loss.data,0.0,atol=EPILSON),f"Perfect predictions should have 0 loss got {perfect_loss.data} "

    # Test known case
    predictions = Tensor([1.0, 2.0, 3.0])
    targets = Tensor([1.5, 2.5, 2.8])
    loss = loss_fn.forward(predictions, targets)


    #manual calculation
    expected_loss = (0.25+0.25+0.04) /3
    assert np.allclose(loss.data,expected_loss,atol=1e-6),f"Expected {expected_loss}, got {loss.data}"
    
    #testing that loss is always non-negative
    random_pred = Tensor(np.random.randn(10))
    random_target = Tensor(np.random.randn(10))
    random_loss = loss_fn.forward(random_pred,random_target)
    assert random_loss.data >= 0, f"MSE loss should be non-negative, got{random_loss.data}"

    print("MSELoss works correctly")

if __name__ == "__main__":
    testing_mse_loss()

