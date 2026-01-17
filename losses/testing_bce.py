import os
import sys
from unittest import result
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses import BinaryCrossEntropyLoss,EPILSON
from Tensor import Tensor
from activations.activations import ReLU
from layers.layers import Linear
import numpy as np

def testing_bce__loss():
    print("Testing binary cross entropy")

    loss_fn = BinaryCrossEntropyLoss()

    #testing perfect predictions
    perfect_predictions = Tensor([0.9999,0.0001,0.9999,0.0001])
    targets = Tensor([1.0,0.0,1.0,0.0])
    perfect_loss = loss_fn.forward(perfect_predictions,targets)
    assert perfect_loss.data < 0.01, f"Perfect predictions should have very low loss, got {perfect_loss.data}"

     # Test worst predictions
    worst_predictions = Tensor([0.0001, 0.9999, 0.0001, 0.9999])
    worst_targets = Tensor([1.0, 0.0, 1.0, 0.0])
    worst_loss = loss_fn.forward(worst_predictions, worst_targets)
    assert worst_loss.data > 5.0, f"Worst predictions should have high loss, got {worst_loss.data}"
    
    #testing uniform predictions (p=0.5)
    uniform_predictions = Tensor([0.5,0.5,0.5,0.5])
    uniform_targets = Tensor([1.0,0.0,1.0,0.0])
    uniform_loss = loss_fn.forward(uniform_predictions,uniform_targets)
    expected_uniform = -np.log(0.5) #should be about 0.693
    assert  np.allclose(uniform_loss.data,expected_uniform,atol=0.01),f"Uniform predictions should have loss = {expected_uniform:.3f}, got {uniform_loss.data:.3f}"

    #test numerical stability at boundaries
    boundary_predictions = Tensor([0.0,1.0,0.0,1.0])
    boundary_targets = Tensor([0.0,1.0,1.0,0.0])
    boundary_loss = loss_fn.forward(boundary_predictions,boundary_targets)
    assert not np.isnan(boundary_loss.data),"Loss should not be NaN at boundaries"
    assert not np.isinf(boundary_loss.data),"Loss should not be infinite at boundaries"

    print("BinaryCrossEntropyLoss works correctly.")

if __name__ == "__main__":
    testing_bce__loss()
