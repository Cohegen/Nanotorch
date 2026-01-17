import os
import sys
from unittest import result
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses import CrossEntropyLoss,EPILSON
from Tensor import Tensor
from activations.activations import ReLU
from layers.layers import Linear
import numpy as np

def testing_cross_entropy_loss():
    print("Testing Cross-Entropy Loss...")

    loss_fn = CrossEntropyLoss()

    #testing perfect predictions
    perfect_logits = Tensor([[10.0,-10.0,-10.0],[-10.0,10.0,-10.0]]) #very confident predictions
    targets = Tensor([0,1]) #matches the confident predictions
    perfect_loss = loss_fn.forward(perfect_logits,targets)
    assert perfect_loss.data < 0.01, f"Perfect predictions should have very low loss, got {perfect_loss.data}"

    #testing uniform predictions 
    uniform_logits = Tensor([[1.0,1.0,1.0],[1.0,1.0,1.0]]) #equal probabilities
    uniform_targets = Tensor([0,1])
    uniform_loss = loss_fn.forward(uniform_logits,uniform_targets)
    expected_uniform_loss = np.log(3) #log(3) is approximately 1.099 for 3 classes
    assert np.allclose(uniform_loss.data,expected_uniform_loss,atol=0.1),f"Uniform predictions should have loss = log(3)= {expected_uniform_loss:.3f}, got {uniform_loss.data:.3f} "

    # Testing that wrong confident predictions have high loss
    wrong_logits = Tensor([[10.0, -10.0, -10.0], [-10.0, -10.0, 10.0]])  # Confident but wrong
    wrong_targets = Tensor([1, 1])  # Opposite of confident predictions
    wrong_loss = loss_fn.forward(wrong_logits, wrong_targets)
    assert wrong_loss.data > 5.0, f"Wrong confident predictions should have high loss, got {wrong_loss.data}"

    #testing numerical stability with large logits
    large_logits = Tensor([[100.0,50.0,25.0]])
    large_targets = Tensor([0])
    large_loss = loss_fn.forward(large_logits,large_targets)
    assert not np.isnan(large_loss.data),"Loss should not be NaN with large logits"
    assert not np.isinf(large_loss.data),"Loss should not be infinite with large logits"

    print("CrossEntropyLoss works correctly")

if __name__ == "__main__":
    testing_cross_entropy_loss()