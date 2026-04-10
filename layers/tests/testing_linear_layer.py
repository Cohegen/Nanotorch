import os
import sys
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
from activations.activations import ReLU, Softmax
from layers.layers import XAVIER_SCALE_FACTOR, Layer,Linear


def testing_linear_layer():
    """Test linear layer implementation."""
    print("Testing Linear Layer...")

    #test layer creation
    layer = Linear(784,256)
    assert layer.in_features == 784
    assert layer.out_features == 256
    assert layer.weight.shape == (784,256)
    assert layer.bias.shape == (256,)

    #testing Xvier intialization (weights should be reasonably scaled)
    weight_std = np.std(layer.weight.data)
    expected_std = np.sqrt(XAVIER_SCALE_FACTOR/784)
    assert 0.5 * expected_std < weight_std < 2.0 * expected_std, f"Weight std{weight_std} not close to Xavier {expected_std}"

    # test bias intialization (should be zeros)
    assert np.allclose(layer.bias.data,0), "Bias should be initialized to zeros"

    #test forward pass
    x = Tensor(np.random.rand(32,784)) #batch of 32 samples
    y = layer.forward(x)
    assert y.shape == (32,256),f"Expected shape (32,256), got{y.shape}"

    #test no bias option
    layer_no_bias = Linear(10,5,bias=False)
    assert layer_no_bias.bias is None
    params = layer_no_bias.parameters()
    assert len(params) == 1 # only weight, no bias

    #test parameters method
    params = layer.parameters()
    assert len(params) == 2 #weight and bias
    assert params[0] is layer.weight
    assert params[1] is layer.bias

    print("Linear layer works correctly.")

if __name__ == "__main__":
    testing_linear_layer()