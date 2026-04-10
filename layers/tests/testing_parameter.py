"""
Here we test linear parameter collection.
This test ensures Linear Layer parameters can be collected for optimization
"""
import os
import sys
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
from activations.activations import ReLU, Softmax
from layers.layers import XAVIER_SCALE_FACTOR, Layer,Linear


def testing_parameter_collection_linear():
    """Testing Linear layer parameter collection."""

    layer = Linear(10,5)

    #verify parameter collection works
    params = layer.parameters()
    assert len(params) == 2 , "Should return 2 parameters (weight anf bias)"
    assert params[0].shape == (10,5), "First param should be weight"
    assert params[1].shape == (5,), "Second param should be bias"

    #test layer without bias
    layer_no_bias = Linear(10,5,bias=False)
    params_no_bias = layer_no_bias.parameters()
    assert len(params_no_bias) == 1, "Should return 1 parameter i.e weight only"

    print("Parameter collection works correctly")

if __name__ == "__main__":
    testing_parameter_collection_linear()
