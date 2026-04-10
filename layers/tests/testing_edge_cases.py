"""
Testing edge cases and error handling
"""
import os
import sys
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
from activations.activations import ReLU, Softmax
from layers.layers import XAVIER_SCALE_FACTOR, Layer,Linear

def test_edge_cases():
    """Testing Linear layer edge cases."""
    print("Edge Case tests: linear layer")

    layer = Linear(10,5)

    #testing single sample
    x_2d = Tensor(np.random.rand(1,10))
    y = layer.forward(x_2d)
    assert y.shape == (1,5), "Should handle single sample"

    #test zero batch size (edge case)
    x_empty = Tensor(np.random.rand(0,10))
    y_empty = layer.forward(x_empty)
    assert y_empty.shape == (0,5), "Should handle empty batch"

    #test nuemrical stability with large weights
    layer_large = Linear(10,5)
    layer_large.weight.data = np.ones((10,5))*100
    x = Tensor(np.ones((1,10)))
    y = layer_large.forward(x)
    assert not np.any(np.isnan(y.data)), "Should not produce NaN with large weights"
    assert not np.any(np.isinf(y.data)),"Should not produce inf with large weights"


    #test with no bias
    layer_no_bias = Linear(10,5,bias=False)
    x = Tensor(np.random.rand(4,10))
    y = layer_no_bias.forward(x)
    assert y.shape == (4,5),"Should work without bias"

    print("Edges cases handled correctly")

if __name__ == "__main__":
    test_edge_cases()