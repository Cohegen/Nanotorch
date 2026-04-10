import os
import sys 
import numpy as np


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear

def testing_count_layer_parameters():
    """
    this function intends to validate whether the _count_layer_parameters helper function
    counts parameters from a single layer's weight and bias 

    """

    profiler = Profiler()

    #testing layer with weights and bias 
    class LayerWithBias:
        def __init__(self):
            self.weight = Tensor(np.random.randn(10,5))
            self.bias = Tensor(np.random.randn(5))

    layer= LayerWithBias()
    count = profiler._count_layer_parameters(layer)
    assert count == 55, f"Expected 55(10*5+5),got {count}"
    print(f"Layer with bias: {count} parameters")

    #testing Layer with weight only i.e no bias
    class LayerNoBias:
        def __init__(self):
            self.weight = Tensor(np.random.randn(8,4))

    layer_no_bias = LayerNoBias()
    count = profiler._count_layer_parameters(layer_no_bias)
    assert count == 32, f"Expected 32 (8*4),got{count}"
    print(f"Layer without bias:{count} parameters")

    #tesing object without weight attribute
    class NoWeight:
        pass

    count = profiler._count_layer_parameters(NoWeight())
    assert count == 0,f"Expected 0, got {count}"
    print("No weight attribute: 0 parameters")
    print("_count_layers_parameters works correctly")

if __name__ == "__main__":
    testing_count_layer_parameters()