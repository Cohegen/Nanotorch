import os 
import sys



##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from quantization import INT8_MAX_VALUE, INT8_MIN_VALUE, QuantizedLinear, quantize_int8,dequantize_int8,__collect_layer_inputs
import numpy as np

def testing_collect_layer_inputs():
    """
    This functions seeks to validate whether the 
    collect_layer_inputs works correctly
    """
    #creating a simple model
    layer1 = Linear(4,8)
    layer1.weight = Tensor(np.random.randn(4,8)*0.5)
    layer1.bias = Tensor(np.random.randn(8)*0.1)
    activation = ReLU()
    layer2 = Linear(8,3)
    layer2.weight = Tensor(np.random.randn(8,3)*0.5)
    layer2.bias = Tensor(np.random.randn(3)*0.1)
    model = Sequential(layer1,activation,layer2)

    samples= [Tensor(np.random.randn(1,4)) for _ in range(5)]

    #collecting inputs for layer at index 0
    inputs_at_0 = __collect_layer_inputs(model,0,samples)
    assert len(inputs_at_0) == 5
    assert inputs_at_0[0].shape == (1,4), "Layer 0 inputs should match original shape"

    #collecting inputs for layer at index 2 (after Linear +ReLU)
    inputs_at_2 = __collect_layer_inputs(model,2,samples)
    assert len(inputs_at_2) == 5
    assert inputs_at_2[0].shape == (1,8),f"Layer 2 inputs should be (1,8),got {inputs_at_2[0].shape}"

    #verifying max_samples limiting
    inputs_limited = __collect_layer_inputs(model,2,samples,max_samples=2)
    assert len(inputs_limited) == 2, "Should respect max_samples"

    print("Collect layer inputs works correctly")

if __name__ == "__main__":
    testing_collect_layer_inputs()