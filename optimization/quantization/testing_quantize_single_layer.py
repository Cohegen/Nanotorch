import os 
import sys



##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from quantization import INT8_MAX_VALUE, INT8_MIN_VALUE, QuantizedLinear, quantize_int8,dequantize_int8,__collect_layer_inputs,_quantize_single_layer
import numpy as np

def testing_quantized_single_layer():
    """
    This function intends to test whether the _quantize_single_layer helper function
    works correctly
    """
    #create a linear layer 
    layer = Linear(4,3)
    layer.weight = Tensor(np.random.randn(4,3)*0.5)
    layer.bias = Tensor(np.random.randn(3)*0.1)

    #quantize withot calibration
    q_layer = _quantize_single_layer(layer)
    assert isinstance(q_layer,QuantizedLinear)
    assert q_layer.q_weight is not None,"Quantized weights should exist"
    assert q_layer.input_scale is None,"Without calibration, input_scale should be None"

    #Quantize with calibration
    cal_inputs = [Tensor(np.random.randn(1,4)) for _ in range(5)]
    q_layer_cal = _quantize_single_layer(layer,calibration_inputs=cal_inputs)
    assert isinstance(q_layer_cal,QuantizedLinear)
    assert q_layer_cal.input_scale is not None, "With calibration, input_scale should be set "

    #verifying forward pass works
    x = Tensor(np.random.randn(2,4))
    output =q_layer.forward(x)
    assert output.shape == (2,3),f"Output shape should be (2,3), got {output.shape}"

    print("Quantize single layer helper functions works correctly")

if __name__ == "__main__":
    testing_quantized_single_layer()