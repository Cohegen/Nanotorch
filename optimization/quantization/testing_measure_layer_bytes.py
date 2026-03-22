import os 
import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from Tensor.tensor import BYTES_PER_FLOAT32
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from quantization import INT8_MAX_VALUE, INT8_MIN_VALUE, QuantizedLinear, quantize_int8,dequantize_int8,__collect_layer_inputs,_quantize_single_layer, quantize_model,_measure_layer_bytes
import numpy as np

def testing_measure_layer_bytes():
    """
    This function intends to test whether _measure_layer_bytes
    works correctly

    """
    #testing FP32 Linear layer
    linear = Linear(10,5)
    linear.weight = Tensor(np.random.randn(10,5))
    linear.bias = Tensor(np.random.randn(5))
    params,bytes_ = _measure_layer_bytes(linear)

    assert params == 55, f"Expected 55 params (10*5 + 5), got{params}"
    assert bytes_ == 55 * BYTES_PER_FLOAT32,f"Expected{55* BYTES_PER_FLOAT32} bytes, got {bytes_}"

    #testing ReLU (no parameters)
    relu = ReLU()
    params_relu,bytes_relu = _measure_layer_bytes(relu)
    assert params_relu == 0, "ReLU should have 0 params"
    assert bytes_relu == 0, f"ReLU should have 0 bytes"

    #testing QuantizedLinear layer
    q_linear = QuantizedLinear(linear)
    params_q,bytes_q = _measure_layer_bytes(q_linear,is_quantized=True)
    assert params_q >0, "QuantizedLinear should have params"
    assert bytes_q < bytes_, f"Quantized bytes ({bytes_q}) should be less than FP32 ({bytes_})"

    print(f"  FP32: {params} params, {bytes_} bytes")
    print(f"  INT8: {params_q} params, {bytes_q} bytes")
    print(f"  Ratio: {bytes_ / bytes_q:.1f}x")
    print("Measure layer bytes works correctly!")


if __name__ == "__main__":
    testing_measure_layer_bytes()