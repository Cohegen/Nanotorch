import os 
import sys



##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from quantization import INT8_MAX_VALUE, INT8_MIN_VALUE, quantize_int8,dequantize_int8
import numpy as np

def testing_dequantize_int8():
    """
    This function intends to test the dequantize_int8 function
    """

    #testing full flow quantize->dequantize
    original = Tensor([[-1.5,0.0,3.2],[1.1,-0.8,2.7]])
    q_tensor, scale, zero_point = quantize_int8(original)
    restored = dequantize_int8(q_tensor, scale, zero_point)

     # Verifying round-trip error is small
    error = np.mean(np.abs(original.data - restored.data))
    assert error < 0.1, f"Round-trip error too high: {error}"

    #verifying output is float32
    assert restored.data.dtype == np.float32

    print("INT8 dequantization works correctly")


if __name__ == "__main__":
    testing_dequantize_int8()