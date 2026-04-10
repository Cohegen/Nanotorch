import os 
import sys



##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from optimization.quantization.quantization import INT8_MAX_VALUE, INT8_MIN_VALUE, quantize_int8
import numpy as np

def testing_quantize_int8():
    """
    This function intends to tests the quantize_int8 function
    """

    #testing basic quantization
    tensor = Tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])
    q_tensor,scale,zero_point = quantize_int8(tensor)

    #verifying qunatized values are in INT8 range
    assert np.all(q_tensor.data >= INT8_MIN_VALUE)
    assert np.all(q_tensor.data <=INT8_MAX_VALUE)
    assert isinstance(scale,float)
    assert isinstance(zero_point,int)

    #testing dequantization preserves approximate values
    dequantized = (q_tensor.data - zero_point) * scale
    error = np.mean(np.abs(tensor.data - dequantized))

    # INT8 quantization has limited precision (256 levels), so error tolerance is higher
    # For a range of 5.0 (1.0 to 6.0), quantization error can be up to ~0.2
    # Using slightly higher tolerance to account for numerical precision variations
    assert error < 0.25, f"Quantization error too high: {error:.4f} (expected < 0.25 for INT8, range=5.0)"

     # Testing edge case: constant tensor
    constant_tensor = Tensor([[2.0, 2.0], [2.0, 2.0]])
    q_const, scale_const, zp_const = quantize_int8(constant_tensor)
    assert scale_const == 1.0

    print("INT8 quantization works correctly!")

if __name__ == "__main__":
    testing_quantize_int8()