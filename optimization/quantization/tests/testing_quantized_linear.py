import os 
import sys



##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from quantization import INT8_MAX_VALUE, INT8_MIN_VALUE, QuantizedLinear, quantize_int8,dequantize_int8
import numpy as np


def testing_quantized_linear():
    """
    This function intends to test QuantizedLayer class
    to see if it works correctly
    """
    #creating original linear layer 
    original = Linear(4,3)
    original.weight = Tensor(np.random.randn(4,3)*0.5) #smaller range for testing
    original.bias = Tensor(np.random.randn(3)*0.1)

    #creating quantized version
    quantized = QuantizedLinear(original)

    #testing forward pass
    x = Tensor(np.random.randn(2,4)*0.5)

    #original forward pass 
    original_output = original.forward(x)

    #quantized forward pass 
    quantized_output = quantized.forward(x)

    #compare outputs 
    error = np.mean(np.abs(original_output.data- quantized_output.data))
    assert error <0.1,f"Quantization error too high: {error}"

    #testing memory usage 
    memory_info = quantized.memory_usage()
    print(f"  Compression ratio: {memory_info['compression_ratio']:.2f}×")
    print(f"  Original bytes: {memory_info['original_bytes']}")
    print(f"  Quantized bytes: {memory_info['quantized_bytes']}")

    #the compression should be close to 4x (allowing for quantization parameter overhead)
    assert memory_info['compression_ratio'] > 2.5, f"Should achieve ~4× compression, got {memory_info['compression_ratio']:.2f}×"

    print(f"  Memory reduction: {memory_info['compression_ratio']:.1f}x")
    print("QuantizedLinear works correctly!")

if __name__ == "__main__":
    testing_quantized_linear()