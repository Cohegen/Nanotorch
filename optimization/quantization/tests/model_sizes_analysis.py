import os 
import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from Tensor.tensor import BYTES_PER_FLOAT32
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from quantization import INT8_MAX_VALUE, INT8_MIN_VALUE, QuantizedLinear, quantize_int8,dequantize_int8,__collect_layer_inputs,_quantize_single_layer, quantize_model,_measure_layer_bytes,analyze_model_sizes
import numpy as np
from typing import Tuple,Dict,List,Optional


def model_size_analysis():
    """
    This function intends to the analyze_model_size helper
    function in quantization.py
    """

    #creating and quantizing a model for testing (using Sequential from layers)
    layer1_orig = Linear(100,50)
    activation_orig = ReLU()
    layer2_orig = Linear(50,10)
    layer1_orig.weight = Tensor(np.random.randn(100,50))
    layer1_orig.bias= Tensor(np.random.randn(50))
    layer2_orig.weight= Tensor(np.random.randn(50,10))
    layer2_orig.bias = Tensor(np.random.randn(10))
    original_model = Sequential(layer1_orig,activation_orig,layer2_orig)

    #creating quantized copy
    layer1_quant = Linear(100,50)
    activation_quant = ReLU()
    layer2_quant = Linear(50,10)
    layer1_quant.weight =Tensor(np.random.randn(100,50))
    layer1_quant.bias = Tensor(np.random.randn(50,10))
    layer2_quant.weight = Tensor(np.random.randn(50,10))
    layer2_quant.bias = Tensor(np.random.randn(10))
    quantized_model = Sequential(layer1_quant,activation_quant,layer2_quant)

    quantize_model(quantized_model)
    #analyzing sizes
    comparison = analyze_model_sizes(original_model,quantized_model)

    # Verifying compression achieved
    assert comparison['compression_ratio'] > 2.0, "Should achieve significant compression"
    assert comparison['memory_saved_percent'] > 50, "Should save >50% memory"

    print(f"  Compression ratio: {comparison['compression_ratio']:.1f}x")
    print(f"  Memory saved: {comparison['memory_saved_percent']:.1f}%")
    print("Model size analysis works correctly!")


if __name__ == "__main__":
    model_size_analysis()



