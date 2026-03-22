import os 
import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from Tensor.tensor import BYTES_PER_FLOAT32
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from quantization import BYTES_PER_INT8, INT8_MAX_VALUE, INT8_MIN_VALUE, MB_TO_BYTES, QuantizedLinear, quantize_int8,dequantize_int8,__collect_layer_inputs,_quantize_single_layer, quantize_model,_measure_layer_bytes,analyze_model_sizes
import numpy as np
from typing import Tuple,Dict,List,Optional


def quantization_memory_analysis():
    """
    This function analyses  memory reduction across different model sizes
    """
    model_sizes = [
        ("Small", 1_000_000),
        ("Medium", 10_000_000),
        ("Large", 100_000_000),
    ]

    print(f"{'Model':<10} {'FP32 (MB)':<12} {'INT8 (MB)':<12} {'Reduction':<12}")
    print("-" * 50)

    for name,params in model_sizes:
        fp32_mb = params *BYTES_PER_FLOAT32 /MB_TO_BYTES
        int8_mb = params * BYTES_PER_INT8 / MB_TO_BYTES
        reduction = fp32_mb / int8_mb

        print(f"{name:<10} {fp32_mb:>10.1f}  {int8_mb:>10.1f}  {reduction:>10.1f}x")

    print("\nKey Insight: Memory reduction is consistent at 4x across all model sizes")
    print("This enables deployment on memory-constrained devices")

if __name__ == "__main__":
    quantization_memory_analysis()