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

def quantization_accuracy_analysis():
    """
    Analyzes accuracy vs memory trade-off for quantization
    """

    #simulating quantization impact on different layers types
    layer_types = [
         ("Embeddings", 0.99, "Low impact - lookup tables"),
        ("Attention", 0.97, "Moderate impact - many small ops"),
        ("MLP", 0.98, "Low impact - large matrix muls"),
        ("Output", 0.95, "Higher impact - final predictions")
    ]

    print(f"{'Layer Type':<15} {'Acc Retention':<15} {'Observation'}")
    print("-" * 50)

    for layer, retention, note in layer_types:
        print(f"{layer:<15} {retention:>13.1%}  {note}")

    print("\nKey Insight: Overall model accuracy retention: ~98-99% typical")
    print("Output layers most sensitive to quantization")

if __name__ == "__main__":
    quantization_accuracy_analysis()