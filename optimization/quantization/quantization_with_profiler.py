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
from optimization.profiling.profiling import Profiler


def quantization_with_profiler():
    """
    Demonstrates memory saving using Profiler from profiling module

    """
    print("Measuring Quantization Memory Savings with Profiler")
    print("=" * 70)

    profiler = Profiler()

    #creating a simple model
    model = Linear(512,256)
    model.name = "baseline_model"

    print("\nBEFORE: FP32 Model")
    print("-"*70)

    #measuring baseline
    param_count = profiler.count_parameters(model)
    input_shape = (32,512)
    memory_stats= profiler.measure_memory(model,input_shape)

    print(f"   Parameters: {param_count:,}")
    print(f"   Parameter memory: {memory_stats['parameter_memory_mb']:.2f} MB")
    print(f"   Peak memory: {memory_stats['peak_memory_mb']:.2f} MB")
    print(f"   Precision: FP32 (4 bytes per parameter)")

    #quantizing the model 
    print("\nQuantizing to INT8..")
    #quantize_model expects model with .layers attribute, so we wrap single layer in Sequential
    wrapped_model = Sequential(model)
    quantize_model(wrapped_model) # modifies model in-place, returns None
    quantized_model = wrapped_model.layers[0] if wrapped_model.layers else model
    quantized_model.name = "quantized_model"

    print("\nAFTER: INT8 Quantized Model")
    print("-" * 70)

    #measures quantized (simulated in practice INT8 uses 1 byt)
    # for demos we show the theoritical savings
    quantized_param_count = profiler.count_parameters(quantized_model)
    theoretical_memory_mb = param_count * BYTES_PER_INT8 / MB_TO_BYTES

    print(f"   Parameters: {quantized_param_count:,} (same count, different precision)")
    print(f"   Parameter memory (theoretical): {theoretical_memory_mb:.2f} MB")
    print(f"   Precision: INT8 (1 byte per parameter)")

    print("\nMemory Savings")
    print("="*70)

    savings_ratio = memory_stats['parameter_memory_mb'] / theoretical_memory_mb
    savings_percent = (1 - 1/savings_ratio) * 100
    savings_mb = memory_stats['parameter_memory_mb'] - theoretical_memory_mb

    print(f"   Compression ratio: {savings_ratio:.1f}x smaller")
    print(f"   Memory saved: {savings_mb:.2f} MB ({savings_percent:.1f}% reduction)")
    print(f"   Original: {memory_stats['parameter_memory_mb']:.2f} MB -> Quantized: {theoretical_memory_mb:.2f} MB")

    print("\nKey Insight:")
    print(f"   INT8 quantization reduces memory by 4x (FP32 -> INT8)")
    print(f"   This enables: 4x larger models, 4x bigger batches, or 4x lower cost!")
    print(f"   Critical for edge devices with limited memory (mobile, IoT)")
    print("\nThis is the power of quantization: same functionality, 4x less memory!")

if __name__ == "__main__":
    quantization_with_profiler()