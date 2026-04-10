import os 
import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from Tensor.tensor import BYTES_PER_FLOAT32
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from quantization import BYTES_PER_INT8, INT8_MAX_VALUE, INT8_MIN_VALUE, MB_TO_BYTES, QuantizedLinear, quantize_int8,dequantize_int8,__collect_layer_inputs,_quantize_single_layer, quantize_model,_measure_layer_bytes,analyze_model_sizes,EPILSON
import numpy as np

def testing_module():
    """
    This function intends to whether the quantization module
    works correctly
    """
    print("\nTesting Quantization Module")

    #creating a realistic model using explicit composition 
    layer1 = Linear(784,128)
    activation1 = ReLU()
    layer2 = Linear(128,64)
    activation2 = ReLU()
    layer3 = Linear(64,10)
    model =Sequential(layer1,activation1,layer2,activation2,layer3)

    #intialize with realistic weights
    for layer in [layer1,layer2,layer3]:
        if isinstance(layer,Linear):
            #xavier intialization
            fan_in,fan_out = layer.weight.shape 
            std = np.sqrt(2.0/ (fan_in + fan_out))
            layer.weight = Tensor(np.random.rand(fan_in,fan_out)*std)
            layer.bias = Tensor(np.zeros(fan_out))

    #generate realistic calibration data 
    calibration_data = [Tensor(np.random.randn(1,784)*0.1) for _ in range(20)]

    #testing original model
    test_input = Tensor(np.random.randn(8,784)*0.1)
    original_output = model.forward(test_input)

    #quantizing the model
    quantize_model(model,calibration_data)

    #testing the quantized model
    quantized_output = model.forward(test_input)

    #verifying functioonality is preserved
    assert quantized_output.shape == original_output.shape ,"Output shape mismatch"

    #verifying reasonable accuracy preservation
    mse = np.mean((original_output.data - quantized_output.data)**2)
    relative_error = np.sqrt(mse) / (np.std(original_output.data) + EPILSON)
    assert relative_error < 0.1, f"Accuracy degradation too high: {relative_error:.3f}"

    #verifying memory savings
    #creating equivalent original model for comparison
    orig_layer1 = Linear(784,128)
    orig_act1 = ReLU()
    orig_layer2 = Linear(128,64)
    orig_act2 = ReLU()
    orig_layer3 = Linear(64,10)
    original_model = Sequential(orig_layer1,orig_act1,orig_layer2,orig_act2,orig_layer3)

    for i,layer in enumerate(model.layers):
        if isinstance(layer,QuantizedLinear):
            #restoring original weights for comparison
            original_model.layers[i].weight = dequantize_int8(
                layer.q_weight,layer.weight_scale,layer.weight_zero_point
            )
            if layer.q_bias is not None:
                original_model.layers[i].bias = dequantize_int8(
                    layer.q_bias,layer.bias_scale,layer.bias_zero_point
                )

        memory_comparison = analyze_model_sizes(original_model,model)
        assert memory_comparison['compression_ratio'] > 2.0, "Insufficient compression achieved"

    print(f"Compression achieved: {memory_comparison['compression_ratio']:.1f}x")
    print(f"Accuracy preserved: {relative_error:.1%} relative error")
    print(f"Memory saved: {memory_comparison['memory_saved_mb']:.1f}MB")

    #testing edge cases
    print("Testing edge cases..")

    #testing constant tensor quantization
    constant_tensor = Tensor([[1.0,1.0],[1.0,1.0]])
    q_const,scale_const,zp_const = quantize_int8(constant_tensor)
    assert scale_const == 1.0, "Constant tensor quantization failed"

    #testing zero tensor
    zero_tensor = Tensor([[0.0, 0.0], [0.0, 0.0]])
    q_zero, scale_zero, zp_zero = quantize_int8(zero_tensor)
    restored_zero = dequantize_int8(q_zero, scale_zero, zp_zero)
    assert np.allclose(restored_zero.data, 0.0, atol=1e-6), "Zero tensor restoration failed"

    print("Edge cases handled correctly")
    print("\n" + "=" * 50)

if __name__ == "__main__":
    testing_module()