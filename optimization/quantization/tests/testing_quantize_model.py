import os 
import sys



##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from quantization import INT8_MAX_VALUE, INT8_MIN_VALUE, QuantizedLinear, quantize_int8,dequantize_int8,__collect_layer_inputs,_quantize_single_layer, quantize_model
import numpy as np

def testing_quantize_model():
    """
    This function intends to test whether the 
    quantize_model function works correctly
    """

    #creating test model using explicit layer composition
    layer1 = Linear(4,8)
    activation =ReLU()
    layer2 = Linear(8,3)

    #intializing weight tensors 
    layer1.weight = Tensor(np.random.randn(4,8)*0.5)
    layer1.bias = Tensor(np.random.randn(8)*0.1)
    layer2.weight = Tensor(np.random.randn(8,3)*0.5)
    layer2.bias = Tensor(np.random.randn(3)*0.1)

    #uses Sequential from layers
    model = Sequential(
        layer1,
        activation,
        layer2
    )

    #test original model
    x = Tensor(np.random.randn(2,4))
    original_output = model.forward(x)

    #creates calibration data 
    calibration_data = [Tensor(np.random.randn(1,4)) for _ in range(5)]

    #Quantize model
    quantize_model(model,calibration_data)

    #verifying layers were replaced
    assert isinstance(model.layers[0],QuantizedLinear)
    assert isinstance(model.layers[1],ReLU) #should remain unchanged
    assert isinstance(model.layers[2],QuantizedLinear)

    #test quantized model
    quantized_output = model.forward(x)

    #compare outputs 
    error = np.mean(np.abs(original_output.data - quantized_output.data))
    print(f"  Model quantization error: {error:.4f}")
    assert error < 0.2, f"Model quantization error too high: {error}"

    print("Model quantization works correctly!")

if __name__ == "__main__":
    testing_quantize_model()
