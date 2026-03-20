import numpy as np
import time 
from typing import Tuple,Dict,List,Optional
import warnings
import os 
import sys



##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from Tensor.tensor import BYTES_PER_FLOAT32

#constants for INT8 quantization
INT8_MIN_VALUE = -128
INT8_MAX_VALUE = 127
INT8_RANGE = 256 #number of possible INT8 values (from -128 to 127 inclusive)
EPILSON = 1e-8 #small value for numerical stability (constant tensor detection)

#constants for memory calculations
BYTES_PER_FLOAT32 = 4  # Standard float32 size in bytes
BYTES_PER_INT8 = 1  # INT8 size in bytes
MB_TO_BYTES = 1024 * 1024  # Megabytes to bytes conversion
"""
Quantization engine system architecture

┌─────────────────────────────────────────────────────────────┐
│                    Layer 4: Model Quantization              │
│  quantize_model() - Convert entire neural networks          │
├─────────────────────────────────────────────────────────────┤
│                    Layer 3: Layer Quantization              │
│  QuantizedLinear - Quantized linear transformations         │
├─────────────────────────────────────────────────────────────┤
│                    Layer 2: Tensor Operations               │
│  quantize_int8() - Core quantization algorithm              │
│  dequantize_int8() - Restore to floating point              │
├─────────────────────────────────────────────────────────────┤
│                    Layer 1: Foundation                      │
│  Scale & Zero Point Calculation - Parameter optimization    │
└─────────────────────────────────────────────────────────────┘

**Core Functions:**
- `quantize_int8()` - Convert FP32 tensors to INT8
- `dequantize_int8()` - Convert INT8 back to FP32
- `QuantizedLinear` - Quantized version of Linear layers
- `quantize_model()` - Quantize entire neural networks

**Key Features:**
- **Automatic calibration** - Find optimal quantization parameters
- **Error minimization** - Preserve accuracy during compression
- **Memory tracking** - Measure actual savings achieved
- **Production patterns** - Industry-standard algorithms
"""

def quantize_int8(tensor:Tensor)->Tuple[Tensor,float,int]:
    """
    Quantizes FP32 tensor to INT8 using symmetric quantization.

    Args:
        tensor:Input FP32 tensor to quantize

    Returns:
        q_tensor:Quantized INT8 tensor
        scale:Scaling factor (float)
        zero_point :Zero point offset (int)


     EXAMPLE:
    >>> tensor = Tensor([[-1.0, 0.0, 2.0], [0.5, 1.5, -0.5]])
    >>> q_tensor, scale, zero_point = quantize_int8(tensor)
    >>> print(f"Scale: {scale:.4f}, Zero point: {zero_point}")
    Scale: 0.0118, Zero point: -43

    """
    data = tensor.data

    #finding the dynamic range
    min_val = float(np.min(data))
    max_val = float(np.max(data))

    #handling edge case (costant tensor)
    if abs(max_val - min_val) <EPILSON:
        scale = 1.0
        zero_point = 0
        quantized_data = np.zeros_like(data,dtype=np.int8)
        return Tensor(quantized_data),scale,zero_point


    #Calculating scale and zero_point for standard quantization
    #mapping [min_val,max_val] to [INT8_MIN_VALUE,INT8_MAX_VALUE] (INT8 range)
    scale= (max_val - min_val) / (INT8_RANGE -1)
    zero_point = int(np.round(INT8_MIN_VALUE- min_val /scale))

    #clamp zero_point to vaid INT8 range
    zero_point = int(np.clip(zero_point,INT8_MIN_VALUE,INT8_MAX_VALUE))

    #apply quantization formula: q = (x/scale) + zero_point
    quantized_data = np.round(data/scale+ zero_point)

    #clamping to INT8 range and convert to int8
    quantized_data = np.clip(quantized_data,INT8_MIN_VALUE,INT8_MAX_VALUE).astype(np.int8)

    return Tensor(quantized_data),scale,zero_point


def dequantize_int8(q_tensor:Tensor,scale:float,zero_point:int) ->Tensor:
    """
    Dequantize INT8 tensor back to FP32

    Args:
        q_tensor:Quantized INT8 tensor
        scale:Scaling factor from Quantization
        zero_point:Zero point offset from quantization

    Returns:
         Reconstructed FP32 tensor

     EXAMPLE:
    >>> q_tensor = Tensor([[-100, 0, 50]])  # INT8 values
    >>> scale, zero_point = 0.02, -25
    >>> fp32_tensor = dequantize_int8(q_tensor, scale, zero_point)
    >>> print(fp32_tensor.data)
    [[-1.5, 0.5, 1.5]]  # Reconstructed FP32 values

    """
    dequantized_data = (q_tensor.data.astype(np.float32)-zero_point)*scale
    return Tensor(dequantized_data)


class QuantizedLinear:
    """
    Quantized version of Linear layer using INT8 arithmetic
    """

    def __init__(self,linear_layer:Linear):
        """
        Creating quantized version of existing linear layer.

        EXAMPLE:
        >>> original_layer = Linear(128, 64)
        >>> original_layer.weight = Tensor(np.random.randn(128, 64) * 0.1)
        >>> original_layer.bias = Tensor(np.random.randn(64) * 0.01)
        >>> quantized_layer = QuantizedLinear(original_layer)
        >>> print(quantized_layer.q_weight.data.dtype)
        int8
        """
        self.original_layer = linear_layer

        #quantized weights
        self.q_weight,self.weight_scale,self.weight_zero_point = quantize_int8(linear_layer.weight)

        #quantizing bias if it exists
        if linear_layer.bias is not None:
            self.q_bias,self.bias_scale,self.bias_zero_point = quantize_int8(linear_layer.bias)
        else:
            self.q_bias = None
            self.bias_scale = None 
            self.bias_zero_point = None 

        #store input quantization parameters (set during calibration)
        self.input_scale = None
        self.input_zero_point =None 
        # Note: do not overwrite bias quantization parameters here.
        # `self.bias_scale` / `self.bias_zero_point` are set above by quantize_int8().


    def calibrate(self,sample_inputs:List[Tensor]):
        """
        Calibrates input quantization parameters using sample data.

        EXAMPLE:
        >>> layer = QuantizedLinear(Linear(64, 32))
        >>> sample_data = [Tensor(np.random.randn(1, 64)) for _ in range(10)]
        >>> layer.calibrate(sample_data)
        >>> print(layer.input_scale is not None)
        True

        """
        #collect all input values
        all_values = []
        for inp in sample_inputs:
            all_values.extend(inp.data.flatten())

        all_values = np.array(all_values)

        #calculating input quantization paramters 
        min_val = float(np.min(all_values))
        max_val = float(np.max(all_values))

        if abs(max_val- min_val) <EPILSON:
            self.input_scale = 1.0
            self.input_zero_point = 0

        else:
            self.input_scale = (max_val - min_val) /(INT8_RANGE - 1)
            self.input_zero_point = int(np.round(INT8_MIN_VALUE - min_val / self.input_scale))
            self.input_zero_point = np.clip(self.input_zero_point,INT8_MIN_VALUE,INT8_MAX_VALUE)

    def forward(self,x:Tensor)->Tensor:
        """
        Forward pass with quantized computation.


         EXAMPLE:
        >>> layer = QuantizedLinear(Linear(4, 3))
        >>> x = Tensor(np.array([[1.0, 2.0, 3.0, 4.0]]))
        >>> output = layer.forward(x)
        >>> print(output.shape)
        (1, 3)
        """

        #dequantize weights 
        weight_fp32 = dequantize_int8(self.q_weight,self.weight_scale,self.weight_zero_point)

        #perform computation 
        result = x.matmul(weight_fp32)

        #add bias if it exists 
        if self.q_bias is not None:
            bias_fp32 = dequantize_int8(self.q_bias,self.bias_scale,self.bias_zero_point)
            result = Tensor(result.data + bias_fp32.data)

        return result 

    
    def __call__(self,x:Tensor)->Tensor:
        """
        Allows the quantized linear layer to be called like a function
        """
        return self.forward(x)

    
    def parameters(self)->List[Tensor]:
        """
        Returning quantized paramters
        """
        params = [self.q_weight]
        if self.q_bias is not None:
            params.append(self.q_bias)
        return params 

    def memory_usage(self)->Dict[str,float]:
        """
        Calculate memory usage in bytes
        """
        #original FP32 usage
        original_weight_bytes = self.original_layer.weight.data.size *BYTES_PER_FLOAT32
        original_bias_bytes = 0
        if self.original_layer.bias is not None:
            original_bias_bytes = self.original_layer.bias.data.size *BYTES_PER_FLOAT32

        #quantized INT8 usage
        quantized_weight_bytes = self.q_weight.data.size * BYTES_PER_INT8
        quantized_bias_bytes = 0
        if self.q_bias is not None:
            quantized_bias_bytes = self.q_bias.data.size *BYTES_PER_INT8

        #add overhead for scales and zero points
        #2 floats: one scale for weight, one scale for bias 
        overhead_bytes = BYTES_PER_FLOAT32 * 2

        quantized_total = quantized_weight_bytes + quantized_bias_bytes + overhead_bytes
        original_total = original_weight_bytes + original_bias_bytes

        return {
            'original_bytes':original_total,
            'quantized_bytes': quantized_total,
            'compression_ratio':original_total / quantized_total if quantized_total > 0 else 1.0
        }

        