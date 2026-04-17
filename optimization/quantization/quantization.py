
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

        #dequantize weights and restore the original Linear layout
        # Linear.forward uses x @ weight.T because weights are stored as
        # (out_features, in_features).
        weight_fp32 = dequantize_int8(self.q_weight,self.weight_scale,self.weight_zero_point)

        #perform computation 
        result = x.matmul(weight_fp32.transpose(-2, -1))

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

"""
## Model Quantization
We follow two steps
   1.Collecing layer inputs- forward calibration data through preceding layers to get the activation
     distribution at each layer's input.
    2. Quantize single layer - replacing one Linear layer with its QuantizedLinear equivalent

Then the composition function `quantize_model()` ties them together to transform a full model.

```
Model Transformation Process:

Input Model:                    Quantized Model:
┌─────────────────────────────┐    ┌─────────────────────────────┐
│ layers[0]: Linear(784, 128) │    │ layers[0]: QuantizedLinear  │
│ layers[1]: ReLU()           │    │ layers[1]: ReLU()           │
│ layers[2]: Linear(128, 64)  │ →  │ layers[2]: QuantizedLinear  │
│ layers[3]: ReLU()           │    │ layers[3]: ReLU()           │
│ layers[4]: Linear(64, 10)   │    │ layers[4]: QuantizedLinear  │
└─────────────────────────────┘    └─────────────────────────────┘
   Memory: 100%                      Memory: ~25%
   Interface: Same                   Interface: Identical
```
"""

"""
## Collecting Layer Inputs

The helper function declared below helps us know what the inputs look like at runtime.
This helper forwards calibration samples through all preceding layers to collect the activation tensors that
arrive at a given layer index
```
Calibration Data Flow for Layer at Index i:

  Sample Data          Layers 0..i-1          Activations at Layer i
  ┌──────────┐      ┌──────────────────┐      ┌──────────────────┐
  │ sample_0  │ ──→ │ forward through  │ ──→  │ activation_0     │
  │ sample_1  │ ──→ │ preceding layers │ ──→  │ activation_1     │
  │ ...       │     │ (0, 1, ..., i-1) │      │ ...              │
  │ sample_N  │ ──→ │                  │ ──→  │ activation_N     │
  └──────────┘      └──────────────────┘      └──────────────────┘
```
"""

def __collect_layer_inputs(model,layer_index:int,calibration_data:List[Tensor],max_samples:int =10) ->List[Tensor]:
    """
    Forward calibration data through preceding layers to collect inputs for a specific layer.


    Args:
        model:Model with .layers attribute (Sequential pattern)
        layer_index: Index of thr layer we want inputs for
        calibration_data:List of sample input tensors
        max_samples:Maximum numbers of samples to process

    Returns:
        List of Tensor activations arriving at layer_index

     EXAMPLE:
    >>> model = Sequential(Linear(4, 8), ReLU(), Linear(8, 3))
    >>> samples = [Tensor(np.random.randn(1, 4)) for _ in range(5)]
    >>> inputs_at_layer2 = _collect_layer_inputs(model, 2, samples)
    >>> print(len(inputs_at_layer2))  # 5 activation tensors
    5

    """
    sample_inputs = []
    for data in calibration_data[:max_samples]:
        x = data 
        for j in range(layer_index):
            x = model.layers[j].forward(x)
        sample_inputs.append(x)

    return sample_inputs


"""
   Quantizing a single Layer 
This helper function takes one Linear layer, wraps it in a QuantizedLinear, and optionally
calibrates it using pre-allocated activation samples.
This is the atomic operation that `quantize_model` applies to each eligible layer.

```
Single Layer Quantization:

  Linear Layer          QuantizedLinear
  ┌──────────────┐      ┌──────────────────────────┐
  │ weight: FP32 │  →   │ q_weight: INT8           │
  │ bias: FP32   │      │ q_bias: INT8             │
  │              │      │ weight_scale, zero_point  │
  └──────────────┘      │ calibrated: Yes/No       │
                        └──────────────────────────┘
       4 bytes/param          1 byte/param + overhead
```
"""

def _quantize_single_layer(layer:Linear,calibration_inputs:Optional[List[Tensor]]=None):
    """
    Quantizing a single Linear layer and optionally calibrating it.

    Args:
        layer:Liner layer to quantize
        calibration_inputs : Optinal list activations tensors for calibration

    Returns:
         QuantizedLinear :The quantized replacement layer

    EXAMPLE:
    >>> original = Linear(8, 3)
    >>> original.weight = Tensor(np.random.randn(8, 3) * 0.5)
    >>> quantized = _quantize_single_layer(original)
    >>> print(quantized.q_weight.data.dtype)
    int8
    """

    quantized_layer = QuantizedLinear(layer)

    if calibration_inputs is not None:
        quantized_layer.calibrate(calibration_inputs)

    return quantized_layer


"""
Model Quantization 
- Here all the helper functions are compressed into the full model quantization function.
-For each limear layer in the model, we collect its calibration inputs and replace it with a quatized version

```
quantize_model() orchestrates the full pipeline:

  For each layer in model.layers:
      │
      ├── isinstance(layer, Linear)?
      │   ├── YES → _collect_layer_inputs()  → calibration activations
      │   │         _quantize_single_layer()  → QuantizedLinear
      │   │         Replace model.layers[i]
      │   │
      │   └── NO  → Keep unchanged (ReLU, etc.)
```
"""

def quantize_model(model,calibration_data:Optional[List[Tensor]]=None) -> None:
    """
    Quantized all Liear layers in a model in-place

    Args:
        model:Model to quantize (with .layers or similar structure)
        calibration_data:Optional list of sample inputs for calibration

    Returns:
        None (modifies model in-place)

     EXAMPLE:
    >>> layer1 = Linear(10, 5)
    >>> activation = ReLU()
    >>> layer2 = Linear(5, 2)
    >>> model = Sequential(layer1, activation, layer2)
    >>> quantize_model(model)
    >>> # Now model uses quantized layers
    """

    if hasattr(model,'layers'):
        for i, layer in enumerate(model.layers):
            if isinstance(layer,Linear):
                #collect calibration inputs if data provided
                cal_inputs = None
                if calibration_data is not None:
                    cal_inputs = __collect_layer_inputs(model,i,calibration_data)

                #replace with quantized version
                model.layers[i]= _quantize_single_layer(layer,cal_inputs)

    elif isinstance(model,Linear):
        raise ValueError(
            f"Cannot quantize single Linear layer in-place\n"
            f"   quantize_model() modifies model in-place, but a single layer has no container to modify\n"
            f"    In-place modification requires a container (like Sequential) that holds layer references\n"
            f"      Use QuantizedLinear directly: quantized_layer = QuantizedLinear(your_linear_layer)"
        )

    else:
        raise ValueError(
            f"Unsupported model type for quantization: {type(model).__name__}\n"
            f"    quantize_model() expects a model with .layers attribute (like Sequential)\n"
            f"   The function iterates through model.layers to find and replace Linear layers\n"
            f"    Wrap your layers in Sequential: model = Sequential(layer1,activation,layer2)"

        )


"""
## Model Size Comparison
- To compare memory usage between original and quantized models, we need to measure
bytes at individual layer level first, then aggregate.
- We follow two steps:
   1. Measure one layer - counting parameters and bytes for a single layer, handling both FP32 (Linear) and INT8 (QuantizedLinear) layer correctly.
   2. Aggregate and compare - Sum across all layers and compute compression metrics.

```
Per-Layer Measurement:

  Layer Type          Measurement Strategy
  ┌──────────────┐    ┌───────────────────────────────────┐
  │ Linear       │ →  │ params × 4 bytes (FP32)           │
  │ QuantizedLin │ →  │ memory_usage() dict (INT8 + ovhd) │
  │ ReLU/other   │ →  │ 0 params, 0 bytes (no weights)    │
  └──────────────┘    └───────────────────────────────────┘
```

"""

"""
## Measuring a Single Layer
This heper function measres the parameter count and byte usage for one layer.
It handles the key distiction: FP32 layers store parameters at 4 bytes each,
while QuantizedLinear layers use INT8 storage with a small overhead for scale/zero_point metadata.

```
Byte Accounting per Layer Type:

  FP32 Linear:                     QuantizedLinear:
  ┌─────────────────────────┐      ┌─────────────────────────────────┐
  │ weight: N × 4 bytes     │      │ q_weight: N × 1 byte            │
  │ bias:   M × 4 bytes     │      │ q_bias:   M × 1 byte            │
  │                         │      │ overhead: ~8 bytes (scale+zp)    │
  │ Total: (N+M) × 4       │      │ Total: (N+M) × 1 + overhead     │
  └─────────────────────────┘      └─────────────────────────────────┘
```

"""
def _measure_layer_bytes(layer,is_quantized:bool=False)->Tuple[int,int]:
    """
    Measures parameter count and byte usage for single layer.

    Args:
       layer:A single layer (Linear,QuantizedLinear,ReLU)
       is_quantized: whether to measure as quantized (uses memory_usage() for QuantizedLinear)

    Returns:
         Tuple of (param_count,byte_count)

    EXAMPLE:
    >>> linear = Linear(100, 50)
    >>> params, bytes_ = _measure_layer_bytes(linear)
    >>> print(f"Params: {params}, Bytes: {bytes_}")
    Params: 5050, Bytes: 20200
    """

    if is_quantized and isinstance(layer,QuantizedLinear):
        memory_info = layer.memory_usage()
        param_count = sum(p.data.size for p in layer.parameters())
        return param_count,memory_info['quantized_bytes']

    if hasattr(layer,'parameters'):
        params = layer.parameters()
        param_count = sum(p.data.size for p in params)
        byte_count = param_count * BYTES_PER_FLOAT32
        return param_count,byte_count

    return 0 , 0


def analyze_model_sizes(original_model,quantized_model) -> Dict[str,float]:
    """
    This function compares memory usage between original and quantized models.

    Args:
        original_model:Model before quantization
        quantized_model:Model after quantization

    Returns:
         Dictionary with compression metrics
    
    >>> layer1 = Linear(100, 50)
    >>> layer2 = Linear(50, 10)
    >>> model = Sequential(layer1, layer2)
    >>> quantize_model(model)
    >>> stats = analyze_model_sizes(model, model)
    >>> print(f"Reduced to {stats['compression_ratio']:.1f}x smaller") 
    
    """
    original_params = 0
    original_bytes = 0
    for layer in original_model.layers:
        p,b = _measure_layer_bytes(layer,is_quantized=False)
        original_params +=p
        original_bytes += b


    #measuring quantized model
    quantized_params = 0
    quantized_bytes = 0
    for layer in quantized_model.layers:
        is_q = isinstance(layer,QuantizedLinear)
        p,b = _measure_layer_bytes(layer,is_quantized=is_q)
        quantized_params += p
        quantized_bytes += b

    compression_ratio = original_bytes / quantized_bytes if quantized_bytes > 0 else 1.0
    memory_saved = original_bytes - quantized_bytes

    return {
        'original_params': original_params,
        'quantized_params': quantized_params,
        'original_bytes': original_bytes,
        'quantized_bytes': quantized_bytes,
        'compression_ratio': compression_ratio,
        'memory_saved_mb': memory_saved / MB_TO_BYTES,
        'memory_saved_percent': (memory_saved / original_bytes) * 100 if original_bytes > 0 else 0
    }

class Quantizer:
    """
    A complete quantization system class

    Provides INTT8 quantization with calibration for 4x memory reduction

    This class delegates to the standalone functions (quantize_int8,dequantize_int8)
    hence providing a clean OOP interface for experiments

    API that exist here:
      - Standalone quantize_model(): modifies model in-place
      - Quantizer.quantize_model(): Returns stats dict (for experiments)

    """
    @staticmethod
    def quantize_tensor(tensor:Tensor) ->Tuple[Tensor,float,int]:
        """
        Quantizes FP32 tensor to INT8
        It delegates to quantize_int8()
        """
        return quantize_int8(tensor)

    @staticmethod
    def quantize_tensor(q_tensor:Tensor,scale:float,zero_point:int)->Tensor:
        """
        Dequantizes INT8 tensor back to FP32
        It delegates to dequantize_int8()
        """
        return dequantize_int8(q_tensor,scale,zero_point)

    @staticmethod 
    def quantize_model(model,calibration_data:Optional[List[Tensor]]=None)->Dict[str,any]:
        """
        Quantizes all linear layersin model and returns the stats

        Unlike the individula quantize_model() which modifies in-place
        It returns a dictionary with quantization info for experiments

        Returns:
            Dict with quantized_layers,original_size_mb,quantized_size_mb,compression_ratio
        """
        quantized_layers = {}
        original_size = 0
        total_elements = 0
        param_idx = 0

        #iterates through model parameters
        for layer in model.layers:
            for param in layer.parameters():
                param_size = param.data.nbytes
                original_size += param_size
                total_elements += param.data.size

                #quantizing parameters using the individual function
                q_param,scale,zp = quantize_int8(param)

                quantized_layers[f'param_{param_idx}'] = {
                    'quantized': q_param,
                    'scale': scale,
                    'zero_point': zp,
                    'original_shape': param.data.shape
                }
                param_idx += 1

        #INT8 uses 1byte per element
        quantized_size = total_elements

        return {
                 'quantized_layers': quantized_layers,
            'original_size_mb': original_size / MB_TO_BYTES,
            'quantized_size_mb': quantized_size / MB_TO_BYTES,
            'compression_ratio': original_size / quantized_size if quantized_size > 0 else 1.0

            }

    @staticmethod
    def compare_models(original_model,quantized_info:Dict) ->Dict[str,float]:
        """
        Compares memory usage between original and quantized models
        """
        return{
             'original_mb': quantized_info['original_size_mb'],
            'quantized_mb': quantized_info['quantized_size_mb'],
            'compression_ratio': quantized_info['compression_ratio'],
            'memory_saved_mb': quantized_info['original_size_mb'] - quantized_info['quantized_size_mb']
        }
    

def verify_quantization_works(original_model,quantized_model):
    """
    This function verifies whether quantization actually redces memory using real .nbytes measurements

    Args:
       original_model:Model with FP32 parameters (Sequential with .parameters())
       quantized_model:model with INT8 quantized parameters (Sequential with QuantizedLinear layers)

    Retuns:
         dict:Verification results with actual_reduction,original_mb,quantized_mb


     Example:
        >>> original = Sequential(Linear(100, 50))
        >>> quantized = Sequential(Linear(100, 50))
        >>> quantize_model(quantized)
        >>> results = verify_quantization_works(original, quantized)
        >>> assert results['actual_reduction'] >= 3.5  # Real 4× reduction
    """

    #collect actual bytes from original FP32 model
    original_bytes = sum(
        param.data.nbytes for param in original_model.parameters()
        if hasattr(param,'data') and hasattr(param.data,'nbytes')

    )

    #collecting actual bytes from quantized INT8 model
    quantized_bytes = sum(
        layer.q_weight.data.nbytes + (layer.q_bias.data.nbytes if layer.q_bias is None else 0)
        for layer in quantized_model.layers
        if isinstance(layer,QuantizedLinear)
    )

    #calculating actual reduction
    actual_reduction = original_bytes /max(quantized_bytes,1)

    #display results
    print(f"   Original model: {original_bytes / MB_TO_BYTES:.2f} MB (FP32)")
    print(f"   Quantized model: {quantized_bytes / MB_TO_BYTES:.2f} MB (INT8)")
    print(f"   Actual reduction: {actual_reduction:.1f}x")
    print(f"   {'PASS' if actual_reduction >= 3.5 else 'FAIL'} Meets 4x reduction target")

    #verifying whether target is met
    assert actual_reduction >= 3.5, f"Expected ~4x reduction, got {actual_reduction:.1f}x"

    print(f"\nVERIFIED: Quantization achieves real {actual_reduction:.1f}x memory reduction!")

    return {
        'actual_reduction': actual_reduction,
        'original_mb': original_bytes / MB_TO_BYTES,
        'quantized_mb': quantized_bytes / MB_TO_BYTES,
        'verified': actual_reduction >= 3.5
    }
