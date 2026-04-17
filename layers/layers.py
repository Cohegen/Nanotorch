import numpy as np
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
from activations.activations import ReLU, Sigmoid 

#constants for weight initialization
XAVIER_SCALE_FACTOR = 1.0 #xavier/glorot intializations uses sqrt(1/fan_in)
HE_SCALE_FACTOR = 2.0 #He intialization uses sqrt(2/fan_in) for ReLU

#constants for dropout
DROPOUT_MIN_PROB = 0.0 #minimum dropout probability
DROPOUT_MAX_PROB = 1.0 #maximum dropout probability (drop everything)

"""
Here we implement two essential layers
1. **Linear Layer** - The workhorse of neural network
2. **Dropout Layer** - Prevents overfitting

###Key Design principles:
- All methods defined INSIDE classes.
- Forward methods return new tensors, preserving immutability
- parameters() method enables optimizer intergration
"""

class Layer:
    """
    Base class for all neural network layers.
    """
    def __init__(self):
        self.training = True
        self._parameters = {}
        self._buffers = {}
        self._modules = {}

    def __setattr__(self, name, value):
        if isinstance(value, Tensor):
            if getattr(value, 'requires_grad', False):
                self._parameters[name] = value
            else:
                self._buffers[name] = value
        elif isinstance(value, Layer):
            self._modules[name] = value
        super().__setattr__(name, value)

    def train(self, mode=True):
        """Sets the layer to training mode."""
        self.training = mode
        for module in self._modules.values():
            module.train(mode)

    def eval(self):
        """Sets the layer to evaluation mode."""
        return self.train(False)

    def register_buffer(self, name, tensor):
        """Adds a buffer to the module."""
        if not isinstance(tensor, Tensor):
            tensor = Tensor(tensor)
        self._buffers[name] = tensor
        setattr(self, name, tensor)

    def apply(self, fn):
        """Applies a function recursively to every submodule."""
        for module in self._modules.values():
            module.apply(fn)
        fn(self)
        return self

    def parameters(self):
        """
        Return list of all trainable parameters, including those of submodules.
        """
        params = list(self._parameters.values())
        for module in self._modules.values():
            params.extend(module.parameters())
        return params

    def named_parameters(self, prefix=''):
        """Returns an iterator over module parameters, yielding both the name and the parameter."""
        for name, param in self._parameters.items():
            yield prefix + ('.' if prefix else '') + name, param
        for name, module in self._modules.items():
            yield from module.named_parameters(prefix + ('.' if prefix else '') + name)

    def named_buffers(self, prefix=''):
        """Returns an iterator over module buffers, yielding both the name and the buffer."""
        for name, buf in self._buffers.items():
            yield prefix + ('.' if prefix else '') + name, buf
        for name, module in self._modules.items():
            yield from module.named_buffers(prefix + ('.' if prefix else '') + name)

    def state_dict(self, prefix='', destination=None):
        """Returns a dictionary containing parameters and registered buffers."""
        if destination is None:
            destination = {}

        for name, param in self._parameters.items():
            key = prefix + ('.' if prefix else '') + name
            destination[key] = np.array(param.data, copy=True)

        for name, buf in self._buffers.items():
            key = prefix + ('.' if prefix else '') + name
            destination[key] = np.array(buf.data, copy=True)

        for name, module in self._modules.items():
            module_prefix = prefix + ('.' if prefix else '') + name
            module.state_dict(prefix=module_prefix, destination=destination)

        return destination

    def load_state_dict(self, state_dict, strict=True):
        """Loads parameters and buffers from a state dictionary."""
        expected = self.state_dict()
        expected_keys = set(expected.keys())
        provided_keys = set(state_dict.keys())

        missing_keys = sorted(expected_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_keys)

        if strict and missing_keys:
            raise KeyError(f"Missing keys in state_dict: {missing_keys}")
        if strict and unexpected_keys:
            raise KeyError(f"Unexpected keys in state_dict: {unexpected_keys}")

        for name, param in self.named_parameters():
            if name in state_dict:
                _load_tensor_value(param, state_dict[name], name)

        for name, buf in self.named_buffers():
            if name in state_dict:
                _load_tensor_value(buf, state_dict[name], name)

        return {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self, x, *args, **kwargs):
        return self.forward(x, *args, **kwargs)

    def to(self, device):
        """Move to device (only CPU supported in NanoTorch)"""
        return self

    def __repr__(self):
        return f"{self.__class__.__name__}()"

   
class Parameter(Tensor):
    """
    A kind of Tensor that is to be considered a module parameter.
    """
    def __init__(self, data):
        super().__init__(data, requires_grad=True)

class Linear(Layer):
    """
    Linear (fully connected) layer: y = xW^T + b

    This matches PyTorch's implementation where weights are (out_features, in_features).
    """   

    def __init__(self,in_features,out_features,bias=True):
        """
        Intializing linear layer with proper weight intialization
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        #Xavier/Glorot intialization for stable gradients
        scale = np.sqrt(XAVIER_SCALE_FACTOR/ in_features)
        # Weight shape is (out_features, in_features) to match PyTorch
        weight_data = np.random.randn(out_features, in_features)* scale
        self.weight = Parameter(weight_data)

        #initializze bias to zeros of None
        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Parameter(bias_data)
        else:
            self.bias = None


    def forward(self,x):
        """
        Forward pass through linear layer: y = x @ W.T + b
        """
        # linear transformation y = x @ W.T
        # self.weight is (out, in), so self.weight.transpose() is (in, out)
        output = x.matmul(self.weight.transpose(-2, -1))

        ##add bias if present
        if self.bias is not None:
            output = output + self.bias 

        return output

    def __repr__(self):
        """String representation for debugging"""
        bias_str = f", bias={self.bias is not None}"
        return f"Linear(in_features={self.in_features},out_features={self.out_features}{bias_str})"


class Dropout(Layer):
    """
    Dropout layer for regularization.

    During training: randomly zeros elements with probability p, scales survivors by 1/(1-p)
    During inference: passes input through unchanged

    This prevent overfitting by forcing network to not rely on specific neurons.
    """

    def __init__(self, p=0.5):
        """
        Initializing dropout layer.
        """
        super().__init__()
        if not DROPOUT_MIN_PROB <= p <= DROPOUT_MAX_PROB:
            raise ValueError(f"Dropout probability must be between {DROPOUT_MIN_PROB} and {DROPOUT_MAX_PROB}, got {p}")

        self.p = p

    def forward(self,x):
        """
        Forward pass through dropout layer.
        """
        if not self.training or self.p == DROPOUT_MIN_PROB:
            #during inference or no dropoout, pass through unchange
            return x
        if self.p ==DROPOUT_MAX_PROB:
            #Drop everything
            return Tensor(np.zeros_like(x.data))

        #during training apply dropout
        keep_prob = 1.0 - self.p 

        #create random mask: True where we keep elements
        mask = np.random.random(x.data.shape) < keep_prob 

        #applying mask and scale
        mask_tensor = Tensor(mask.astype(np.float32))
        scale = Tensor(np.array(1.0/keep_prob))

        #using tensor operations: x*mask*scale
        output = x*mask_tensor*scale
        return output

    def __repr__(self):
        return f"Dropout(p={self.p})"

class Sequential:
    """
    A container that chains layers together sequentially.

    """ 

    def __init__(self, *layers):
        """Initialize with layers to chain together"""
        #accepting both Sequential(layer1,layer2) and Sequential([layer1,layer2])
        if len(layers) == 1 and isinstance(layers[0],(list,tuple)):
            self.layers =list(layers[0])
        else:
            self.layers = list(layers)
        self.training = True

    def train(self):
        """Sets all layers to training mode."""
        self.training = True
        for layer in self.layers:
            if hasattr(layer, 'train'):
                layer.train()

    def eval(self):
        """Sets all layers to evaluation mode."""
        self.training = False
        for layer in self.layers:
            if hasattr(layer, 'eval'):
                layer.eval()

    def forward(self,x):
        """Forward pass through all layers sequentially."""
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, training=self.training)
            else:
                x = layer.forward(x)
        return x

    def __call__(self,x):
        """Allows the model to be called like a function. """
        return self.forward(x)

    def parameters(self):
        """Collect all parameters from all layers."""
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params 

    def named_parameters(self, prefix=''):
        """Returns an iterator over parameters in child layers."""
        for index, layer in enumerate(self.layers):
            if hasattr(layer, 'named_parameters'):
                layer_prefix = prefix + ('.' if prefix else '') + f"layers.{index}"
                yield from layer.named_parameters(layer_prefix)

    def named_buffers(self, prefix=''):
        """Returns an iterator over registered buffers in child layers."""
        for index, layer in enumerate(self.layers):
            if hasattr(layer, 'named_buffers'):
                layer_prefix = prefix + ('.' if prefix else '') + f"layers.{index}"
                yield from layer.named_buffers(layer_prefix)

    def state_dict(self, prefix='', destination=None):
        """Returns a dictionary containing parameters and buffers for all child layers."""
        if destination is None:
            destination = {}
        for index, layer in enumerate(self.layers):
            if hasattr(layer, 'state_dict'):
                layer_prefix = prefix + ('.' if prefix else '') + f"layers.{index}"
                layer.state_dict(prefix=layer_prefix, destination=destination)
        return destination

    def load_state_dict(self, state_dict, strict=True):
        """Loads a state dictionary into all child layers."""
        expected = self.state_dict()
        expected_keys = set(expected.keys())
        provided_keys = set(state_dict.keys())

        missing_keys = sorted(expected_keys - provided_keys)
        unexpected_keys = sorted(provided_keys - expected_keys)

        if strict and missing_keys:
            raise KeyError(f"Missing keys in state_dict: {missing_keys}")
        if strict and unexpected_keys:
            raise KeyError(f"Unexpected keys in state_dict: {unexpected_keys}")

        for name, param in self.named_parameters():
            if name in state_dict:
                _load_tensor_value(param, state_dict[name], name)

        for name, buf in self.named_buffers():
            if name in state_dict:
                _load_tensor_value(buf, state_dict[name], name)

        return {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}

    def __repr__(self):
        layer_reprs = ",".join(repr(layer) for layer in self.layers)
        return f"Sequential({layer_reprs})"


def _load_tensor_value(tensor, value, key):
    """Loads array-like data into an existing tensor after shape validation."""
    source = value.data if isinstance(value, Tensor) else value
    source = np.array(source, dtype=tensor.data.dtype, copy=True)
    if source.shape != tensor.data.shape:
        raise ValueError(
            f"Shape mismatch for '{key}': expected {tensor.data.shape}, got {source.shape}"
        )
    tensor.data = source
    tensor.shape = tensor.data.shape
    tensor.size_val = tensor.data.size
    tensor.dtype = tensor.data.dtype


        



