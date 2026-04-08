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

    All layers should inherit from this class and implement
      -forward(x): computes layer output
      -parameters(): returns list of trainable parameters

    The __call__ method is provided to make layers callable.
    """

    def forward(self,x):
        """
        Forward pass through layer.

        Args:
          x: Inpuy tensor

        Returns:
          Output tensor after transformation
        """
        raise NotImplementedError("Subclasses must implement forward()")

    def __call__(self,x,*args,**kwargs):
        """Allows layer to be called like a function."""
        return self.forward(x,*args,**kwargs)

    def parameters(self):
        """
        Return list of trainable parameters.

        Returns:

           List of Tensor objects (weights and biases)
        """

        return [] #base class has no parameters

    def __repr__(self):
        """String representation of the layer."""
        return f"{self.__class__.__name__}()"

   
class Linear(Layer):
    """
    Linear (fully connected) layer: y = xW +b

    This is the fundemental building block of neural networks.
    Applies a linear transformation to incoming data
    """   

    def __init__(self,in_features,out_features,bias=True):
        """
        Intializing linear layer with proper weight intialization
        """
        self.in_features = in_features
        self.out_features = out_features

        #Xavier/Glorot intialization for stable gradients
        scale = np.sqrt(XAVIER_SCALE_FACTOR/ in_features)
        weight_data = np.random.randn(in_features,out_features)* scale
        self.weight = Tensor(weight_data, requires_grad=True)

        #initializze bias to zeros of None
        if bias:
            bias_data = np.zeros(out_features)
            self.bias = Tensor(bias_data, requires_grad=True)
        else:
            self.bias = None


    def forward(self,x):
        """
        Forward pass through linear layer.
        """

        #linear transformation y=Wx
        output = x.matmul(self.weight)

        ##add bias if present
        if self.bias is not None:
            output = output + self.bias 

        return output

    def parameters(self):
        """Return list of trainable parameters."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params 

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
        if not DROPOUT_MIN_PROB <= p <= DROPOUT_MAX_PROB:
            raise ValueError(f"Dropout probability must be between {DROPOUT_MIN_PROB} and {DROPOUT_MAX_PROB}, got {p}")

        self.p = p

    def forward(self,x,training=True):
        """
        Forward pass through dropout layer.
        """
        if not training or self.p == DROPOUT_MIN_PROB:
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

    def __call__(self,x,training=True):
        """Allows the layer to be called like a function"""
        return self.forward(x,training)

    def parameters(self):
        """Dropout has no parameters. """
        return []

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

    def forward(self,x):
        """Forward pass through all layers sequentially."""
        for layer in self.layers:
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

    def __repr__(self):
        layer_reprs = ",".join(repr(layer) for layer in self.layers)
        return f"Sequential({layer_reprs})"


        



