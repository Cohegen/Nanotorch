import sys
import os
# Add parent directory to path to allow importing Tensor module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tensor import Tensor
import numpy as np
from typing import Optional

#constant for nuemrical comparisons
TOLERANCE = 1e-10

#export only activation classes
__all__  = ['Sigmoid','ReLU','LeakyReLU','SiLU','Mish','PReLU','SwiGLU','ELU','Tanh','GELU','Softmax']


class Sigmoid:
    """
    sigmoid activation function
    it maps any real number to the range (0,1) making it perfect for probabilities and binary decisions

    """

    def parameters(self):
        """Returns empty list"""
        return []

    def forward(self,x:Tensor) -> Tensor:
        """Applying sigmoid activation element-wise"""
        
        ##clipping extreme values to prevent overflow
        ##clipping at +/- 500 ensures exp() stays within float64 range
        z = np.clip(x.data,-500,500)

        result_data = np.zeros_like(z)

        #postive values (including zero)
        pos_mask = z >= 0
        result_data[pos_mask] = 1.0 / (1.0 + np.exp(-z[pos_mask]))

        #negative values
        neg_mask = z < 0
        exp_z = np.exp(z[neg_mask])
        result_data[neg_mask] = exp_z / (1.0 + exp_z)

        return Tensor(result_data)

    def __call__(self,x:Tensor) -> Tensor:
        """Allows the activation to be called like a function"""
        return self.forward(x)

    def backward(self,grad: Tensor) -> Tensor:
        pass

    
##Rectified Linear Unit activation function
class ReLU:
    """Class implementing the ReLU activation function f(x) = max(0,x)"""

    def parameters(self):
        """Returns an empty list since activations have no learnable parameters"""
        return []

    def forward(self,x: Tensor) ->Tensor:
        """Applying ReLU activation element-wise"""

        result = np.maximum(0,x.data)
        return Tensor(result)


    def __call__(self,x: Tensor) -> Tensor:
        """Allows the activation to be called like a function"""
        return self.forward(x)

    def backward(self,grad: Tensor) -> Tensor:
        """Computes gradient """
        pass


class LeakyReLU:
    """Leaky ReLU activation: f(x) = x if x > 0 else negative_slope * x."""

    def __init__(self, negative_slope: float = 0.01):
        self.negative_slope = negative_slope

    def parameters(self):
        return []

    def forward(self, x: Tensor) -> Tensor:
        result = np.where(x.data > 0, x.data, self.negative_slope * x.data)
        return Tensor(result)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class SiLU:
    """SiLU / Swish activation: f(x) = x * sigmoid(x)."""

    def parameters(self):
        return []

    def forward(self, x: Tensor) -> Tensor:
        z = np.clip(x.data, -500, 500)
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        return Tensor(x.data * sigmoid)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class Mish:
    """Mish activation: f(x) = x * tanh(softplus(x))."""

    def parameters(self):
        return []

    def forward(self, x: Tensor) -> Tensor:
        softplus = np.log1p(np.exp(-np.abs(x.data))) + np.maximum(x.data, 0)
        return Tensor(x.data * np.tanh(softplus))

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class PReLU:
    """Parametric ReLU with a learnable negative slope."""

    def __init__(self, init: float = 0.25):
        self.weight = Tensor([init], requires_grad=True)

    def parameters(self):
        return [self.weight]

    def forward(self, x: Tensor) -> Tensor:
        negative_slope = float(self.weight.data.reshape(-1)[0])
        result = np.where(x.data > 0, x.data, negative_slope * x.data)
        return Tensor(result)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)


class SwiGLU:
    """SwiGLU activation: split in half and apply a * SiLU(gate)."""

    def parameters(self):
        return []

    def forward(self, x: Tensor, dim: int = -1) -> Tensor:
        size = x.data.shape[dim]
        if size % 2 != 0:
            raise ValueError("SwiGLU requires an even-sized dimension for splitting")

        first_half, second_half = np.split(x.data, 2, axis=dim)
        gate = second_half * (1.0 / (1.0 + np.exp(-np.clip(second_half, -500, 500))))
        return Tensor(first_half * gate)

    def __call__(self, x: Tensor, dim: int = -1) -> Tensor:
        return self.forward(x, dim)


class ELU:
    """Exponential Linear Unit activation."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def parameters(self):
        return []

    def forward(self, x: Tensor) -> Tensor:
        result = np.where(x.data > 0, x.data, self.alpha * (np.exp(x.data) - 1.0))
        return Tensor(result)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

class Tanh:
    """Coded implementation of Tanh activation function 
    It maps any real number to a range (-1,1)
    zero-centred alternative to sigmoid
    """
    def parameters(self):
        """Return empty list (activation have no learnable parameters)"""
        return []

    def forward(self,x:Tensor) -> Tensor:
        result = np.tanh(x.data)
        return Tensor(result)


    def __call__(self,x:Tensor) -> Tensor:
        """Allows the activation to be called like a function"""
        return self.forward(x)

    def backward(self,grad: Tensor) -> Tensor:
        """Compute gradient"""
        pass

class GELU:
    """
    GELU activation: smooth approxiation to ReLU, used in modern transformers.

    """

    def parameter(self):
        """Returns empty list"""
        return []

    def forward(self,x: Tensor) -> Tensor:
        """Applying GELU activation element-wise"""
        sigmoid_part = 1/ (1.0+np.exp(-1.702*x.data))
        result = x.data * sigmoid_part
        return Tensor(result)

    def __call__(self, x:Tensor) -> Tensor:
        """Allows the activation to be called like a function"""
        return self.forward(x)

    def backward(self,grad: Tensor) -> Tensor:
        """Computes gradient"""
        pass   

class Softmax:
    """Softmax activation function
    Converts any vector in a probability distribution
    Sum of all outputs equals 1.0
    """

    def parameters(self):
        """Return empty list"""
        return []

    def forward(self,x:Tensor,dim: int =-1) -> Tensor:
        """Applies softmax activation function along a specified dimension. """

        x_max_data  = np.max(x.data,axis=dim,keepdims=True)
        x_max = Tensor(x_max_data)

        #subtrcting max for numerical stability
        x_shifted = x- x_max

        #computing exponentials
        exp_values = Tensor(np.exp(x_shifted.data))

        #sum along dimensions
        exp_sum_data = np.sum(exp_values.data,axis=dim,keepdims=True)
        exp_sum = Tensor(exp_sum_data)

        #normalize to get probabilities
        result = exp_values / exp_sum
        return result 

    def __call__(self, x: Tensor,dim: int=-1) -> Tensor:
        """Allow the activation to be called like a function"""
        return self.forward(x,dim)
        
