import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from typing import Optional, List,Tuple
from Tensor.tensor import Tensor 

#Constants for small numerical differentiation
EPILSON = 1e-7

_grad_enabled = True

class no_grad:
    """
    Context manager and decorator to disable gradient tracking.
    """
    def __enter__(self):
        global _grad_enabled
        self.prev = _grad_enabled
        _grad_enabled = False

    def __exit__(self, *args):
        global _grad_enabled
        _grad_enabled = self.prev

    def __call__(self, func):
        import functools
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

class Function:
    """
    Base class for differentiable operations.

    Every operation that needs gradients i.e add, multiply,matmul etc
    will inherit from this class and implement the apply() method

    args:
        saved_tensors: stores inputs needed for backward pass
        apply(): this method computes gradients using chain rule
    """

    def __init__(self,*tensors):
        """
        Intializes function with input tensors

        Args:
          *tensors: input tensors that will be saved for backward pass

        """
        self.saved_tensors = tensors

    def apply(self,grad_outputs):
        """
        Computes gradients for inputs.

        Args:
          grad_output: Gradient flowing backward from the output

        Return:

           Tuple of gradients for each input tensor

        **it must be implements by subclasses**
        """
        raise NotImplementedError("Each function must implemented apply() method")

class AddBackward(Function):
    """
    Gradient computation for tensor addition.

    **Mathematical Rule:** If z = a + b, then ∂z/∂a = 1 and ∂z/∂b = 1
    Addition distributes gradients equally to both inputs
    The gradient flowing backward is passed unchanged to each input.

    Broadcasting Handling: When input shapes differ due to broadcasting,
    we sum gradients appropriately to match original tensor shapes.

    """

    def apply(self,grad_output):
        """
        Compute gradients for addition.

        Args:
           grad_output: Gradients flowing backward from output

        Returns:
            Tuple of (grad_a,grad_b) for the two inputs

        Mathematical Foundations:
         - ∂(a+b)/∂a = 1 → grad_a = grad_output
        - ∂(a+b)/∂b = 1 → grad_b = grad_output
        """
        #extracting input tensors from self.saved_tensors
        a,b = self.saved_tensors
        #initializing grad_a,grad_b to None
        grad_a, grad_b = None, None

        #computing gradients for first input
        if isinstance(a,Tensor) and a.requires_grad:
            grad_a = grad_output

        #computing gradients for second input
        if isinstance(b,Tensor) and b.requires_grad:
            grad_b = grad_output

        return grad_a,grad_b

class MulBackward(Function):
    """
    Gradient computation for multiplication.

    **Mathematical Rule:** If z = a*b, then ∂z/∂a = b and ∂z/∂b = a
    Each input's gradient equals the gradient output
    multiplied by yhe OTHER input's value i.e product rule

    """

    def apply(self,grad_output):
        """
        Computes gradients for multiplication.

        Args:
           grad_output:gradients flowing backward from output

        Returns:
           Tuple of (grad_a,grad_b) for the two inputs

        **Mathematical Foundation:**
        - ∂(a*b)/∂a = b → grad_a = grad_output * b
        - ∂(a*b)/∂b = a → grad_b = grad_output * a
        """

        #extracting input tensors from self.saved_tensors
        a,b = self.saved_tensors
        #initializing grad_a and grad_b to None
        grad_a, grad_b = None, None

        #computing gradients for first input
        if isinstance(a,Tensor) and a.requires_grad:
            if isinstance(b,Tensor):
                grad_a = grad_output * b.data 
            else:
                grad_a = grad_output * b 

        #computing gradient for second input
        if isinstance(b,Tensor) and b.requires_grad:
            grad_b = grad_output * a.data 

        return grad_a,grad_b 

class SubBackward(Function):
    """
    Gradient computation for tensor subtraction

    **Mathematical Rule:** If z = a- b, the ∂z/∂a = 1 and ∂z/∂b = -1

    """

    def apply(self,grad_output):
        """
        Computes gradients for subtraction.

        Returns:
           Tuple of (grad_a,grad_b) where grad_b is negated
        """
        #extracting a,b from saved tensors
        a,b = self.saved_tensors
        grad_a, grad_b = None, None

        #computing gradient of first input
        if isinstance(a,Tensor) and a.requires_grad:
            grad_a = grad_output # ∂(a-b)/∂a = 1

        #computing gradient of second input
        if isinstance(b,Tensor) and b.requires_grad:
            grad_b = -grad_output # ∂(a-b)/∂b = -1

        return grad_a,grad_b
class DivBackward(Function):
    """
    Gradient computation for tensor division.

    Mathematical Rule: if z = a/b, then:
    - ∂z/∂a = 1/b
    - ∂z/∂b = -a/b²
    """
    def apply(self,grad_output):
        """
        Computes gradients for division using quotient rule

        Returns:
            Tuple of (grad_a,grad_b)
        """
        ##extracting a,b from self.saved tensors
        a,b = self.saved_tensors
        grad_a, grad_b = None, None

        #computing gradient of a
        if isinstance(a,Tensor) and a.requires_grad:
            # ∂(a/b)/∂a = 1/b
            if isinstance(b,Tensor):
                grad_a = grad_output / b.data
            else:
                grad_a = grad_output / b 

        if isinstance(b,Tensor) and b.requires_grad:
             # ∂(a/b)/∂b = -a/b²
             grad_b = -grad_output * a.data / (b.data ** 2)

        return grad_a,grad_b

class MatMulBackward(Function):
    """
    Gradient computation for matrix multiplication.

    **Mathematical Rule:** If Z = A @ , then:
    - ∂Z/∂A = grad_Z @ B.T
    - ∂Z/∂B = A.T @ grad_Z

    Matrix Multiplication gradients involve transposing
    one input and multiplying with the gradient output.
    """

    def apply(self,grad_output):
        """
        Computes the gradients for matrix multiplication.

        Args:
           grad_output: Gradient flowing backward from output

        Returns:
            Tuple of (grad_a,grad_b) for the two matrix inputs

        **Mathematical Foundation:**
         - ∂(A@B)/∂A = grad_output @ B.T
        - ∂(A@B)/∂B = A.T @ grad_output

        Batched operation: For 3D+ tensors, we transpose only the last two dimensions using np.swapaxes,
        preserving the batch dimensions.

        """
        ##extracting a,b from self.saved_tensors
        a,b = self.saved_tensors
        
        grad_a, grad_b = None, None

        ##Gradient for first input
        if isinstance(a,Tensor) and a.requires_grad:
            #for batched tensors, transpose only last two dims
            if b.data.ndim >= 2:
                b_T = np.swapaxes(b.data,-2,-1)
            else:
                b_T = b.data.T
            grad_a = np.matmul(grad_output, b_T)

        #Gradient for second input
        if isinstance(b,Tensor) and b.requires_grad:
            # for batched tensors, transpose only last two dims
            if a.data.ndim >= 2:
                a_T = np.swapaxes(a.data,-2,-1)
            else:
                a_T = a.data.T 
            grad_b = np.matmul(a_T,grad_output)

        return grad_a,grad_b

class TransposeBackward(Function):
    """
    Gradient computation for transpose operation

    **Mathematical Rule:** If Y = X.T, then:
    - ∂Y/∂X = grad_Y.T

    The gradient of transpose is just transpose the gradient!
    This is because transpose is a linear operation that just rearranges elements.

    """

    def __init__(self,tensor,dim0,dim1):
        """

        Args:
           tensor:Input tensor
           dim0: First dimension to swap 
           dim2; Second dimension to swap

        """
        super().__init__(tensor)
        self.dim0 = dim0
        self.dim1 = dim1 

    def apply(self,grad_output):
        """
        Computes gradient for transpose.

        Args:
           grad_output: gradients flowing backward from output

        Returns:
             Tuple with single gradient for input tensor

        """
        ##extracting x from self.saved_tensors
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x,Tensor) and x.requires_grad:
            # Transpose gradient using the same dims
            if self.dim0 is None and self.dim1 is None:
                #transposing last two dimensions
                if grad_output.ndim < 2:
                    grad_x = grad_output.copy()
                else:
                    axes = list(range(grad_output.ndim))
                    axes[-2],axes[-1] = axes[-1],axes[-2]
                    grad_x = np.transpose(grad_output,axes)
            else:
                #specific dimensions: swap them back
                axes = list(range(grad_output.ndim))
                axes[self.dim0],axes[self.dim1] = axes[self.dim1],axes[self.dim0]
                grad_x = np.transpose(grad_output,axes)

            return (grad_x,)

class PermuteBackward(Function):
    """
    Gradient computation for arbitrary axis permutation i.e general transpose

    **Mathematical Rule:**if Y = X.permute(axes): then:
     - ∂Y/∂X = grad_Y.permute(inverse_axes)

    If axes = (0,2,1,3), the inverse is (0,2,1,3) i.e self inverse.
    More generally, if axes = (2,0,1) the inverse is (1,2,0)

    To reverse a permutation, we need to know where each axis went
    If axis i went to position axes[i], then in the inverse, postion axes[i] should go to i
    """

    def __init__(self,tensor,axes):
        """
        Args:
          tensor:Input tensor
          axes: Tuple of axis indices defining the permutation

        """

        super().__init__(tensor)
        self.axes = axes
        #computing inverse permuation i.e if axes[i] = j, then inverse_axes[j] =1
        self.inverse_axes = tuple(np.argsort(axes))

    def apply(self,grad_output):
        """
        Compute gradient for permuatation

        The gradient is permuted back using the inverse permutation

        **Mathematical Foundations:**
         - ∂(X.permute(axes))/∂X = grad_output.permute(inverse_axes)

        """

        ##extracting a,b from self.saved_tensors
        x, = self.saved_tensors
        grad_x = None

        if isinstance(x,Tensor)and x.requires_grad:
            #permute the gradient back to original axis order
            grad_x = np.transpose(grad_output,self.inverse_axes)

        return (grad_x,)

class EmbeddingBackward(Function):
    """
    Gradient computation for embedding lookup operation

    **Mathematical Rule:** If Y = Embedding[indices], then:
    - ∂Loss/∂Embedding[i] = sum of all gradients where index==i

    Embedding lookup is a gather operation. The backward is a scatter operation that accumulates
    gradients to the embedding weights.

    """

    def __init__(self,weight,indices):
        """
        Args:
           weight: Embedding weight matrix
           indices: Indices used for lookup
        """
        super().__init__(weight)
        self.indices = indices 

    def apply(self,grad_output):
        """
        Computes gradient for embedding lookup

        Args:
            grad_output: gradient flowing backward from output

        Returns:
            Tuple with single gradient for weight tensor

        **Mathematical Foundation**
         - ∂(Embedding[indices])/∂Embedding = scatter gradients to selected rows
        - Multiple indices can point to same embedding → gradients accumulate
        """

        weight, = self.saved_tensors
        grad_weight = None

        if isinstance(weight,Tensor) and weight.requires_grad:
            #intializae gradients with zeros
            grad_weight = np.zeros_like(weight.data)

            #scatter gradients back to embedding weights
            #np.add.at accumalatesgradients for repeated indices
            indices_flat = self.indices.data.astype(int).flatten()
            grad_output_reshaped = grad_output.reshape(-1,grad_output.shape[-1])

            np.add.at(grad_weight,indices_flat,grad_output_reshaped)

        return (grad_weight,)

class SliceBackward(Function):
    """
    Gradient computation for tensor slicing/indexing operations.

     **Mathematical Rule:** If Y = X[key], then:
    - ∂Loss/∂X[key] = grad_output
    - ∂Loss/∂X[other positions] = 0

    Slicing is a masking operation. The backward places
    gradients back into the original tensor positions, with zeros 
    everywhere else.
    """

    def __init__(self,tensor,key):
        """
        Args:
            tensor: Original tensor being sliced
            key: Slicing key (index,slice,tuple of slices,etc)
        """
        super().__init__(tensor)
        self.key = key
        self.original_shape = tensor.shape

    def apply(self,grad_output):
        """
        Computes gradient for slicing operation

        Args:
           grad_output: gradient flowing backward from sliced output

        Returns: 
           Tuple with single gradient for input tensors 
        
        *Mathematical Foundation:**
        -Slicing extracts a subset of elements
        -Backward scatters gradients back to original positions
        -Unsliced positions receive zero gradient

        """
        #extracting tensor from self.saved_tensors
        tensor, = self.saved_tensors
        grad_input = None

        if isinstance(tensor,Tensor) and tensor.requires_grad:
            #create an array with same shape as original tensor
            grad_input = np.zeros(self.original_shape,dtype=np.float32)

            #place the gradient back into the sliced positions
            # this is the inverse of the forward slicing operation
            grad_input[self.key] = grad_output

        return (grad_input,)


class ReshapeBackward(Function):
    """
    Gradient computation for reshape operation.

    **Mathematical Rule:** If Y = X.reshape(new_shape) then:
      - ∂Y/∂X = grad_Y.reshape(X.shape)

    Reshape rearranges the same elements.
    The gradient is simply reshaped back to the original shape

    """
    def __init__(self,tensor,original_shape):
        """
        Args:
           tensor: Input tensor
           original_shape: shape before reshape
        """
        super().__init__(tensor)
        self.original_shape = original_shape

    def apply(self, grad_output):
        """
        Computes gradient for reshape

        Args:
           grad_output: gradients flowing backward from output

        Returns:
           Tuple with sigle gradinet for input tensor

        **Mathematical Foundations:**
        - ∂(X.reshape(...))/∂X = grad_output.reshape(X.shape)
        - Just reshape the gradient back!
        """
        #extracting input tensor x from self.saved_tensors
        x, = self.saved_tensors

        #set gradient of x to None
        grad_x = None

        if isinstance(x,Tensor) and x.requires_grad:
            #reshape gradient back to original shape
            grad_x = grad_output.reshape(self.original_shape)

        return (grad_x,)

class SumBackward(Function):
    """
    Gradient Computation for tensor sum

     **Mathematical Rule:** If z = sum(a), then ∂z/∂a[i] = 1 for all i

     Sum distributes the gradient equally to all input elements/
     The gradient is broadcast fro the reduced output back to input shape

    """  

    def apply(self,grad_output):
        """
        Computes gradients for sum operation

        Args:
           grad_output: gradient flowing backward from output

        Returns:
           Tuple containing gradient for the input tensor

        **Mathematical Foundation:**
          - ∂sum(a)/∂a[i] = 1 → grad_a = ones_like(a) * grad_output
        """
        tensor, = self.saved_tensors

        if isinstance(tensor,Tensor) and tensor.requires_grad:
            #gradient is 1 for all elements scaled by grad_output
            return np.ones_like(tensor.data) * grad_output,
        return None,

class MeanBackward(Function):
    """
    Gradient computation for tensor mean.

    Mean is sum divided by the number of reduced elements, so the backward
    broadcasts the upstream gradient and scales it by 1 / element_count.
    """

    def __init__(self, tensor, axis=None, keepdims=False):
        super().__init__(tensor)
        self.axis = axis
        self.keepdims = keepdims

    def apply(self, grad_output):
        tensor, = self.saved_tensors

        if not (isinstance(tensor, Tensor) and tensor.requires_grad):
            return (None,)

        grad = np.asarray(grad_output, dtype=np.float32)
        input_shape = tensor.data.shape

        if self.axis is None:
            count = tensor.data.size
            expanded = np.broadcast_to(grad, input_shape)
        else:
            axes = self.axis
            if not isinstance(axes, tuple):
                axes = (axes,)
            axes = tuple(ax if ax >= 0 else ax + len(input_shape) for ax in axes)

            count = 1
            for ax in axes:
                count *= input_shape[ax]

            if not self.keepdims:
                for ax in sorted(axes):
                    grad = np.expand_dims(grad, axis=ax)

            expanded = np.broadcast_to(grad, input_shape)

        return (expanded / count,)

class ReLUBackward(Function):
    """
    Gradient computation for ReLU activation.

    ReLU: f(x) = max(0,x)
    Derivative: f'(x) = 1 if x > 0 else 0
    """      

    def __init__(self,input_tensor):
        """
        Intialize with input tensor
        """
        super().__init__(input_tensor)

    def apply(self,grad_output):
        """
        Computes gradient for ReLU.
        """

        #extracting input tensor from self.saved_tensors
        tensor,= self.saved_tensors

        if isinstance(tensor,Tensor) and tensor.requires_grad:
            #relu gradient = 1 if x> 0 else 0
            relu_grad = (tensor.data > 0).astype(np.float32)
            return grad_output * relu_grad,
        return None,

class SigmoidBackward(Function):
    """
    Gradient computation for sigmoid activation.

    Sigmoid: σ(x) = 1/(1 + exp(-x))
    Derivative: σ'(x) = σ(x) * (1 - σ(x))

    """   
    def __init__(self,input_tensor,output_tensor):
        """
        Initialize with both input and output.

        Args:
           input_tensor: original input to sigmoid
           output_tensor: output of sigmoid (saves recomputation)

        """
        super().__init__(input_tensor)
        self.output_data = output_tensor.data

    def apply(self,grad_output):
        """
        Compute gradient for sigmoid
        """
        #exxtract tensors from self.saved_tensors
        tensor, = self.saved_tensors

        if isinstance(tensor,Tensor) and tensor.requires_grad:
             # σ'(x) = σ(x) * (1 - σ(x))
             sigmoid_grad = self.output_data * (1-self.output_data)
             return grad_output * sigmoid_grad,
        return None,

class SoftmaxBackward(Function):
    """
    Gradient computation for softmax activation

    Softmax: softmax(x)[i] = exp(x[i]) / sum(exp(x))
    derivative: ∂softmax/∂x[i] = softmax[i] * (δ[i,j] - softmax[j])

    for gradient computation:
        grad_x[i] = softmax[i] * (grad_y[i] - sum(grad_y * softmax))

    the gradient depends on all elements of softmax due to the normalization, 
    not just the element being differentiated.
    """

    def __init__(self,input_tensor,output_tensor,dim=-1):
        """
        Intialze with input,output and dimension

        Args:
            input_tensor: original input to softmax
            output_tensor: output of softmax
            dim = dimension along which softmax was applied.
        """

        super().__init__(input_tensor)
        self.output_data = output_tensor.data
        self.dim = dim

    def apply(self,grad_output):
        """
        Compute gradient for softmax.

        Mathematical formula:
        ∂L/∂x[i] = softmax[i] * (∂L/∂y[i] - sum_j(∂L/∂y[j] * softmax[j]))

        This can be vectorized as:
        grad_x = softmax * (grad_y - sum(grad_y * softmax, keepdims=True))
        """
        tensor, = self.saved_tensors

        if isinstance(tensor,Tensor) and tensor.requires_grad:
            #compute sum ( grad_output* softmax) along the softmac dimension
            sum_term = np.sum(grad_output*self.output_data,axis=self.dim,keepdims=True)

            #softmax gradient: softmax * (grad_output -sum_term)
            grad_x = self.output_data * (grad_output - sum_term)

            return (grad_x,)

        return (None,)

class GELUBackward(Function):
    """
    Gradient computation for GELU activation.

    GELU: f(x) = x * Φ(x) where Φ is the CDF of standard normal
    Approximation: gelu(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

    GELU is smoother than ReLU, providing non-zero gradients for 
    negative values, which helps training deep networks.

    """

    def __init__(self,input_tensor):
        """
        Initialize with input tensor
        """
        super().__init__(input_tensor)

    def apply(self,grad_output):
        """
        Compute gradient for GELU.

        Mathematical formula (using approximation):
        ∂gelu/∂x ≈ 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * (...)
        """

        tensor, = self.saved_tensors

        if isinstance(tensor,Tensor) and tensor.requires_grad:
            x = tensor.data
            #GELU derivative approximation
            # Using the tanh approximation: gelu(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            sqrt_2_over_pi = np.sqrt(2.0/np.pi)
            x_cubed = x**3
            tanh_arg = sqrt_2_over_pi * (x + 0.044715 * x_cubed)
            tanh_out = np.tanh(tanh_arg)
            sech_squared = 1 - tanh_out ** 2

            # Derivative: 0.5 * (1 + tanh(...)) + 0.5 * x * sech²(...) * d(tanh_arg)/dx
            d_tanh_arg = sqrt_2_over_pi * (1 + 0.134145* x**2)
            gelu_grad = 0.5 * (1 + tanh_out) + 0.5 * x* sech_squared * d_tanh_arg

            return (grad_output * gelu_grad,)

        return (None,)


class MSEBackward(Function):
    """
    Gradient computation for mean squared error loss

    MSE: L: = mean((predictions- targets)**2)
    Derivative: ∂L/∂predictions = 2 * (predictions - targets) / N

    """

    def __init__(self,predictions,targets):
        """Initialize with predctions and targets."""
        super().__init__(predictions)
        self.targets_data = targets.data 
        self.num_samples = np.size(targets.data)

    def apply(self,grad_output):
        """
        Compute gradient for MSE loss.

        """
        ##extracting tensors from self.saved_tensors
        predictions, = self.saved_tensors

        if isinstance(predictions,Tensor)and predictions.requires_grad:
            # gradient: 2 * (predictions - targets) / N
            grad = 2.0 * (predictions.data - self.targets_data) / self.num_samples

            return grad * grad_output,
        return None,

class BCEBackward(Function):
    """
    Gradient computation for Binary Cross-Entropy Loss.

    BCE: L = - [y*log(p) + (1-y)*log(1-p)]
    derivative: ∂L/∂p = (p - y) / (p*(1-p)*N)

    """
    def __init__(self,predictions,targets):
        """Initialized with predictions and targets."""
        super().__init__(predictions)
        self.targets_data = targets.data
        self.num_samples = np.size(targets.data)

    def apply(self,grad_output):
        """
        Computes gradients for BCE loss
        """
        predictions, = self.saved_tensors

        if isinstance(predictions, Tensor) and predictions.requires_grad:
            eps = EPILSON
            p = np.clip(predictions.data,eps,1-eps)
            y = self.targets_data

            # gradient: (p-y) / (p*(1-p)* N)
            grad = (p-y) / (p*(1-p)* self.num_samples)

            return grad * grad_output,
        return None,

class CrossEntropyBackward(Function):
    """
    Gradient computation for Cross-Entropy Loss

    CrossEntropy: L = -mean(log_softmax(logits)[targets])

    The gradient with respect to logits:
    ∂L/∂logits = (softmax(logits) - one_hot(targets)) / N

    The gradient is simly the difference between predictions and targets
    It naturally scales with how wrong we are
    It is naturally stable when computed via softmax

    """

    def __init__(self,logits,targets):
        """Intialize with logits and target class indices"""
        super().__init__(logits)
        self.targets_data = targets.data.astype(int)
        self.batch_size = logits.data.shape[0]
        self.num_classes = logits.data.shape[1]

    def apply(self,grad_output):
        """
        Compute gradient for cross-entropy loss.
        """
        logits, = self.saved_tensors

        if isinstance(logits,Tensor) and logits.requires_grad:
            ##compute softmax probabilities
            ## using stable softmax: subtract max for numerical stability
            logits_data = logits.data 
            max_logits = np.max(logits_data,axis=1,keepdims=True)
            exp_logits = np.exp(logits_data - max_logits)
            softmax = exp_logits / np.sum(exp_logits,axis=1,keepdims=True)

            #create one-hot encoding of targets
            one_hot = np.zeros((self.batch_size,self.num_classes),dtype=np.float32)
            one_hot[np.arange(self.batch_size),self.targets_data] = 1.0

            #gradients = (softmax - one_hot) / batch_size
            grad = (softmax - one_hot) / self.batch_size

            return (grad * grad_output,)
        return (None,)

def enable_autograd(quiet=False):
    """
    Enalbles gradient tracking for all Tensor operations.

    This function enhances the existing Tensor class with autograd capabilities.
    By calling it we activate gradients globally.

    Args: 
         quiet (bool) if True, suppress status message. Default : False

    What it does:

        Replaces Tensor operations with gradient-tracking versions
        Adds  backward() methos for reverse-mode differentiation
        Enables computation graph building
        Maintains full backward compatibility

    After calling it:
    -Tensor operations will track computation graphs
    -backward() methos becomes available
    -Gradients will flow through operations
    -requires_grad=True enables tracking per tensor

    """
    if getattr(Tensor, '_autograd_enabled', False):
        return
    
    ##adding gradient infrastructure to Tensor
    #store original __init__ to extend it
    _original_init = Tensor.__init__

    def gradient_aware_init(self,data,requires_grad=False):
        """Extended Tensor init that supports gradient tracking."""
        _original_init(self,data)
        self.requires_grad = requires_grad
        self.grad = None
        self.hooks = []

    #replace __init_ with gradient-aware version
    Tensor.__init__ = gradient_aware_init

    def register_hook(self, hook):
        """
        Registers a backward hook.
        The hook will be called every time a gradient with respect to the tensor is computed.
        The hook should have the following signature:
            hook(grad) -> Tensor or None
        """
        self.hooks.append(hook)
        return len(self.hooks) - 1

    def _apply_hooks(self, grad):
        for hook in self.hooks:
            new_grad = hook(grad)
            if new_grad is not None:
                grad = new_grad
        return grad

    #store original operations
    #these are guaranteed to exist from the Tensor module
    _original_add = Tensor.__add__
    _original_sub = Tensor.__sub__
    _original_mul = Tensor.__mul__
    _original_div = Tensor.__truediv__
    _original_getitem =Tensor.__getitem__

    #these methods are also guaranteed from the Tensor module
    # use .matmul not .__matmul__ to avoid recursion (__matmul__ calls self.matmul)
    _original_matmul = Tensor.matmul
    _original_transpose = Tensor.transpose
    _original_reshape = Tensor.reshape

    #helper function to safely check requires_grad i.e hanfles tensores created
    ## before enable_autograd()
    def _get_requires_grad(tensor):
        """Safely get requires_grad, defaulting to Flase for pre-autograd tensors."""
        return getattr(tensor,'requires_grad',False) if isinstance(tensor,Tensor) else False

    def _ensure_grad_attrs(tensor):
        """Ensure tensor has gradient attributes i.e for tensors created before enable_autograd"""
        if isinstance(tensor,Tensor):
            if not hasattr(tensor,'requires_grad'):
                tensor.requires_grad = False
            if not hasattr(tensor,'grad'):
                tensor.grad= None

    #enhanced operations that track gradients
    def  tracked_add(self,other):
        """
        Addition with gradient tracking.

        Enhances the original __add__ method to build computation graphs
        when requires_grad=True for any input.
        """
        #ensuring that self has gradient attributes
        _ensure_grad_attrs(self)

        #converts scalar to Tensor if needed
        if not isinstance(other,Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        #call original operation
        result = _original_add(self,other)
        _ensure_grad_attrs(result)

        #tracking gradient if needed
        if _grad_enabled and (_get_requires_grad(self) or _get_requires_grad(other)):
            result.requires_grad = True
            result._grad_fn = AddBackward(self,other)

        return result

    def tracked_mul(self,other):
        """
        Multiplication with gradient tracking.

        Enhances the original __mul__ method to build computation
        graphs when requires_grad=True for any inpt.
        """

        _ensure_grad_attrs(self)

        # convert scalar to Tensor if needed for consistency
        if not isinstance(other,Tensor):
            other_tensor = Tensor(other)
        else:
            other_tensor = other
        _ensure_grad_attrs(other_tensor)

        #call original operation
        result = _original_mul(self,other)
        _ensure_grad_attrs(result)

        # track gradient if needed
        if _grad_enabled and (_get_requires_grad(self) or _get_requires_grad(other_tensor)):
            result.requires_grad = True
            result._grad_fn = MulBackward(self,other)

        return result

    def tracked_matmul(self,other):
        """
        Matrix multiplication with gradient tracking.

        Enhances the original matmul method to built computation graphs
        when requires_grad=True for any input
        """
        _ensure_grad_attrs(self)
        _ensure_grad_attrs(other)

        #call original matmul from Tensor module
        result = _original_matmul(self,other)
        _ensure_grad_attrs(result)

        #track gradients if needed
        if _grad_enabled and (_get_requires_grad(self) or _get_requires_grad(other)):
            result.requires_grad = True 
            result._grad_fn = MatMulBackward(self,other)

        return result

    def tracked_transpose(self,dim0=None,dim1=None):
        """
        Tranpose with gradient tracking.

        Enhances the original transpose method to build computation graphs
        when requires_grad=True for the input

        """
        _ensure_grad_attrs(self)

        #call original transpose from Tensor
        result = _original_transpose(self,dim0,dim1)
        _ensure_grad_attrs(result)

        #track gradient if needed
        if _grad_enabled and _get_requires_grad(self):
            result.requires_grad =True
            result._grad_fn = TransposeBackward(self,dim0,dim1)

        return result 

    def tracked_reshape(self, *shape):
        """
        Reshape with gradient tracking

        Enhances the original reshape method to build computation graphs
        when requires_grad=True for the input


        """
        _ensure_grad_attrs(self)
        original_shape = self.shape 

        #call original reshape from Tensor module
        result = _original_reshape(self,*shape)
        _ensure_grad_attrs(result)

        #track gradient if needed
        if _grad_enabled and _get_requires_grad(self):
            result.requires_grad = True 
            result._grad_fn = ReshapeBackward(self,original_shape)

        return result

    def tracked_sub(self,other):
        """
        Subtraction with gradient racking

        Enhances the original __sub__ method to build computation gra[hs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)

        #convert scalr to Tensor if needed
        if not isinstance(other, Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        #call original operation
        result =_original_sub(self,other)
        _ensure_grad_attrs(result)

        #track gradient if needed
        if _grad_enabled and (_get_requires_grad(self) or _get_requires_grad(other)):
            result.requires_grad = True 
            result._grad_fn = SubBackward(self,other)

        return result


    def tracked_div(self,other):
        """
        Division with gradient tracking.

        Enhances the original __truediv__ method to build computation graphs
        when requires_grad=True for any input.
        """
        _ensure_grad_attrs(self)

        #convert scalar to tensor if needed
        if not isinstance(other,Tensor):
            other = Tensor(other)
        _ensure_grad_attrs(other)

        #call original operation
        result = _original_div(self,other)
        _ensure_grad_attrs(result)

        #track gradient if needed
        if _grad_enabled and (_get_requires_grad(self) or _get_requires_grad(other)):
            result.requires_grad = True
            result._grad_fn = DivBackward(self,other)
        
        return result
    
    def tracked_getitem(self,key):
        """
        indexing/slicing with gradient tracking

        Enhances the original __getitem__ method to build computation graphs when
        requires_grad=True for the input
        """

        _ensure_grad_attrs(self)

        #calling original __getitem__ from Tensor
        result = _original_getitem(self,key)
        _ensure_grad_attrs(result)

        #tracking gradients if needed
        if _grad_enabled and _get_requires_grad(self):
            result.requires_grad = True 
            result._grad_fn = SliceBackward(self,key)

        return result 

    def sum_op(self,axis=None,keepdims=False):
        """
        Sum operation with gradient tracking

        Creates a new sum methos that builds computation graphs
        when requires_grad=True.
        """
        _ensure_grad_attrs(self)

        result_data = np.sum(self.data,axis=axis,keepdims=keepdims)
        result = Tensor(result_data)

        if _grad_enabled and _get_requires_grad(self):
            result.requires_grad=True
            result._grad_fn = SumBackward(self)

        return result 

    def mean_op(self,axis=None,keepdims=False):
        """
        Mean operation with gradient tracking.

        Builds a reduction node that routes gradients back to the input and
        scales them by the number of reduced elements.
        """
        _ensure_grad_attrs(self)

        result_data = np.mean(self.data,axis=axis,keepdims=keepdims)
        result = Tensor(result_data)
        _ensure_grad_attrs(result)

        if _grad_enabled and _get_requires_grad(self):
            result.requires_grad = True
            result._grad_fn = MeanBackward(self, axis=axis, keepdims=keepdims)

        return result

    def backward(self,gradient=None):
        """
        Computes gradients via backpropagation.

        This is the key method that makes training possible
        It implements reverse-mode automatic differentiation.

        Algorithm:
             1.Initialize gradient if not provided
             2. Accumulate gradient in self.grad
             3. If this tensor has a _grad_fn, call it to propagate gradients
             4.Recursively call backward() on parent tensors

        """
        #ensuring gradient attributes exists
        _ensure_grad_attrs(self)

        #only compute gradient if required
        if not _get_requires_grad(self):
            return 

        #initializing gradients incase they are not provided i.e for scalar outputs
        if gradient is None:
            if self.data.size == 1:
                gradient = np.ones_like(self.data)
            else:
                raise ValueError(
                    f"backward() called on non-scalr tensor without gradient argument.\n"
                    f"   Tensor shape: {self.shape}\n"
                    f"   Issue: For non-scalar output, you must provide the gradient from the next layer.\n"
                    f"  Fix: Call backward(gradient) with the gradient tensor from the loss function"
                )

        # ensure gradient is numpy (e.g. from Tensor)
        if hasattr(gradient, 'data'):
            gradient = np.asarray(gradient.data, dtype=np.float32)

        # initialize or accumulate gradient
        if self.grad is None:
            self.grad = np.zeros_like(self.data)

        #handling broadcasting: sum gradient to match self.data.shape
        if gradient.shape != self.grad.shape:
            while gradient.ndim > self.grad.ndim:
                gradient = gradient.sum(axis=0)
            for i in range(gradient.ndim):
                if self.grad.shape[i] == 1 and gradient.shape[i] != 1:
                    gradient = gradient.sum(axis=i,keepdims=True)

        self.grad += gradient

        #propagating gradient through computation graph
        grad_fn = getattr(self,'_grad_fn',None)
        if grad_fn is not None:
            grads = grad_fn.apply(gradient)

            for tensor,grad in zip(grad_fn.saved_tensors,grads):
                if isinstance(tensor,Tensor) and tensor.requires_grad and grad is not None:
                    tensor.backward(grad)


    def zero_grad(self):
        """
        Resets gradient to zero

        This function is called before each bakcward pass to prevent gradient accumulation
        from previous iterations.


        """
        self.grad = None

    #install enhanced operations (must be in enable_autograd body, not inside zero_grad)
    Tensor.__add__ = tracked_add 
    Tensor.__sub__ = tracked_sub 
    Tensor.__mul__ = tracked_mul 
    Tensor.__truediv__ = tracked_div 
    Tensor.__getitem__ = tracked_getitem 
    Tensor.matmul = tracked_matmul 
    Tensor.transpose = tracked_transpose 
    Tensor.reshape = tracked_reshape 
    Tensor.sum = sum_op 
    Tensor.mean = mean_op
    Tensor.backward = backward 
    Tensor.zero_grad = zero_grad 
    Tensor.no_grad = no_grad

    #patch activations and losses to track gradients
    try:
        from activations.activations import Sigmoid,ReLU,Softmax,GELU
        from losses.losses import BinaryCrossEntropyLoss,MSELoss,CrossEntropyLoss

        #store origial methods
        _original_sigmoid_forward =Sigmoid.forward
        _original_relU_forward = ReLU.forward 
        _original_softmax_forward = Softmax.forward
        _original_gelu_forward = GELU.forward
        _original_bce_forward = BinaryCrossEntropyLoss.forward
        __original_mse_forward = MSELoss.forward
        _original_ce_forward = CrossEntropyLoss.forward

        def tracked_sigmoid_forward(self,x):
            """Sigmoid with gradient tracking."""
            result_data = 1.0 /(1.0 + np.exp(-x.data))
            result = Tensor(result_data)

            if x.requires_grad:
                result.requires_grad = True 
                result._grad_fn = SigmoidBackward(x,result)

            return result 

        def tracked_relu_forward(self,x):
            """ReLU with gradient tracking."""
            result_data = np.maximum(0,x.data)
            result = Tensor(result_data)

            if getattr(x, "requires_grad", False):
                result.requires_grad = True
                result._grad_fn = ReLUBackward(x)

            return result
        def tracked_softmax_forward(self,x,dim=-1):
            """Softmax with gradient tracking."""
            #call original forward to result using Tensor operations
            result = _original_softmax_forward(self,x,dim=dim)

            #attach the correct gradient function
            if x.requires_grad:
                result.requires_grad = True 
                result._grad_fn = SoftmaxBackward(x,result,dim)

            return result 

        def tracked_gelu_forward(self,x):
            """GELU with gradient tracking."""
            # call the original forward to get result 
            result = _original_gelu_forward(self,x)

            # attach the correct gradient function 
            if getattr(x, "requires_grad", False):
                result.requires_grad = True 
                result._grad_fn = GELUBackward(x)

            return result

        def tracked_bce_forward(self,predictions,targets):
            """Binary cross-entropy with gradient tracking."""
            #compute BCE los
            eps = EPILSON 
            clamped_preds = np.clip(predictions.data,eps,1-eps)
            log_preds =np.log(clamped_preds)
            log_one_minus_preds = np.log(1-clamped_preds)
            bce_per_sample = -(targets.data * log_preds + (1-targets.data)*log_one_minus_preds)
            bce_loss = np.mean(bce_per_sample)

            result = Tensor(bce_loss)

            if predictions.requires_grad:
                result.requires_grad = True 
                result._grad_fn = BCEBackward(predictions,targets)

            return result 

        def tracked_mse_forward(self,predictions,targets):
            """MSE loss with gradient tracking."""
            #compute MSE loss
            diff = predictions.data - targets.data 
            squared_diff = diff**2 
            mse = np.mean(squared_diff)

            result = Tensor(mse)

            if predictions.requires_grad:
                result.requires_grad = True 
                result._grad_fn = MSEBackward(predictions,targets)

            return result 

        def tracked_ce_forward(self,logits,targets):
            """Cross-entropy loss with gradient tracking."""
            from losses.losses import log_softmax

            #compute log-softmax for numerical stability
            log_probs = log_softmax(logits,dim=-1)

            #select log-probabilites for correct classes
            batch_size = logits.shape[0]
            target_indices = targets.data.astype(int)
            selected_log_probs = log_probs.data[np.arange(batch_size),target_indices]

            #return negative mean
            ce_loss = -np.mean(selected_log_probs)

            result = Tensor(ce_loss)

            if logits.requires_grad:
                result.requires_grad = True
                result._grad_fn = CrossEntropyBackward(logits,targets)

            return result


        #install patched methods
        Sigmoid.forward = tracked_sigmoid_forward
        ReLU.forward = tracked_relu_forward
        Softmax.forward = tracked_softmax_forward
        GELU.forward = tracked_gelu_forward
        BinaryCrossEntropyLoss.forward = tracked_bce_forward
        MSELoss.forward = tracked_mse_forward 
        CrossEntropyLoss.forward = tracked_ce_forward

    except ImportError:
        #activations/losses not yet available
        pass

    #Mark as enabled
    Tensor._autograd_enabled = True 

    if not quiet:
        print("Autograd enabled!")

enable_autograd(quiet=True)       





