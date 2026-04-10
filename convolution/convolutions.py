from multiprocessing import Value
from ntpath import abspath
import os
import sys


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
import numpy as np

##enabling autograd for gradient tracking
from autograd.autograd import enable_autograd,Function
enable_autograd()

#constants for convolution defaults
DEFAULT_KERNEL_SIZE = 3 # default kernel size for convolutions
DEFAULT_STRIDE = 1 # defult stride for convolutions
DEFAULT_PADDING = 0 # default padding for convolutions

#constants for memory Allocation
BYTES_PER_FLOAT32 = 4#standard float32 size in bytes
KB_TO_BYTES =1024 #kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024 #megabytes to bytes conversion


def im2col(input_data, kernel_h, kernel_w, stride, padding):
    """
    Rearranges image blocks into columns for fast matrix multiplication.
    
    Args:
        input_data: 4D input array (batch, channels, height, width)
        kernel_h, kernel_w: size of the convolution kernel
        stride: stride of the convolution
        padding: zero-padding applied to the spatial dimensions
        
    Returns:
        Matrix of shape (channels * kernel_h * kernel_w, batch * out_h * out_w)
    """
    batch_size, channels, in_h, in_w = input_data.shape
    out_h = (in_h + 2 * padding - kernel_h) // stride + 1
    out_w = (in_w + 2 * padding - kernel_w) // stride + 1

    if padding > 0:
        img = np.pad(input_data, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
    else:
        img = input_data

    # Use strides to create overlapping patches without explicit copying
    # Resulting shape: (batch, channels, kernel_h, kernel_w, out_h, out_w)
    # This is an advanced NumPy trick for memory efficiency
    shape = (batch_size, channels, kernel_h, kernel_w, out_h, out_w)
    strides = (img.strides[0], img.strides[1], img.strides[2], img.strides[3], img.strides[2] * stride, img.strides[3] * stride)
    
    patches = np.lib.stride_tricks.as_strided(img, shape=shape, strides=strides)
    
    # Transpose and reshape into columns: (C * Kh * Kw, B * Oh * Ow)
    return patches.transpose(1, 2, 3, 0, 4, 5).reshape(channels * kernel_h * kernel_w, -1)


def col2im(cols, input_shape, kernel_h, kernel_w, stride, padding):
    """
    Reconstructs image from columns (inverse of im2col).
    Used for the backward pass to accumulate gradients.
    """
    batch_size, channels, in_h, in_w = input_shape
    out_h = (in_h + 2 * padding - kernel_h) // stride + 1
    out_w = (in_w + 2 * padding - kernel_w) // stride + 1

    # Reshape back to (channels, kernel_h, kernel_w, batch, out_h, out_w)
    cols_reshaped = cols.reshape(channels, kernel_h, kernel_w, batch_size, out_h, out_w)
    
    img_padded = np.zeros((batch_size, channels, in_h + 2 * padding, in_w + 2 * padding), dtype=cols.dtype)
    
    # Accumulate patches back into the image
    for kh in range(kernel_h):
        h_end = kh + stride * out_h
        for kw in range(kernel_w):
            w_end = kw + stride * out_w
            # Transpose batch and channel to match img_padded: (batch, channels, out_h, out_w)
            img_padded[:, :, kh:h_end:stride, kw:w_end:stride] += cols_reshaped[:, kh, kw, :, :, :].transpose(1, 0, 2, 3)

    if padding > 0:
        return img_padded[:, :, padding:-padding, padding:-padding]
    return img_padded


def validate_4d_input(x,layer_name):
    """"
    validates that the input tensor is 4D (batch,channels,height,width)

    Args:
       x:Input Tensor to validate
       layer_name:Name of the calling layer

    Raises:
        ValueError if input is not 4D 
    """
    if len(x.shape) == 4:
        return
    
    if len(x.shape) == 3:
        raise ValueError(
            f"{layer_name} expected 4D input (batch,channels,height,width), got 3D:{x.shape}\n"
            f"Missing batch dimension\n"
            f"{layer_name} processes batches of images, not single images\n"
            f"Add batch dim:x.reshape(1,{x.shape[0]},{x.shape[1]},{x.shape[2]})"
        )

    elif len(x.shape) == 2:
        raise ValueError(
            f"{layer_name} expected 4D input (batch,channels,height,width),got 2D: {x.shape}\n"
            f"Got a matrix, expected an image tensor\n"
            f"{layer_name} needs spatial dimensions (height,width) plus batch and channels\n"
            f"If this is flattened image,reshapeit:x.reshape(1,channels,height,width)"

        )
    else:
        raise ValueError(
            f"{layer_name} expected 4D input (batch,channels,height,width), got {len(x.shape)}D:{x.shape}\n"
            f"Wrong number of dimensions\n"
            f" {layer_name} expects: (batch_size,channels,height,width)\n"
            f" Reshape your input to 4D with the correct dimensions"
        )



class Conv2dBackward(Function):
    """
    Gradient computation for 2D convolution.

    Computes gradients for Conv2d backward pass:
    grad_input is gradient wrt input (for backprop to previous layer)
    grad_weight is gradient wrt filters (for weight updates)
    grad_bias is gradient wrt bias (for bias updates)
    """

    def __init__(self, x, weight, bias, stride, padding, kernel_size, method='im2col'):
        # Registering all tensors that neeed gradients with autograd
        if bias is not None:
            super().__init__(x, weight, bias)
        else:
            super().__init__(x, weight)
        self.x = x
        self.weight = weight
        self.bias = bias
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.method = method

    def apply(self, grad_output):
        """
        Compute gradients for convolution input and parameters

        Args:
           grad_output: Gradient flowing back from next layer
           Shape: (batch_size,out_channels,out_height,out_width)

        Returns:
        Tuple of (grad_input,grad_weight,grad_bias)

        """
        batch_size, out_channels, out_height, out_width = grad_output.shape
        _, in_channels, in_height, in_width = self.x.shape
        kernel_h, kernel_w = self.kernel_size

        if self.method == 'naive':
            # Applying padding to input if need (for gradient computation)
            if self.padding > 0:
                padded_input = np.pad(
                    self.x.data,
                    ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                    mode='constant',
                    constant_values=0
                )
            else:
                padded_input = self.x.data

            # initialize gradients
            grad_input_padded = np.zeros_like(padded_input)
            grad_weight = np.zeros_like(self.weight.data)
            grad_bias = None if self.bias is None else np.zeros_like(self.bias.data)

            # computing gradients explicit loops
            for b in range(batch_size):
                for out_ch in range(out_channels):
                    for out_h in range(out_height):
                        for out_w in range(out_width):
                            # Position in input
                            in_h_start = out_h * self.stride
                            in_w_start = out_w * self.stride

                            # gradient values flowing back to this position
                            grad_val = grad_output[b, out_ch, out_h, out_w]

                            # distribute gradient to weight and input
                            for k_h in range(kernel_h):
                                for k_w in range(kernel_w):
                                    for in_ch in range(in_channels):
                                        # input position
                                        in_h = in_h_start + k_h
                                        in_w = in_w_start + k_w

                                        # gradient wrt weight
                                        grad_weight[out_ch, in_ch, k_h, k_w] += (
                                            padded_input[b, in_ch, in_h, in_w] * grad_val
                                        )

                                        # gradient wrt input
                                        grad_input_padded[b, in_ch, in_h, in_w] += (
                                            self.weight.data[out_ch, in_ch, k_h, k_w] * grad_val
                                        )

            # compute gradient wrt bias
            if grad_bias is not None:
                for out_ch in range(out_channels):
                    grad_bias[out_ch] = grad_output[:, out_ch, :, :].sum()

            # remove padding from input gradient
            if self.padding > 0:
                grad_input = grad_input_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
            else:
                grad_input = grad_input_padded

        else:  # im2col method
            # 1. Gradient wrt bias
            grad_bias = None
            if self.bias is not None:
                grad_bias = np.sum(grad_output, axis=(0, 2, 3))

            # 2. Reshape grad_output for matrix multiplication
            # (Out_C, B * Oh * Ow)
            grad_output_reshaped = grad_output.transpose(1, 0, 2, 3).reshape(out_channels, -1)

            # 3. Gradient wrt weight
            # grad_weight = grad_output_reshaped @ x_cols.T
            # Need x_cols from forward pass. Since we don't store it in backward object for memory, 
            # we recompute it or rely on a stored version if we decide to cache it.
            # Recomputing for simplicity (matches standard autograd patterns)
            x_cols = im2col(self.x.data, kernel_h, kernel_w, self.stride, self.padding)
            grad_weight = np.matmul(grad_output_reshaped, x_cols.T)
            grad_weight = grad_weight.reshape(self.weight.shape)

            # 4. Gradient wrt input (grad_input)
            # grad_cols = w_matrix.T @ grad_output_reshaped
            w_matrix = self.weight.data.reshape(out_channels, -1)
            grad_cols = np.matmul(w_matrix.T, grad_output_reshaped)
            
            # 5. Transform columns back to image
            grad_input = col2im(grad_cols, self.x.shape, kernel_h, kernel_w, self.stride, self.padding)

        return grad_input, grad_weight, grad_bias

class Conv2d:
    """
    2D Convolution layer for spatial feature extraction.

    Implements convolution with explicit loops to demonstrate
    computational complexity and memory access patterns.

    Args:
       in_channels: Number of input channels
       out_channels:Number of output feature maps
       kernel_size:Size of convolution kernel(int or tuple)
       stride:Stride of convolution (default:1)
       padding:Zero-padding added to input (default: 0)
       bias: Whether to add learnable bias (default:True)
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, method='im2col'):
        """
        Initialize Conv2d layer with proper weight initialization.

        Args:
           in_channels: Number of input channels
           out_channels: Number of output feature maps
           kernel_size: Size of convolution kernel (int or tuple)
           stride: Stride of convolution (default: 1)
           padding: Zero-padding added to input (default: 0)
           bias: Whether to add learnable bias (default: True)
           method: Convolution implementation method ('naive' or 'im2col')
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.method = method

        #handling kernel_size as int or tuple
        if isinstance(kernel_size,int):
            self.kernel_size = (kernel_size,kernel_size)
        else:
            self.kernel_size = kernel_size

        self.stride = stride 
        self.padding = padding 

        # He intialization for ReLU networks
        kernel_h,kernel_w = self.kernel_size 
        fan_in = in_channels * kernel_h * kernel_w
        std = np.sqrt(2.0/fan_in)

        #Weight shape: (out_channels,in_channels,kernel_h,kernel_w)
        self.weight = Tensor(
            np.random.normal(0,std,
            (out_channels,in_channels,kernel_h,kernel_w)),
            requires_grad=True
        )

        #bias intialization
        if bias:
            self.bias = Tensor(np.zeros(out_channels),requires_grad=True)

        else:
            self.bias = None 
            
    def _compute_output_shape(self,in_h,in_w):
        """
        Calculates output spatial dimensions for convolution
        """

        kernel_h,kernel_w = self.kernel_size
        out_height = (in_h + 2 * self.padding - kernel_h) // self.stride + 1
        out_width = (in_w+2 *self.padding -kernel_w) // self.stride + 1
        return out_height,out_width

    def _apply_padding(self,x_data):
        """
        Zero-pads the spatial dimensions of the input numpy array
        """
        if self.padding > 0:
            return np.pad(
                x_data,
                ((0,0),(0,0),
                (self.padding,self.padding),
                (self.padding,self.padding)),
                mode='constant',
                constant_values=0
            )
        else:
            return x_data

    def _convolve_loops(self,padded,batch_size,out_h,out_w):
        """
        Core convolution with sliding window dot products over the input
        """ 

        out_channels = self.out_channels
        in_channels = self.in_channels
        kernel_h, kernel_w = self.kernel_size

        output = np.zeros((batch_size,out_channels,out_h,out_w))

        for b in range(batch_size):
            for out_ch in range(out_channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        in_h_start = oh*self.stride
                        in_w_start = ow* self.stride 

                        conv_sum = 0.0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                for in_ch in range(in_channels):
                                    input_val = padded[
                                        b,in_ch,
                                        in_h_start + k_h,
                                        in_w_start + k_w
                                    ]
                                    weight_val = self.weight.data[out_ch,in_ch,k_h,k_w]
                                    conv_sum += input_val * weight_val 

                        output[b,out_ch,oh,ow] = conv_sum

        return output


        
    def forward(self, x):
        """
        Forward pass through Conv2d layer
        """
        # input validation and shape extraction
        validate_4d_input(x, "Conv2D")

        batch_size, in_channels, in_height, in_width = x.shape

        # computing output dimensions
        out_height, out_width = self._compute_output_shape(in_height, in_width)

        if self.method == 'naive':
            # applying padding
            padded_input = self._apply_padding(x.data)

            # run convolution loops
            output = self._convolve_loops(padded_input, batch_size, out_height, out_width)

            # Adding bias if present
            if self.bias is not None:
                for out_ch in range(self.out_channels):
                    output[:, out_ch, :, :] += self.bias.data[out_ch]

        else:  # im2col method
            kernel_h, kernel_w = self.kernel_size
            
            # 1. Transform input into columns
            # Shape: (C * Kh * Kw, B * Oh * Ow)
            x_cols = im2col(x.data, kernel_h, kernel_w, self.stride, self.padding)
            
            # 2. Reshape weights into a matrix for matmul
            # Shape: (Out_C, C * Kh * Kw)
            w_matrix = self.weight.data.reshape(self.out_channels, -1)
            
            # 3. Compute output using matrix multiplication
            # Shape: (Out_C, B * Oh * Ow)
            output_cols = np.matmul(w_matrix, x_cols)
            
            # 4. Reshape back to image format (B, Out_C, Oh, Ow)
            output = output_cols.reshape(self.out_channels, batch_size, out_height, out_width).transpose(1, 0, 2, 3)

            # 5. Add bias
            if self.bias is not None:
                output += self.bias.data.reshape(1, -1, 1, 1)

        # Returning Tensor with gradient tracking enabled
        result = Tensor(output, requires_grad=(x.requires_grad or self.weight.requires_grad))

        # Attaching backward function for gradient computation
        if result.requires_grad:
            result._grad_fn = Conv2dBackward(
                x, self.weight, self.bias,
                self.stride, self.padding, self.kernel_size,
                self.method
            )
        return result




    def parameters(self):
        """Returns trainable parameters."""
        params = [self.weight]
        if self.bias is not None:
            params.append(self.bias)
        return params 

    def __call__(self,x):
        """Enable model(x) syntax"""
        return self.forward(x)
        
class MaxPool2dBackward(Function):
    """
    Gradient computation for 2D max pooling.

    Max pooling gradients flow only to the postions that
    were selected as the maximum in the forward pass
    """
    def __init__(self,x,output_shape,kernel_size,stride,padding):
        super().__init__(x)
        self.x = x
        self.output_shape = output_shape 
        self.kernel_size = kernel_size 
        self.stride = stride 
        self.padding = padding
        #stores max postions for gradient routing
        self.max_postionas = {}

    def apply(self,grad_output):
        """
        Routes gradients back to max position

        Args:
            grad_output :Gradient from next layer 

        Returs:
            Gradiet wrt input

        """
        batch_size,channels,in_height,in_width = self.x.shape 
        _,_,out_height,out_width = self.output_shape
        kernel_h,kernel_w = self.kernel_size

        #Applying padding if needed
        if self.padding > 0:
            padded_input = np.pad(
                self.x.data,
                ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),
                mode='constant',constant_values=-np.inf
            )
            grad_input_padded = np.zeros_like(padded_input)

        else:
            padded_input = self.x.data 
            grad_input_padded = np.zeros_like(self.x.data)
            
        #Routing gradients to max positions 
        for b in range(batch_size):
            for c in range(channels):
                for out_h in range(out_height):
                    for out_w in range(out_width):
                        in_h_start = out_h* self.stride
                        in_w_start = out_w* self.stride 

                        #Finding max position in this window
                        max_val = -np.inf
                        max_h,max_w = 0,0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                in_h = in_h_start + k_h
                                in_w = in_w_start + k_w
                                val = padded_input[b,c,in_h,in_w]
                                if val > max_val:
                                    max_val = val
                                    max_h,max_w = in_h,in_w 

                            #Routing gradient to max position
                            grad_input_padded[b,c,max_h,max_w] += grad_output[b,c,out_h,out_w]


        #Removing padding
        if self.padding > 0:
            grad_input = grad_input_padded[:,:,
                                            self.padding:-self.padding,
                                            self.padding:-self.padding,

                                            ]
        else:
            grad_input = grad_input_padded

        #Returning as tuple
        return (grad_input,)


class MaxPool2d:
    """
    2D Max Pooling layer for spatial dimensio reduction

    Applies maximum operation over spatial windows, preserving
    the strongest activations while reducing computational load

    Args:
        kernel_size:Size of pooling window (int or tuple)
        stride:Stride of pooling operation operation (default:same as kernel_size)
        padding:Zero-padding added to input (default:0)

    """ 
    def __init__(self,kernel_size,stride=None,padding=0):
        """
        Intializes MaxPool2d layer.

        """
        super().__init__()

        #handling kernel_size as int or tuple
        if isinstance(kernel_size,int):
            self.kernel_size = (kernel_size,kernel_size)
        else:
            self.kernel_size = kernel_size

        #default stride equal kernel_size which is non-overlapping
        if stride is None:
            self.stride = self.kernel_size[0]

        else:
            self.stride = stride 

        self.padding = padding 

    def _compute_pool_output_shape(self,in_h,in_w):
        """
        Calculates output spatial dimensions for pooling
        """

        kernel_h,kernel_w = self.kernel_size 
        out_height = (in_h + 2 *self.padding - kernel_w) // self.stride + 1
        out_width = (in_w + 2 * self.padding - kernel_w) // self.stride + 1
        return out_height,out_width

    def _maxpool_loops(self,padded,batch_size,channels,out_h,out_w):
        """
        Finds maximum value in each window
        """ 
        kernel_h,kernel_w = self.kernel_size
        output = np.zeros((batch_size,channels,out_h,out_w))

        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        in_h_start = oh*self.stride 
                        in_w_start = ow* self.stride 

                        max_val = -np.inf
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                input_val = padded[
                                    b,c,
                                    in_h_start + k_h,
                                    in_w_start + k_w
                                ]
                                max_val = max(max_val,input_val)

                        output[b,c,oh,ow] = max_val

        return output


    def forward(self,x):
        """
        Forward pass through MaxPool2d layer
        """
        #validating input
        validate_4d_input(x,"MaxPool2d")

        batch_size,channels,in_height,in_width =x.shape

        #Computing output dimensions
        out_height,out_width = self._compute_pool_output_shape(in_height,in_width)

        #Apply padding (we use -inf for maxpooling so padded values are never selected)
        if self.padding > 0:
            padded_input = np.pad(
                x.data,
                ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),
                mode='constant',constant_values=-np.inf
            )
        else:
            padded_input = x.data 

        #running max pooling loops
        output = self._maxpool_loops(padded_input,batch_size,channels,out_height,out_width)

        #Returning Tensor with gradient tracking
        result = Tensor(output,requires_grad=x.requires_grad)

        if result.requires_grad:
            result._grad_fn = MaxPool2dBackward(
                x,result.shape,self.kernel_size,self.stride,self.padding
            )

        return result 
    def parameters(self):
        """
        Returns empty list since pooling has no parameters
        """
        return []

    def __call__(self,x):
        """Enable model(x) syntax."""
        return self.forward(x)


class AvgPool2d:
    """
    2D Average Pooling layer for spatial dimension reduction

    Applies average operation over spatial windows, smoothing
    features while reducting computational load.

    Args:
        kernel_size:size of pooling window it can int or tuple
        stride:stride of pooling operation its default is same as kernel_size
        padding:zero-padding added to input (default:0)
    """

    def __init__(self,kernel_size,stride=None,padding=0):
        """
        Intialize AvgPool2d layer
        """
        super().__init__()

        #handling kernel_size as int or tuple
        if isinstance(kernel_size,int):
            self.kernel_size = (kernel_size,kernel_size)
        else:
            self.kernel_size = kernel_size

        #default strides equals kernel_size
        if stride is None:
            self.stride = self.kernel_size[0]
        else:
            self.stride = stride 

        self.padding = padding 

    def _compute_pool_output_shape(self,in_h,in_w):
        """
        Calculate output spatial dimensions for pooling
        """
        kernel_h,kernel_w = self.kernel_size
        out_height = (in_h + 2 * self.padding - kernel_h) // self.stride + 1
        out_width = (in_w + 2 * self.padding - kernel_w) // self.stride + 1
        return out_height,out_width

    def _avgpool_loops(self,padded,batch_size,channels,out_h,out_w):
        """
        Core average pooling it computes mean of each window
        """
        kernel_h,kernel_w = self.kernel_size
        output = np.zeros((batch_size,channels,out_h,out_w))

        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_h):
                    for ow in range(out_w):
                        in_h_start = oh * self.stride
                        in_w_start = ow * self.stride

                        window_sum = 0.0
                        for k_h in range(kernel_h):
                            for k_w in range(kernel_w):
                                input_val = padded[
                                    b, c,
                                    in_h_start + k_h,
                                    in_w_start + k_w
                                ]
                                window_sum += input_val
                        output[b,c,oh,ow] = window_sum / (kernel_h * kernel_w)

        return output 
                        
                            

    def forward(self,x):
        """
        Forward pass through AvgPool2d layer
        """

        #validating input
        validate_4d_input(x,"AvgPool2d")

        batch_size,channels,in_height,in_width = x.shape 

        #computes output dimensions
        out_height,out_width = self._compute_pool_output_shape(in_height,in_width)

        #Applying padding (here we use zeros for average pooling)
        if self.padding > 0:
            padded_input = np.pad(
                x.data,
                ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding)),
                mode='constant',constant_values=0
            )
        else:
            padded_input = x.data 

        #Run average pooling loops
        output = self._avgpool_loops(padded_input,batch_size,channels,out_height,out_width)

        #Return Tensor with gradient tracking
        result = Tensor(output,requires_grad=x.requires_grad)
        return result 

        

    def parameters(self):
        """Returns empty list since pooling has no parameters"""
        return []


    def __call__(self,x):
        """Enable model(x) syntax"""
        return self.forward(x)

class BatchNorm2d:
    """
    Batch Normalization for 2D spatial inputs (images)

    Normalizes activations across batch and spatial dimensions for each channel,
    then applies learnable scale (gamma) and shift (beta) parameters.

    Args:
        num_features:number of channels (C in NCHW format)
        eps:small constant for numerical stability (default:1e-5)
        momentum:Momentum for running statistics update (default:0.1)
    """      

    def __init__(self,num_features,eps=1e-5,momentum=0.1):
        """
        Initialize BatchNorm2d layer.

        """
        super().__init__()

        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        ##Learnable parameters(requires_grad=True for training)
        #gamma (scale): intialized to 1 so output = normalized input intially
        self.gamma =Tensor(np.ones(num_features),requires_grad=True)
        #beta (shift): initialized to 0 so no shif initially
        self.beta = Tensor(np.zeros(num_features),requires_grad=True)
        
        #Running statistivs which are not trained but accumulated during training
        #These are used during evaluation for consistent normalization
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)


        #Training mode flag
        self.training = True 

    def train(self):
        """
        Set layer to training mode.
        """
        return self 

    def eval(self):
        """
        Set layer to evaluation mode.
        """
        self.training= False 
        return self 

    def _validate_input(self,x):
        """
        Validating that input tensor has the correct shape for BatchNorm2d
        """
        if len(x.shape) !=4:
            if len(x.shape) ==3:
                raise ValueError(
                    f"BatchNorm2d expected 4D input (batch,channel,height,width), got got 3D: {x.shape}"
                    f"Missing batch dimension\n"
                    f"BatchNorm computes statistics over the batch dimension\n"
                    f"Add batch dim: x.reshape(1,{x.shape[0]},{x.shape[-1]},{x.shape[2]})"
                )

            elif len(x.shape) == 2:
                raise ValueError(
                    f"BatchNorm expected 4D input (batch,channels,height,width),got 2D:{x.shape}\n"
                    f"Got a matrix,expected an image tensor\n"
                    f"BatchNorm normalizes over spatial dimensions per channel"
                    f"If this is a flattened image, reshape it:x.reshape(1,channels,height,width)"


                )
            
            else:
                raise ValueError(
                    f"BatchNorm2d expected 4D input (batch,channels,height,width), got {len(x.shape)}D: {x.shape}\n"
                    f"Wrong number of dimensions\n"
                    f"BatchNorm expects:(batch_size,channels,height,width)\n"
                    f"Reshape your input to 4D with the correct dimensions"
                )

        batch_size,channels,height,width = x.shape

        if channels != self.num_features:
            raise ValueError(
                f"BatchNorm2d channel mismatch:expected {self.num_features} channels, got {channels}\n"
                f"Input has {channels} channels but BatchNorm2d was created for {self.num_features}\n"
                f"BatchNorm(num_features) must match the channel dimension of your input\n"
                F"Either fix your input shape or created BatchNorm2d({channels})"

            ) 


    def _get_stats(self,x):
        """
        Gets mean and variance for normalization i.e batch or running stats
        """

        if self.training:
            ##computing batch statistics per channel
            #mean over batch and spatial dimensions: axes (0,2,3)
            batch_mean = np.mean(x.data,axis=(0,2,3))
            batch_var = np.var(x.data,axis=(0,2,3))

            #updating running statistics i.e exponential moving average
            self.running_mean = (1-self.momentum)*self.running_mean + self.momentum * batch_mean
            self.running_var = (1-self.momentum)*self.running_var + self.momentum * batch_var 

            return batch_mean,batch_var
        else:
            #using running statistics which are frozen during eval
            return self.running_mean,self.running_var 

    def forward(self,x):
        """
        Forward pass function through BatchNorm2d

        This function composes _validate_input,_get_stats 
        and normalize+scale
        """
        self._validate_input(x)

        batch_size,channels,height,width = x.shape 
        mean,var = self._get_stats(x)

        #Normalization:(x-mean) /sqrt(var + eps)
        #Reshaping mean and var for broadcasting:(C,) to (1,C,1,1)
        mean_reshaped = mean.reshape(1,channels,1,1)
        var_reshaped = var.reshape(1,channels,1,1)

        x_normalized = (x.data - mean_reshaped) / np.sqrt(var_reshaped + self.eps)

        #Applying scale (gamma)  and shift(beta)
        gamma_reshaped = self.gamma.data.reshape(1,channels,1,1)
        beta_reshaped = self.beta.data.reshape(1,channels,1,1)

        output = gamma_reshaped * x_normalized + beta_reshaped

        #returning Tensor with gradient tracking
        result =Tensor(output,requires_grad=x.requires_grad or self.gamma.requires_grad)

        return result 

    def parameters(self):
        """
        Returns learnable parameters(gamma and beta)

        """
        return [self.gamma,self.beta]
        
    def __call__(self,x):
        """
        Enable model(x) syntax
        """
        return self.forward(x)





              