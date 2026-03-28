from math import sqrt
import os 
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import numpy as np
import time
from typing import Dict,List,Tuple,Optional,Any,Union
import warnings

#constants for performance measurement
DEFAULT_WARMUP_ITERATIONS = 2 #Default warmup iterations for timing
DEFAULT_TILING_ITERATIONS = 5 #Default timing iterations for measurement
BYTES_PER_FLOAT32 = 4 #standard float32 size in bytes

from Tensor import Tensor 

def vectorized_matmul(a:Tensor,b:Tensor) ->Tensor:
    """
    High-Performance matrix multiplication using vectorized operations.

    This implemenation leverages optimized BLAS libraries that use:
        - SIMD instructions for parallel computation
        - Cache-blocking for memory efficiency
        - Multi-threadding for CPU parallelization

    Args:
       a :First tensor for multiplication (Mxk for batchxMxK)
       b:Second tensor for multiplication (KxN or batchxKxN)
    
    Returns:
        Result tensor of shape (MxN or batchxMxN)

     EXAMPLE:
    Matrix multiplication visualization:
    >>> a = Tensor([[1, 2], [3, 4]])  # 2×2
    >>> b = Tensor([[5, 6], [7, 8]])  # 2×2
    >>> result = vectorized_matmul(a, b)
    >>> print(result.data)
    [[19 22]    # [1×5+2×7, 1×6+2×8] = [19, 22]
     [43 50]]   # [3×5+4×7, 3×6+4×8] = [43, 50] 
    """

    #input validation for matrix multiplication
    if len(a.shape) < 2 or len(b.shape) < 2:
        raise ValueError(
            f"Matrix Multiplication requires 2D+ tensors\n"
            f"  Got shapes{a.shape} and {b.shape} ({len(a.shape)}D and {len(b.shape)}D tensors)\n"\
            f"  Matrix multiplication computes dot products between rows and columns,which requires at least 2D tensors\n"
            f"  Add dimensions with reshape: a.reshape(1,{a.shape[-1] if len(a.shape) >=1 else'n' }) for row vector"

        )
    
    if a.shape[-1] != b.shape[-2]:
        raise ValueError(
            f"Matrix Multiplication shape mismatch: {a.shape} @{b.shape}\n"
            f" Inner dimensions don't match: a.shape[-1]={a.shape[-1]} vs b.shape[-2]={b.shape[-2]}\n"
            f" For A @ B, each row of A (length {a.shape[-1]}) must match each column of B (length{b.shape[-2]})\n"
            f"Try: b.reshape({a.shape[-1]},1) or a.reshape(-1,{b.shape[-2]})"

        )

    #we use Numpy's highly optimized matrix multiplication
    #This calls BLAS GEMM(General Matrix Multiply),which uses:
    # - SIMD vectorization for parallel arithmetic
    # cache blocking for memory efficiency
    #Multi-threading on multicore systems
    result_data = np.matmul(a.data,b.data)

    return Tensor(result_data)
        

def fused_gelu(x:Tensor)->Tensor:
    """
    Fused GELU activation that combines all operations in a single kernel


    GELU combines the benefits of RELU and sigmoid:
        - It is smooth everywhere(unlike RELU's discontinuity at 0)
        - It is non-saturating for positive values (unlike sigmoid)
        - It probabilistic interpretation: x*P(X<= x) where X~N(0,1)

    Mathematical Definition:
    GELU(x) = x* Φ(x) where Φ(x) is the standard normal CDF

    Fast Approximation
     GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))

     Args:
        x: Input tensor to apply GELU activation

    Returns:
        GELU-activated tensor (same shape as input)

    EXAMPLE:
    >>> x = Tensor([-2, -1, 0, 1, 2])
    >>> result = fused_gelu(x)
    >>> print(result.data)
    [-0.04550026 -0.15865526  0.          0.8413447   1.9544997 ]
    # Notice: smooth transition through 0, positive bias
    """

    #mathematical constant for GELU approximation
    sqrt_2_over_pi = np.sqrt(2.0/np.pi)

    #fused GELU computation all operations in single expression
    #This minimizes memory bandwidth by avoiding intermediate arrays
    #Numpy's expression evaluator will optimize this into efficient machine code
    result_data = 0.5* x.data * (
          1.0 + np.tanh(sqrt_2_over_pi * (x.data + 0.044715 * x.data**3))
    )
    return Tensor(result_data)

def unfused_gelu(x:Tensor) -> Tensor:
    """
    Deliberately unfused GELU implementation to act as the
    opponent of the fused gelu kernel.

    This version creates multiple intermediate tensors to simulate
    the memory bandwidth overhead of unfused operations.

    Args:
       x:Input tensor

    Returns:
        GELU-activated tensor

     EXAMPLE:
    >>> x = Tensor([0.5, 1.0, -0.5])
    >>> result = unfused_gelu(x)
    >>> print(result.shape)
    (3,)  # Same as input
    """
    #unfused version of gelu i.e it creates intermediate arrays
    sqrt_2_over_pi = np.sqrt(2.0/np.pi)

    #each operation creates a temporary array (simulating kernel launches)
    temp1 = Tensor(x.data**3)  # x³
    temp2 = Tensor(0.044715 * temp1.data)  # 0.044715 * x³
    temp3 = Tensor(x.data + temp2.data)  # x + 0.044715 * x³
    temp4 = Tensor(sqrt_2_over_pi * temp3.data)  # √(2/π) * (...)
    temp5 = Tensor(np.tanh(temp4.data))  # tanh(...)
    temp6 = Tensor(1.0 + temp5.data)  # 1 + tanh(...)
    temp7 = Tensor(x.data * temp6.data)  # x * (1 + tanh(...))
    result = Tensor(0.5 * temp7.data)  # 0.5 * x * (...)

    return result

