import os 

import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from acceleration import vectorized_matmul,fused_gelu
import numpy as np

def testing_fused_gelu():
    """
    This function intends to test whether the fused_gelu function works correctly
    """

    #testing basic properties
    x = Tensor([-3,-1,0,1,3])
    result = fused_gelu(x)

    #gelu(0) = 0
    assert abs(result.data[2]) <1e-6,f"GELU(0) should be 0, got {result.data[2]}"

    #GELU is smooth and increasing
    assert result.data[4] > result.data[3] > result.data[2], "GELU should be increasing"

    #if GELU has positive bias 
    assert result.data[3] >0.8, "GELU(1) should be close to 1"
    assert result.data[1] > -0.2, "GELU(-1) should be slightly negative"

    #test numerically stability with extreme values
    x_extreme = Tensor([-10,-5,0,5,10])
    result_extreme = fused_gelu(x_extreme)

    assert not np.any(np.isnan(result_extreme.data)), "No NaN values allowed"
    assert not np.any(np.isinf(result_extreme.data)),"No infinite values allowed"

    #testing large tensor processing
    x_large = Tensor(np.random.randn(1000,1000).astype(np.float32))
    result_large= fused_gelu(x_large)

    assert result_large.shape== x_large.shape, "Shape preservation failed"
    assert result_large.data.dtype == np.float32, "Data type preservation failed"

    #testing that positive inputs are mostly preserved (GELU ≈ x for large positive x )
    x_positive = Tensor([5.0])
    result_positive = fused_gelu(x_positive)
    assert result_positive.data[0] > 4.9, "Large positive values should be nearly preserved "

    print("fused_gelu works correctly")

if __name__ == "__main__":
    testing_fused_gelu()