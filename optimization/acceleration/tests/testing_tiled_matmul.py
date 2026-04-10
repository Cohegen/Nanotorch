import os 

import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from acceleration import vectorized_matmul,DEFAULT_TILING_ITERATIONS, DEFAULT_WARMUP_ITERATIONS, tiled_matmul
import numpy as np

def testing_tiled_matmul():
    """
    This function tests cache-aware tiled matrix multiplications
    """

    #testing correctness against vectorized matmul version
    a = Tensor(np.random.randn(128,128).astype(np.float32))
    b =Tensor(np.random.randn(128,128).astype(np.float32))

    result_tiled = tiled_matmul(a,b,tile_size=32)
    result_vectorized = vectorized_matmul(a,b)

    assert np.allclose(result_tiled.data,result_vectorized.data,atol=1e-5), \
        "Tiled and Vectorized results should match"

    #testing different tile sizes
    for tile_size in [16,32,64]:
        result = tiled_matmul(a,b,tile_size=tile_size)
        assert result.shape == (128,128),f"Wrong shape for tile_size={tile_size}"

    #tests for shape validation
    try:
        wrong_a = Tensor(np.random.randn(128, 64).astype(np.float32))
        wrong_b = Tensor(np.random.randn(128, 64).astype(np.float32))
        tiled_matmul(wrong_a, wrong_b)
        assert False, "Should have raised ValueError for shape mismatch"
    except ValueError as e:
        assert "shape mismatch" in str(e).lower()

    print("tiled_matmul works correctly!")

if __name__ == "__main__":
    testing_tiled_matmul()
