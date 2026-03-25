import os 
import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from compression import measure_sparsity,structured_prune,low_rank_approximate
import numpy as np

def testing_low_rank_approximate():
    """
    This function tests whether the Truncated SVD decomposition and reconstruction
    happens within the low_rank_approximation works
    """

    #creating a test weight matrix
    original_weight = np.random.randn(20,15)
    original_params = original_weight.size 

    #applying low-rank approximation
    U,S,V = low_rank_approximate(original_weight,rank_ratio=0.4)

    #checking dimensions
    target_rank = int(0.4*min(20,15)) #min (20,15) = 15 so 0.4*15 = 6
    assert U.shape == (20,target_rank),f"Expected U shape (20,{target_rank}), got {U.shape}"
    assert S.shape == (target_rank,), f"Expected Shape ({target_rank},), got {S.shape}"
    assert V.shape == (target_rank,15),f"Expected V shape ({target_rank},15), got {V.shape}"

    #checking parameter reduction
    compressed_params = U.size + S.size + V.size 
    compression_ratio = compressed_params / original_params
    assert compression_ratio <1.0,f"Should compress, but ratio is {compression_ratio}"

    #checking reconstruction quality
    reconstructed = U @ np.diag(S) @ V
    reconstruction_error = np.linalg.norm(original_weight-reconstructed)
    relative_error = reconstruction_error / np.linalg.norm(original_weight)

    #low-rank approxiation trades accuracy for compression error is expected
    assert relative_error < 0.7,f"Reconstruction error too high: {relative_error}"

    print("low_rank_approximate works correctly")

if __name__ == "__main__":
    testing_low_rank_approximate()
