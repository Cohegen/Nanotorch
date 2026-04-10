import os
import sys

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 

from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from attention import _compute_attention_scores,_scale_scores,MultiHeadAttention
import numpy as np
import math 

def testing_unit_merge_heads():
    """
    This function tests merging reshape where why try to reshape 4D to 3D correclty to recombine
    heads to form embeddingd
    """
    mha = MultiHeadAttention(embed_dim=64,num_heads=8)

    #creating 4D tensor as if from split_heads
    x_4d = Tensor(np.random.randn(2,8,10,8))
    merged = mha._merge_heads(x_4d,2,10)
    assert merged.shape == (2,10,64),f"Expected (2,10,64),got{merged.shape}"

    #verify round-trip:split then merge ercovers original data 
    original = Tensor(np.random.randn(2,10,64))
    split = mha._split_heads(original,2,10)
    recovered = mha._merge_heads(split,2,10)
    assert np.allclose(original.data,recovered.data),"Split->merge should recover orginal data"
    print("Merge heads: correct 3D shape and round-trip!")

if __name__ == "__main__":
    testing_unit_merge_heads()