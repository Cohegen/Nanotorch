import os
import sys



sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 

from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from attention import _compute_attention_scores,_scale_scores
from attention import MultiHeadAttention
import numpy as np
import math 

def testing_multihead_attention():
    """
    This function intends to test configuration,parameter counting,shape preservation and masking support
    """
    #testing intialization phase 
    embed_dim,num_heads = 64,8
    mha = MultiHeadAttention(embed_dim,num_heads)

    #checking configuration
    assert mha.embed_dim == embed_dim
    assert mha.num_heads == num_heads
    assert mha.head_dim == embed_dim // num_heads

    #testing parameter counting (4 Linear layers, each has weight + bias)
    params = mha.parameters()
    assert len(params) == 8, f"Expected 8 parameters (4 layers x 2 ), got {len(params)}"

    #testing forward pass 
    batch_size,seq_len = 2,6
    x = Tensor(np.random.randn(batch_size,seq_len,embed_dim))

    output = mha.forward(x)

    #checking output shape preservation
    assert output.shape == (batch_size,seq_len,embed_dim),f"Output shape {output.shape} incorrect"

    #testing with casual mask
    mask = Tensor(np.tril(np.ones((batch_size,seq_len,seq_len))))
    output_masked = mha.forward(x,mask)
    assert output_masked.shape == (batch_size,seq_len,embed_dim)

    #testing different head configurations
    mha_small = MultiHeadAttention(embed_dim=32,num_heads=4)
    x_small = Tensor(np.random.randn(1,5,32))
    output_small = mha_small.forward(x_small)
    assert output_small.shape == (1,5,32)

    print("MultiHeadAttention works correctly")

if __name__ == "__main__":
    testing_multihead_attention()   