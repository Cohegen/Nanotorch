import os
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
from embeddings import Embedding,PositionalEncoding
import numpy as np
import math 

def testing_positional_encoding_init():
    """
    This function is intended to test whether the Position embedding matrix intialization
    with correct shape happens correctly
    """
    pos_enc = PositionalEncoding(max_seq_len=512,embed_dim=64)

    #checking stored attributes
    assert pos_enc.max_seq_len == 512, f"Expected max_seq_len = 512, got{pos_enc.max_seq_len}"
    assert pos_enc.embed_dim == 64, f"Expected embed_dim=64, got{pos_enc.embed_dim}"

    #checking positional embeddings shape 
    assert pos_enc.position_embeddings.shape == (512,64), \
        f"Expected shape (512,64), got {pos_enc.position_embeddings.shape}"

     # Check values are reasonably small (additive initialization)
    limit = math.sqrt(2.0 / 64)
    assert np.all(pos_enc.position_embeddings.data >= -limit - 1e-6), "Values should be >= -limit"
    assert np.all(pos_enc.position_embeddings.data <= limit + 1e-6), "Values should be <= limit"
    #checking parameters returns the position embeddings
    params = pos_enc.parameters()
    assert len(params) == 1, f"Expected 1 parameter, got {len(params)}"

    print("PositonalEncoding.__init__ works correctly")

if __name__ == "__main__":
    testing_positional_encoding_init()
    