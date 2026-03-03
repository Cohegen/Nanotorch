import os
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
from embeddings import Embedding
import math
import numpy as np

def testing_embedding_init():
    """
    This function is intended to test Embeddig.__init__
    """
    embed = Embedding(vocab_size=100,embed_dim=64)

    #checking stored attributes
    assert embed.vocab_size == 100, f"Expected vocab_size=100,got {embed.vocab_size}"
    assert embed.embed_dim == 64, f"Expected embed_dim=64,got {embed.embed_dim}"

    #checking weight shape
    assert embed.weight.shape == (100,64),f"Expected weight shape (100,64), got {embed.weight.shape}"

    #checking Xavier bounds: limits = sqrt(6/(100+64)) = 0.191
    limit =  math.sqrt(6.0/ (100+64))
    assert np.all(embed.weight.data >= -limit - 1e-6), "Weight should be >= -limit"
    assert np.all(embed.weight.data <= limit + 1e-6), "Weight should be <= limit"

    print("Embedding.__init__ works correctly")

if __name__ == "__main__":
    testing_embedding_init()