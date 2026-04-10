import os
import sys
from unittest import TestLoader


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
from embeddings import Embedding,PositionalEncoding,_compute_sinusoidal_table,create_sinusoidal_embeddings
from embeddings import EmbeddingLayer 
import numpy as np
import math 

def testing_embedLayer():
    """
    This function intends to test whether `__init__` method works correctly 
    assembles sub-components for each positional encoding strategy
    """

    #testing whether Learned PE creates PositionalEncoding
    layer_learned = EmbeddingLayer(vocab_size=100,embed_dim=64,pos_encoding='learned')
    assert isinstance(layer_learned.token_embedding,Embedding),"Should create Embedding"
    assert isinstance(layer_learned.pos_encoding,PositionalEncoding),"Should create PositionalEncoding"
    assert len(layer_learned.parameters())== 2, "Learned PE: 2 param tensors (token + position)"

    #testing whether Sinusoidal PE creates fixed Tensor
    layer_sin = EmbeddingLayer(vocab_size=100,embed_dim=64,pos_encoding='sinusoidal')
    assert isinstance(layer_sin.pos_encoding,Tensor),"Sinusoidal PE should be a Tensor"
    assert len(layer_sin.parameters()) == 1, "Sinusoidal PE: 1 param tensor (token only)"

    #testing whether an Invalid PE raises ValueEror
    try:
        EmbeddingLayer(vocab_size=100,embed_dim=64,pos_encoding='invalid')
        assert False, "Should raise ValueError for invalid pos_encoding"
    except ValueError:
        pass 

    #testing whether configuration stored correctly
    assert layer_learned.vocab_size == 100
    assert layer_learned.embed_dim == 64 
    assert layer_learned.scale_embeddings == False 

    print("EmbeddingLayer intialization works correctly")

if __name__ == "__main__":
    testing_embedLayer()