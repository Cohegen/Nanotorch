import os
import sys

 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
from embeddings import Embedding,PositionalEncoding,_compute_sinusoidal_table,create_sinusoidal_embeddings
from embeddings import EmbeddingLayer
import numpy as np
import math 

def testing_embedding_full():
    """
    This function intends to test the full embedding system i.e 
    token + positional embeddings intergration,scaling and batch processing
    """

    #testing Learned positional encoding
    embed_learned = EmbeddingLayer(
        vocab_size=100,
        embed_dim=64,
        max_seq_len=128,
        pos_encoding='learned'
    )

    tokens = Tensor([[1,2,3],[4,5,6]])
    output_learned= embed_learned.forward(tokens)

    assert output_learned.shape == (2,3,64),f"Expected shape (2,3,64),got{output_learned.shape} "

    #testing Sinusoidal positional encoding
    embed_sin = EmbeddingLayer(
        vocab_size=100,
        embed_dim=64,
        pos_encoding='sinusoidal'
    )

    output_sin = embed_sin.forward(tokens)
    assert output_sin.shape == (2,3,64),"Sinusoidal embedding should have same shape"

    #testing positional encoding
    embed_none = EmbeddingLayer(
        vocab_size=100,
        embed_dim=64,
        pos_encoding=None 
    )

    output_none = embed_none.forward(tokens)
    assert output_none.shape == (2,3,64),"No pos encoding should have same shape"

    #testing 1D input handling
    token_1d = Tensor([1,2,3])
    output_1d = embed_learned.forward(token_1d)

    assert output_1d.shape == (3,64),f"Expected shape (3,64) for 1D input, got {output_1d.shape}"
    
    #testing embedding scaling 
    embed_scaled = EmbeddingLayer(
        vocab_size=100,
        embed_dim=64,
        pos_encoding=None,
        scale_embeddings=True 
    )

    #using same weights to ensure fair comparision
    embed_scaled.token_embedding.weight = embed_none.token_embedding.weight 

    output_scaled = embed_scaled.forward(tokens)
    output_unscaled = embed_none.forward(tokens)

    #scaled version should be sqrt(64) times larger
    scale_factor = math.sqrt(64)
    expected_scaled = output_unscaled.data * scale_factor
    assert np.allclose(output_scaled.data,expected_scaled,rtol=1e-5),"Embedding scaling not working correctly"

    #testing parameter counting
    params_learned =embed_learned.parameters()
    params_sin = embed_sin.parameters()
    params_none = embed_none.parameters()

    assert len(params_learned) == 2, "Learned encoding should have 2 parameter tensors"
    assert len(params_sin) == 1, "Sinusoidal encoding should have 1 parmeter tensor"
    assert len(params_none)==1,"No pos encoding should have 1 parameter tensor"

    print("Complete embedding system works correctly")


if __name__ == "__main__":
    testing_embedding_full()

