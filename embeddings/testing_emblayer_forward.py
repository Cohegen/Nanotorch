import os
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor, tensor 
from embeddings import Embedding,EmbeddingLayer,PositionalEncoding,_compute_sinusoidal_table,create_sinusoidal_embeddings
import numpy as np
import math 

def testing_emblayer_forward():
    """
    This function validates the forward composition i.e 
    token lookup + scaling + positional encoding addition across all 
    Positional encoding strategies
    """

    tokens = Tensor([[1,2,3],[4,5,6]])

    #testing the learned position PE forward 
    embed_learned = EmbeddingLayer(vocab_size=100,embed_dim=64,max_seq_len=128,pos_encoding='learned')
    output_learned = embed_learned.forward(tokens)
    assert output_learned.shape == (2,3,64),f"Expected (2,3,4), got {output_learned.shape}"

    #testing sinusodial PE forward 
    embed_sin = EmbeddingLayer(vocab_size=100,embed_dim=64,pos_encoding='sinusoidal')
    output_sin = embed_sin.forward(tokens)
    assert output_sin.shape == (2,3,64),"Sinusoidal should produce same shape"

    #testing with No PE strategy selected
    embed_none = EmbeddingLayer(vocab_size=100,embed_dim=64,pos_encoding=None)
    output_none = embed_none.forward(tokens)
    assert output_none.shape == (2,3,64),"No PE should produce same shape"

    #testing to see how ID input is handled
    tokens_1d = Tensor([1,2,3])
    output_1d = embed_learned.forward(tokens_1d)
    assert output_1d.shape == (3,64),f"Expected (3,64) for 1D input, got {output_1d.shape}"

    #testing embedding scaling by sqrt(embed_dim)
    embed_scaled = EmbeddingLayer(vocab_size=100,embed_dim=64,pos_encoding=None,scale_embeddings=True)
    embed_scaled.token_embedding.weight = embed_none.token_embedding.weight 
    output_scaled = embed_scaled.forward(tokens)
    output_unscaled = embed_none.forward(tokens)
    scale_factor = math.sqrt(64)
    assert np.allclose(output_scaled.data,output_unscaled.data*scale_factor,rtol=1e-5),"Scaling broken"

    print("EmbeddingLayer forward pass works correctly ")

if __name__ == "__main__":
    testing_emblayer_forward()
