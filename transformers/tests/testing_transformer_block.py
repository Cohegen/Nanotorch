import os 
import sys


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
import numpy as np
from transformers import LayerNorm ,MLP, TransformerBlock

def testing_transformer_block():
    """
    This function intends to test whether there's
    shape preservation,residual connections,parameter counting
    """
    #testing transformer block
    embed_dim = 64
    num_heads = 4
    block = TransformerBlock(embed_dim,num_heads)
    
    #testing forward pass
    batch_size, seq_len = 2,8
    x  = Tensor(np.random.randn(batch_size,seq_len,embed_dim))
    output = block.forward(x)
    #check shape preservation
    assert output.shape == (batch_size,seq_len,embed_dim)

    #testing with causal mask (for autoregressive generation)
    mask = Tensor(np.triu(np.ones((seq_len,seq_len))* -np.inf,k=1))
    masked_output = block.forward(x,mask)
    assert masked_output.shape == (batch_size,seq_len,embed_dim)

    #testing parameter counting
    params = block.parameters()
    expected_components = 4 # attention,layer_norm1,layer_norm2,mlp parameters
    assert len(params) > expected_components #should have parameters from all components

    #test different configurations
    large_block = TransformerBlock(embed_dim=128,num_heads=8,mlp_ratio=2)
    assert large_block.mlp.hidden_dim == 256 #128*2

    print("Tranformer block works correctly")

if __name__ == "__main__":
    testing_transformer_block()