import os 
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
import numpy as np
from transformers import LayerNorm ,MLP

def testing_mlp():
    """
    This function intends to test shape preservation 
    and parameter counting
    """

    #testing MLP with standard 4x expansion
    embed_dim = 64
    mlp =MLP(embed_dim)

    #testing forward pass
    batch_size,seq_len = 2,10
    x = Tensor(np.random.randn(batch_size,seq_len,embed_dim))
    output = mlp.forward(x)

    #checking shape preservation
    assert output.shape == (batch_size,seq_len,embed_dim)

    #checking hidden dimension is 4x
    assert mlp.hidden_dim == 4* embed_dim 

    #testing parameter counting
    params = mlp.parameters()
    expected_params = 4# 2 weights + 2 biases
    assert len(params) == expected_params

    #testing custom hidden dimension
    custom_mlp = MLP(embed_dim,hidden_dim=128)
    assert custom_mlp.hidden_dim == 128 

    print("MLP works correctly")

if __name__ == "__main__":
    testing_mlp()