import os
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
from embeddings import Embedding 
import numpy as np

def testing_embedding():
    """
    This function tests token embedding lookup and parameter management
    """
    #testing basic embedding creation and forward pass
    embed = Embedding(vocab_size=100,embed_dim=64)

    #single sequence
    tokens = Tensor([1,2,3])
    output = embed.forward(tokens)

    assert output.shape == (3,64),f"Expected shape (3,64), got {output.shape}"
    assert len(embed.parameters()) == 1,"Should have 1 parameter(weight matrix)"
    assert embed.parameters()[0].shape == (100,64),"Weight matrix has wrong shape"

    #testing batch processing
    batch_tokens = Tensor([[1,2,3],[4,5,6]])
    batch_output = embed.forward(batch_tokens)

    assert batch_output.shape == (2,3,64),f"Expected batch shape (2,3,64), got {batch_output.shape}"

    #testing embedding lookup consistency
    single_lookup = embed.forward(Tensor([1]))
    batch_lookup = embed.forward(Tensor([[1]]))

    #should get same embedding for same token
    assert np.allclose(single_lookup.data[0],batch_lookup.data[0,0]),"Inconsistent embedding lookup "

    #testing parameter access
    params = embed.parameters()
    assert len(params) == 1,"Should have 1 parameter"

    print("Embedding layer works gooooood!")

if __name__ == "__main__":
    testing_embedding()