import os
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
from embeddings import Embedding,PositionalEncoding
import numpy as np
import math 

def testing_positional_encoding():
    """
    This function intends to test Position embedding consistency and shape handling
    """

    #testing basic functionality
    pos_enc = PositionalEncoding(max_seq_len=512,embed_dim=64)

    #creating sample embeddings
    embeddings = Tensor(np.random.randn(2,10,64))
    output = pos_enc.forward(embeddings)

    assert output.shape == (2,10,64),f"Expected shape (2,10,64), got {output.shape}"

    #testing Position consistency
    #same position should always get same encoding
    emb1 = Tensor(np.zeros((1,5,64)))
    emb2 = Tensor(np.zeros((1,5,64)))

    out1 = pos_enc.forward(emb1)
    out2 = pos_enc.forward(emb2)

    assert np.allclose(out1.data,out2.data),"Postion encoding should be consistent"

    #testing differnet postions to get different encodings
    short_emb = Tensor(np.zeros((1,3,64)))
    long_emb =Tensor(np.zeros((1,5,64)))

    short_out =pos_enc.forward(short_emb)
    long_out = pos_enc.forward(long_emb)

    #first 3 positions should match otherwise an error will rise
    assert np.allclose(short_out.data,long_out.data[:,:3,:]),"Position encoding prefix should match"

    #testing parameters
    params  = pos_enc.parameters()
    assert len(params)==1 ,"Should have 1 parameter(position embeddings)"
    assert params[0].shape == (512,64),"Position embedding matrix has wrong shape"

    print("Positional encoding works correctly.")

if __name__ == "__main__":
    testing_positional_encoding()