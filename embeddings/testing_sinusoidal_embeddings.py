import os
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
from embeddings import Embedding,PositionalEncoding,_compute_sinusoidal_table,create_sinusoidal_embeddings
import numpy as np
import math 

def testing_sinusoidal_embeddings():
    """
    This function validates whether our sinusoidal positional encoding function creates the correct
    mathematical patterns

    """
    
    #testing basic shape and properties
    pe = create_sinusoidal_embeddings(512,64)
    assert pe.shape == (512,64),f"Expected shape (512,64), got {pe.shape}"

    #testing whether position 0 is mostly zeros and ones
    pos_0 = pe.data[0]

    #even indices should be sin(0) = 0
    assert np.allclose(pos_0[0::2],0,atol=1e-6),"Even indices at position 0 should be ~0"

    #odd indices should be cos(0) = 1
    assert np.allclose(pos_0[1::2],1,atol=1e-6),"Odd indices at position 0 should be ~1"

    #testing whether different positions have different encodings
    pe_small = create_sinusoidal_embeddings(10,8)

    #checking that consecutive positions are different
    for i in range(9):
        assert not np.allclose(pe_small.data[i],pe_small.data[i+1]),f"Positions {i} and {i+1} are too similar"

    #testing frequency properties
    #higher dimensions should have lower frequencies since they change more slowly
    pe_test = create_sinusoidal_embeddings(100,16)

    #first dimension should change faster than the last dimension
    first_dim_changes = np.sum(np.abs(np.diff(pe_test.data[:10,0])))
    last_dim_changes = np.sum(np.abs(np.diff(pe_test.data[:10,-1])))

    assert first_dim_changes > last_dim_changes,"Lower dimensions should change faster than higher dimensions"

    #testing odd embed_dim handling
    pe_odd = create_sinusoidal_embeddings(10,7)
    assert pe_odd.shape == (10,7), "Should handle odd embedding dimensions"

    #testing whether a Tensor is returned instead of a numpy array
    assert isinstance(pe,Tensor),"Should return a Tensor wrapping the sinusoidal table"

    print("Sinusoidal embeddings work correctly")

if __name__ == "__main__":
    testing_sinusoidal_embeddings()
