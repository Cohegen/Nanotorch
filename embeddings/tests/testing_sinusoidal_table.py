import os
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
from embeddings import Embedding,PositionalEncoding,_compute_sinusoidal_table
import numpy as np
import math 

def testing_sinusoidal_table():
    """
    This function intends to validate whether _compute_sinusoidal_table function in
    embeddings.py builds the raw sin/cos table before it gets wrapped in a Tensor.
    """

    #testing shape and dtype
    table = _compute_sinusoidal_table(10,8)
    assert table.shape == (10,8), f"Expected (10,8), got {table.shape}"
    assert table.dtype == np.float32,f"Expected float32,got {table.dtype}"

    #testing Position 0 pattern i.e sin(0)=0 at even, cos(0)=1 at odd
    assert np.allclose(table[0,0::2],0,atol=1e-6),"Even dims at pos 0 should be sin(0)=0"
    assert np.allclose(table[0,1::2],1,atol=1e-6),"Odd dim at pos 0 should be cos(0)=1"

    #testing frequency decay i.e higher dims change slower
    table_100 = _compute_sinusoidal_table(100,16)
    fast_changes = np.sum(np.abs(np.diff(table_100[:10,0])))
    slow_changes = np.sum(np.abs(np.diff(table_100[:100,-1])))
    assert fast_changes > slow_changes,"Lower dims should oscillate faster"

    #testing odd embed_dim
    table_odd = _compute_sinusoidal_table(5,7)
    assert table_odd.shape == (5,7),"should handle odd embed_dim"

    #testing whether it returns numpy array
    assert isinstance(table,np.ndarray),"Helper function should return raw numpy array"

    print("Sinusoidal table computation works perrrrrfectly!!!!")

if __name__ == "__main__":
    testing_sinusoidal_table()
