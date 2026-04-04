import numpy as np
import time 
from typing import Tuple,Optional,Dict,List
import os 
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Tensor import Tensor 
from memoization import KVCache,_cached_attention_forward

def testing_attention_forward():
    """
    This function intends to test whether the _cached_attention_forward function works

    """
    #tracks which path was taken
    path_taken = []

    class MockBlock:
        def __init__(self):
            self.attention = self

    block =MockBlock()

    def mock_original_forward(x,mask=None):
        path_taken.append("original")
        return x

    #creating a real cache for testing
    cache = KVCache(batch_size=1,max_seq_len=64,num_layers=2,num_heads=4,head_dim=32)

    # Testing PATH 1: Training (seq_len > 1)
    path_taken.clear()
    x_train = Tensor(np.random.randn(1, 10, 128))  # seq_len=10
    result = _cached_attention_forward(block, x_train, cache, 0, mock_original_forward)
    assert "original" in path_taken, "Training path should use original forward"
    assert result.shape == x_train.shape, "Should return same shape"

     # Testing PATH 2: First token (cache empty, seq_pos=0)
    path_taken.clear()
    cache.reset()
    assert cache.seq_pos == 0
    x_first = Tensor(np.random.randn(1, 1, 128))  # seq_len=1, but cache empty
    result = _cached_attention_forward(block, x_first, cache, 0, mock_original_forward)
    assert "original" in path_taken, "First token should use original forward"

    
    print(" _cached_attention_forward path dispatch works correctly!")

if __name__ == "__main__":
    testing_attention_forward()