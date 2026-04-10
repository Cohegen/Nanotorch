import numpy as np
import time 
from typing import Tuple,Optional,Dict,List
import os 
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Tensor import Tensor 
from memoization import KVCache,_create_cache_storage

def test_cache_storage():
    """
    This function intends to test whether the _create_cache_storage
    helper function works correctly
    """

    #mock model with valid attributes
    class MockGPT:
        def __init__(self):
            self.embed_dim = 128
            self.num_layers = 4
            self.num_heads = 4
            self.max_seq_len=64
            self.blocks = [None]*4 #placeholder blocks

    #testing whether the model creates cache
    model = MockGPT()
    cache,head_dim = _create_cache_storage(model)
    assert head_dim == 32, f"Expected head_dim=32 got {head_dim}"
    assert cache.num_layers == 4, "Cache layers should match model"
    assert cache.max_seq_len ==64, "Cache max_seq should match model"
    assert model._cache_enabled==True,"Model should be flagged as cache-related"
    assert model._kv_cache is cache, "Cache should be attached to the model"

    #testing whether missing attributes raises AttributeError
    class IncompleteModel:
        def __init__(self):
            self.embed_dim = 128

    try:
        _create_cache_storage(IncompleteModel())
        assert False, "Should raise AttributeError for incomplete model"
    except AttributeError as e:
        assert "num_layers" in str(e) or "num_heads" in str(e), "Error should name missing attribute"


    #testing whether indivisible embed_dim raises ValueError
    class BadModel:
        def __init__(self):
            self.embed_dim = 127
            self.num_layers=2
            self.num_heads = 4
            self.max_seq_len = 32
            self.blocks = [None]*2

    try:
        _create_cache_storage(BadModel())
        assert False, "Should raise ValueError for indivisible dimensions"
    except ValueError as e:
        assert "divisible" in str(e).lower(), "Error should mention divisibility"

    print("_create_cache_storage works correctly")

if __name__ == "__main__":
    test_cache_storage()
