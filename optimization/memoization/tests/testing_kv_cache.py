import numpy as np
import time 
from typing import Tuple,Optional,Dict,List
import os 
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Tensor import Tensor 
from memoization import KVCache

def testing_kvcache():
    """
    This functions validates whether the KVCache Implementation works correctly
    """
    #testing parameters 
    batch_size,max_seq_len = 2,8 
    num_layers,num_heads,head_dim = 3,4,16

    #creating cache
    cache = KVCache(batch_size,max_seq_len,num_layers,num_heads,head_dim)

    #testing Intial state
    assert cache.seq_pos == 0, "Cache should start at position 0"
    mem_usage = cache.get_memory_usage()
    assert mem_usage['total_mb'] > 0, "Cache should have non-zero memory usage"
    print(f"   Cache initialized: {mem_usage['total_mb']:.2f} MB")

    #testing single token update and retrieval
    key1 = Tensor(np.random.randn(batch_size, num_heads, 1, head_dim))
    value1 = Tensor(np.random.randn(batch_size, num_heads, 1, head_dim))

    #updating layer 0with first token
    cache.update(0,key1,value1)

    #before advance, get() should return empty(seq_pos=0)
    cached_k,cached_v = cache.get(0)
    assert cached_k.shape == (batch_size, num_heads, 0, head_dim), "Before advance, cache should be empty"

    #advance position
    cache.advance()

    #validating whether cache has 1 token
    cached_k,cached_v = cache.get(0)
    assert cached_k.shape == (batch_size, num_heads, 1, head_dim), f"Expected shape (2,4,1,16), got {cached_k.shape}"
    assert cached_v.shape == (batch_size, num_heads, 1, head_dim), f"Expected shape (2,4,1,16), got {cached_v.shape}"

    #testing Multi-token sequence
    key2 = Tensor(np.random.randn(batch_size,num_heads,1,head_dim))
    value2 = Tensor(np.random.randn(batch_size,num_heads,1,head_dim))
    cache.update(0,key2,value2)
    cache.advance()

    cached_k, cached_v = cache.get(0)
    assert cached_k.shape == (batch_size, num_heads, 2, head_dim), "Should have 2 tokens cached"
    assert cached_v.shape == (batch_size, num_heads, 2, head_dim), "Should have 2 tokens cached"

    #testing Multiple layers
    cache.reset()
    key_test = Tensor(np.random.randn(batch_size,num_heads,1,head_dim))
    value_test = Tensor(np.random.randn(batch_size,num_heads,1,head_dim))

    #update all layers with same token
    cache.update(0, key_test, value_test)  # Layer 0
    cache.update(1, key_test, value_test)  # Layer 1
    cache.update(2, key_test, value_test)  # Layer 2
    cache.advance()

    #each layer should have the cached token
    for layer_idx in range(num_layers):
        cached_k,cached_v = cache.get(layer_idx)
        assert cached_k.shape[2] == 1, f"Layer {layer_idx} should have 1 token"

    #testing Reset functionality
    cache.reset()
    assert cache.seq_pos == 0, "Reset should clear sequence position"
    cached_k, cached_v = cache.get(0)
    assert cached_k.shape == (batch_size, num_heads, 0, head_dim), "Reset should clear cache"

    print(" KVCache implementation works correctly!")

if __name__ == "__main__":
    testing_kvcache()