import numpy as np
import time 
from typing import Tuple,Optional,Dict,List
import os 
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Tensor import Tensor 
from memoization import KVCache,_cached_generate,enable_kv_cache,disable_kv_cache


def testing_noninvasive_integration():
   
    # Creating a mock transformer-like object for testing
    class MockTransformerBlock:
        def __init__(self):
            self.attention = self

        def forward(self, x, mask=None):
            # Simple pass-through for testing
            return x

    class MockGPT:
        def __init__(self):
            self.vocab_size = 100
            self.embed_dim = 128
            self.num_layers = 4
            self.num_heads = 4
            self.max_seq_len = 64
            self.blocks = [MockTransformerBlock() for _ in range(self.num_layers)]

    # Test 1: Testing Enable caching
    model = MockGPT()
    print("   Test 1: Enable caching on model")
    cache = enable_kv_cache(model)
    assert hasattr(model, '_kv_cache'), "Model should have _kv_cache attribute"
    assert hasattr(model, '_cache_enabled'), "Model should have _cache_enabled flag"
    assert model._cache_enabled == True, "Cache should be enabled"
    assert cache is model._kv_cache, "Returned cache should match model._kv_cache"

    # Test 2: Testing if Attention forward still works
    print("   Test 2: Attention forward pass still works")
    test_input = Tensor(np.random.randn(1, 10, 128))
    for block in model.blocks:
        output = block.attention.forward(test_input)
        assert output.shape == test_input.shape, "Forward pass should preserve shape"

    # Test 3: Testing Disable caching
    print("   Test 3: Disable caching")
    disable_kv_cache(model)
    assert model._cache_enabled == False, "Cache should be disabled"
    assert not hasattr(model, '_kv_cache'), "Cache object should be removed"

    # Test 4: Can re-enable
    print("   Test 4: Re-enable caching")
    _ = enable_kv_cache(model)
    assert model._cache_enabled == True, "Cache should be re-enabled"

    print(" Non-invasive cache integration works correctly!")

# Run test immediately when developing this module
if __name__ == "__main__":
    testing_noninvasive_integration()
