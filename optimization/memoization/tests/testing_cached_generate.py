import numpy as np
import time 
from typing import Tuple,Optional,Dict,List
import os 
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Tensor import Tensor 
from memoization import KVCache,_cached_generate

def testing_cached_generated():
    """
    This function intends to test whether the _cached_generate
    function works
    """

    vocab_size =50

    #creating a minimal mock model that returns random logits
    class MockModel:
        def __init__(self):
            self.embed_dim = 64
            self.num_layers = 1
            self.num_heads = 2
            self.max_seq_len = 128
            self.blocks = []

        def forward(self,x):
            #return random logits shaped (batch,seq_len,vocab_size)
            batch_size = x.shape[0]
            seq_len =x.shape[1]
            return Tensor(np.random.randn(batch_size,seq_len,vocab_size))

    model = MockModel()

    #creating cache
    cache = KVCache(batch_size=1,max_seq_len=128,num_layers=1,num_heads=2,head_dim=32)

     # Testing Generation of correct number of tokens
    prompt = [0, 1, 2]
    max_new = 5
    generated = _cached_generate(model, prompt, max_new, temperature=1.0, cache=cache)
    assert len(generated) == max_new, f"Expected {max_new} tokens, got {len(generated)}"

     # Testing if  All tokens are valid indices
    for token in generated:
        assert 0 <= token < vocab_size, f"Token {token} out of vocab range [0, {vocab_size})"
    
    #Testing whether Cache position advanced correctly
    # prompt (3 tokens) + generated (5 tokens) = 8 advances
    expected_pos = len(prompt) + max_new
    assert cache.seq_pos == expected_pos, f"Expected cache pos={expected_pos}, got {cache.seq_pos}"

     # Test 4: Generate with low temperature (more deterministic)
    cache.reset()
    generated_low_temp = _cached_generate(model, [0], 3, temperature=0.01, cache=cache)
    assert len(generated_low_temp) == 3, "Should generate 3 tokens with low temperature"

    print("_cached_generate works correctly")

if __name__ == "__main__":
    testing_cached_generated()


