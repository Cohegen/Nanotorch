import os 
import sys


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
import numpy as np
from transformers import LayerNorm ,MLP, TransformerBlock,GPT


def testing_next_token():
    """
    this function intends to test temperature scaling,softmax probability output,
    valid token rage
    """
    model = GPT(vocab_size=5,embed_dim=32,num_layers=1,num_heads=2)

    #testing whether output is a vald token index
    logits = np.array([[1.0,2.0,3.0,4.0,5.0]])
    token = model._sample_next_token(logits,temperature=1.0)
    assert isinstance(token,(int,np.integer)), f"Exected int, got {type(token)}"
    assert 0 <= token < 5, f"Token {token} out of range [0,5]"

    #testing whether very low temperature always picks the highest logit
    np.random.seed(42)
    high_logit_idx = 4 #logits[4] = 5.0 is highest
    low_temp_tokens = [model._sample_next_token(logits,temperature=0.01) for _ in range(20)]
    assert all(t == high_logit_idx for t in low_temp_tokens),(
        f"Low temperature should consistently pick token {high_logit_idx}, got {low_temp_tokens}"
    )

    #verifying softmax math
    #with logits [0,0,0,0,10], softmax should heavily favor index 4
    extreme_logits = np.array([[0.0,0.0,0.0,0.0,10.0]])
    extreme_tokens = [model._sample_next_token(extreme_logits,temperature=1.0) for _ in range(20)]
    assert all(t == 4 for t in extreme_tokens),(
        f"Extreme logits should always pick token 4, got {extreme_tokens}"
    )

    #testing whether high temperature produces more varied tokens
    np.random.seed(0)
    uniform_logits = np.array([[1.0,1.0,1.0,1.0,1.0]])
    high_temp_tokens = set(model._sample_next_token(uniform_logits,temperature=2.0) for _ in range(50))
    assert len(high_temp_tokens) > 1, "High temperature with uniform logits should produce varied tokens"

    print("Token sampling works correctly")


if __name__ == "__main__":
    testing_next_token()