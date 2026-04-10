import os 

import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from Tensor import Tensor 
import numpy as np
from transformers import LayerNorm ,MLP, TransformerBlock,GPT

def testing_module():
    """
    This function intends to test the transformer module
    to make sure all functionalities works correctly.
    """

    print("=" * 50)

    #create model and data 
    vocab_size = 50
    embed_dim = 64 
    num_layers = 2
    num_heads = 4

    model = GPT(vocab_size,embed_dim,num_layers,num_heads)

    #testing batch processing
    batch_size = 3 
    seq_len = 16
    tokens = Tensor(np.random.randint(0,vocab_size,(batch_size,seq_len)))

    #forward pass 
    logits = model.forward(tokens)
    assert logits.shape == (batch_size,seq_len,vocab_size)

    #test generation with different temperatures
    prompt = Tensor(np.random.randint(0,vocab_size,(1,8)))

    #conservative generation
    conservative = model.generate(prompt,max_new_tokens=5,temperature=0.1)
    assert conservative.shape == (1,13)

    #creative generation
    creative = model.generate(prompt,max_new_tokens=5,temperature=2.0)
    assert creative.shape == (1,13)

    #testing parameter counting consistency
    total_params = sum(param.size for param in model.parameters())
    assert total_params > 1000 # should have substantial parameters

    print("Full transformer pipeline works!")

    print("\n" + "=" * 50)

if __name__ == "__main__":
    testing_module()