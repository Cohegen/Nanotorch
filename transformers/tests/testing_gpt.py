import os 
import sys


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
import numpy as np
from transformers import LayerNorm ,MLP, TransformerBlock,GPT

def testing_gpt():
    """
    This function intends to test model forward pass,shape consistency,generation capability
    """

    #testing small GPT model
    vocab_size = 100
    embed_dim = 64 
    num_layers = 2
    num_heads = 4

    model =GPT(vocab_size,embed_dim,num_layers,num_heads)

    #test forward pass
    batch_size, seq_len = 2,8 
    tokens = Tensor(np.random.randint(0,vocab_size,(batch_size,seq_len)))
    logits = model.forward(tokens)

    #check output shape
    expected_shape = (batch_size,seq_len,vocab_size)
    assert logits.shape == expected_shape 

    #testing generation
    prompt = Tensor(np.random.randint(0,vocab_size,(1,5)))
    generated = model.generate(prompt,max_new_tokens=3)

    #checking generation shape
    assert generated.shape == (1,8) # 5prompt + 3 new tokens

    #testing parameter counting
    params = model.parameters()
    assert len(params) > 10 #should have many parameters from all components

    #testing with different model sizes
    larger_model = GPT(vocab_size=200,embed_dim=128,num_layers=4,num_heads=8)
    test_tokens =Tensor(np.random.randint(0,200,(1,10)))
    larger_logits = larger_model.forward(test_tokens)
    assert larger_logits.shape == (1,10,200)

    print("GPT model works correclty")

if __name__ == "__main__":
    testing_gpt()