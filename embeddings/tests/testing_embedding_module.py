import os
import sys


 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 

from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from embeddings import Embedding,PositionalEncoding,_compute_sinusoidal_table,create_sinusoidal_embeddings
from embeddings import EmbeddingLayer
import numpy as np
import math 

def testing_embedding_module():
    """
    This function tests the full module
    """
    #simulating a small transformer setup
    vocab_size = 1000
    embed_dim = 128 
    max_seq_len = 64 

    #creating embedding layer 
    embed_layer = EmbeddingLayer(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        pos_encoding='learned',
        scale_embeddings=True 
    )

    #simulating tokenized sentences
    tokenized_sentences = [
        [1,15,42,7,99], #the cat sat on mat
        [23,7,15,88],#dog chased the ball
        [1,67,15,42,7,99,34]#the big cat sat on mat here
    ]

    #process each sentence
    outputs = []
    for sentence in tokenized_sentences:
        tokens = Tensor(sentence)
        embedded = embed_layer.forward(tokens)
        outputs.append(embedded)

        #verifying output shape 
        expected_shape = (len(sentence),embed_dim)
        assert embedded.shape == expected_shape,f"Wrong shape for sentence: {embedded.shape} != {expected_shape}"

        print("Variable length sentence processing works")

    #testing batch processinng with padding
    #creating padded batch 
    max_len = max(len(s) for s in tokenized_sentences)
    batch_tokens = []

    for sentence in tokenized_sentences:
        #pad with zeros i.e assuming 0 is padding token
        padded = sentence + [0] * (max_len - len(sentence))
        batch_tokens.append(padded)

    batch_tensor = Tensor(batch_tokens)
    batch_output = embed_layer.forward(batch_tensor)

    assert batch_output.shape == (len(tokenized_sentences), max_len, embed_dim), f"Batch output shape incorrect:{batch_output.shape}"
    print("Batch processing with padding works")

    #testing different encoding type
    test_tokens = Tensor([[1,2,3,4,5]])

    #test all position encoding types 
    for pe_type in ['learned','sinusoidal',None]:
        embed_test = EmbeddingLayer(
            vocab_size=100,
            embed_dim=64,
            pos_encoding=pe_type 
        )
        
        output = embed_test.forward(test_tokens)
        assert output.shape == (1,5,64),f"PE type {pe_type} failed shape test"

        #checking parameter counts 
        if pe_type == 'learned':
            assert len(embed_test.parameters()) == 2,f"Learnable PE should have 2 param tensors"
        else:
            assert len(embed_test.parameters()) == 1, f"PE type {pe_type} should have 1 param tensor"

        print("All positional encoding variants work")

        #analysis of memory efficiency check
        #testing whether we're not creating unnecessary copies
        large_embed = EmbeddingLayer(vocab_size=10000,embed_dim=512)
        test_batch = Tensor(np.random.randint(0,10000,(32,128)))

        #multiple forward passes should not accumulate memory
        for _ in range(5):
            output = large_embed.forward(test_batch)
            assert output.shape == (32,128,512), "Large batch processing failed"

        print("Memory efficiently check passed")
        print("\n"+ "="*50)

    
if __name__ == "__main__":
    testing_embedding_module()

