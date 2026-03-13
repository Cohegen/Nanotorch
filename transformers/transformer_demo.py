import os 
import sys


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
import numpy as np
from transformers import LayerNorm ,MLP, TransformerBlock,GPT

def transformer_demo():
    """
    This demo follows the traditional transformer model
    process i.e:
    Text input -> model processing-> text generation
    """

    #creates a small vocabulary (character-level)
    vocab = list("abcdefghijklmnopqrstuvwxyz .")
    vocab_size = len(vocab)
    char_to_idx = {char:i for i,char in enumerate(vocab)}
    idx_to_char = {i:char for i,char in enumerate(vocab)}

    print(f"Vocabulary size: {vocab_size}")
    print(f"Characters: {''.join(vocab)}")

    #create model
    model = GPT(
        vocab_size = vocab_size,
        embed_dim=64,
        num_layers=2,
        num_heads=4,
        max_seq_len=32
    )

    #sample text encoding
    text = "hello world."
    tokens = [char_to_idx[char] for char in text]
    input_tokens = Tensor(np.array([tokens]))

    print(f"\nOriginal text: '{text}'")
    print(f"Tokenized: {tokens}")
    print(f"Input shape: {input_tokens.shape}")

    #forward pass 
    logits = model.forward(input_tokens)
    print(f"Output logits shape: {logits.shape}")
    print(f"Each position predicts nex token from {vocab_size} possibilities")

    #generation demo
    prompt_text = "hello"
    prompt_tokens = [char_to_idx[char] for char in prompt_text]
    prompt = Tensor(np.array([prompt_tokens]))

    print("\nGeneration demo:")
    print(f"Prompt: '{prompt_text}'")

    generated = model.generate(prompt,max_new_tokens=8,temperature=1.0)
    generated_text = ''.join([idx_to_char[idx] for idx in generated.data[0]])

    print(f"Generated: '{generated_text}'")
    print("(Note: Untrained model produces random text)")

    return model 


if __name__ == "__main__":
    transformer_demo()