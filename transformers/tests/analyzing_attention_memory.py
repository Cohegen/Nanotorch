import os 
import sys


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from Tensor import Tensor 
import numpy as np
from transformers import LayerNorm ,MLP, TransformerBlock,GPT

def analyze_attention_memory():
    """
    This function attention memory complexity with sequence length
    """

    embed_dim = 512
    num_heads = 8
    batch_size = 4

    #testing different sequences lengths
    sequence_lengths = [128,256,512,1024,2048]

    print("Attention Matrix Memory Usage:")
    print("Seq Len | Attention Matrix Size | Memory (MB)")
    print("-" * 45)

    for seq_len in sequence_lengths:
        #attention matrix is (batch_szie,num_heads,seq_len,seq_len)
        attention_elements = batch_size * num_heads * seq_len * seq_len

        #4 bytes per float32
        memory_bytes = attention_elements * BYTES_PER_FLOAT32
        memory_mb = memory_bytes / MB_TO_BYTES

        print(f"{seq_len:6d} | {seq_len}×{seq_len} × {batch_size}×{num_heads} | {memory_mb:8.1f}")

    print()
    print("\nConclusion: Attention memory grows quadratically with sequence length:")

if __name__ == "__main__":
    analyze_attention_memory()

