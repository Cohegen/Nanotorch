import os
import sys



sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 

from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from attention import _compute_attention_scores,_scale_scores
from attention import MultiHeadAttention
import numpy as np
import math 

def analyze_attention_complexity():
    """
    This function analyzes attention computational complexity and memory scaling
    """

    #testing different sequence lengths to show O(n**2) scaling 
    embed_dim = 64
    sequence_lengths = [16,32,64,128,256]

    print("\nSequence Length vs Attention Matrix Size:")
    print("Seq Len | Attention Matrix | Memory (KB) | Complexity")
    print("-" * 55)

    for seq_len in sequence_lengths:
        #calculate attention matrix size
        attention_matrix_size = seq_len * seq_len

        #memory for attention weights (float32=4bytes)
        attention_memory_kb = (attention_matrix_size*4) / 1024

        #total complexity (Q@K + softmax + weights@V)
        complexity = 2 * seq_len *embed_dim + seq_len * seq_len

        print(f"{seq_len:7d} | {attention_matrix_size:14d} | {attention_memory_kb:10.2f} | {complexity:10.0f}")

    print(f"\nConclusion:Attention memory scales as O(n^2) with sequence length")
    print(f"For seq_len=1024,attention matrix alone needs {(1024*1024*4)/1024/1024:.1f} MB")

if __name__ == "__main__":
    analyze_attention_complexity()
