import os
import sys



sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 

from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from attention import _compute_attention_scores,_scale_scores
from attention import MultiHeadAttention
import numpy as np
import math 

def analyze_attention_memory_overhead():
    """
    This function analyzes memory overhead during training 
    """
    embed_dim, num_heads = 128,8
    sequence_lengths = [128,256,512,1024]

    print("\nMemory Overhead Analysis (Training vs Inference)")
    print("Seq len |Forward | + Gradients | + Optimizer | Total Memory")
    print("-"*65)

    for seq_len in sequence_lengths:
        #forward pass memory (attention matrix)
        attention_matrix_mb = (seq_len*seq_len*4) / (1024*1024)

        #backward pass adds gradient storage (2x forward)
        backward_memory_mb = 2*attention_matrix_mb

        #optimizer state (Adam: +2x for momentum and velocity)
        optimizer_memory_mb = backward_memory_mb + 2 *attention_matrix_mb

        #total = forward + gradients + optimizer state 
        total_memory_mb= attention_matrix_mb + backward_memory_mb + optimizer_memory_mb

        print(f"{seq_len:7d} | {attention_matrix_mb:6.2f}MB | {backward_memory_mb:10.2f}MB | {optimizer_memory_mb:10.2f}MB | {total_memory_mb:11.2f}MB")

    print("\nConclusion: Training requires ~7x memory of inference (1x forward + 2x gradients + 4x optimizer state)")
        
if __name__ == "__main__":
    analyze_attention_memory_overhead()