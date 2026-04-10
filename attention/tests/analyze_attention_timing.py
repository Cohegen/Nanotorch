import os
import sys



sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 

from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from attention import _compute_attention_scores,_scale_scores
from attention import MultiHeadAttention
import numpy as np
import math 

def analyze_attention_timing():
    """
    This function measures attention computation time vs sequence length.
    """

    embed_dim,num_heads = 64,8
    sequence_lengths = [32,64,128,256]

    print("\nSequence length vs Computation Time:")
    print("Seq Len | Time(ms) | Ops/sec| Scaling")
    print("-"*40)

    import time

    prev_time = None
    for seq_len in sequence_lengths:
        #creating test input
        x = Tensor(np.random.randn(1,seq_len,embed_dim))
        mha = MultiHeadAttention(embed_dim,num_heads)

        #time multiple runs for stability
        times = []
        for _ in range(5):
            start_time = time.time()
            _ = mha.forward(x)
            end_time = time.time()
            times.append((end_time-start_time)*1000) #convert to ms

        avg_time = np.mean(times)
        ops_per_sec = 1000 /avg_time if avg_time >0 else 0

        #calculating scaling factor vs previous 
        scaling = avg_time / prev_time if prev_time else 1.0

        print(f"{seq_len:7d} | {avg_time:8.2f} | {ops_per_sec:7.0f} | {scaling:6.2f}x")
        prev_time = avg_time

    print(f"\nConclusion: Attention time scales roughly as O(n^2) with sequence length")
    

if __name__ == "__main__":
    analyze_attention_timing()