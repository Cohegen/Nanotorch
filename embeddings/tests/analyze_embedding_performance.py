import os
import sys


 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 

from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from embeddings import Embedding,PositionalEncoding,_compute_sinusoidal_table,create_sinusoidal_embeddings
from embeddings import EmbeddingLayer
import numpy as np
import math 

"""
-This program intends to give Performance Insights
- After running we will make the conclusion below:
1.Lookup time is 0(1)  per token, the vocabulary size doesn't affect individual lookups
2.Larger batches improve throughput due to vectorization
3.Memory bandwidth becomes bottleneck fot large embedding dimensions
4.Cache locality is important for repeated token patterns
"""

def analyze_embedding_performance():
    """
    Comparing embedding lookup performance across different
    configurations
    """
    print("\nAnalyzing Embedding Lookup Performance...")
    print("="*60)

    import time 

    #testing differnet vocabulary sizes and batch configurations
    vocab_sizes = [1_000,10_000,100_000]
    embed_dim = 512
    seq_len = 128 
    batch_sizes = [1,16,64,256]

    print(f"{'Vocab Size':<12} {'Batch Size':<12} {'Lookup Time (ms)':<18} {'Throughput (tokens/s)':<20}")
    print("-" * 70)

    for vocab_size in vocab_sizes:
        #create embedding layer
        embed = Embedding(vocab_size,embed_dim)

        for batch_size in batch_sizes:
            #create random token batch 
            tokens = Tensor(np.random.randint(0,vocab_size,(batch_size,seq_len)))

            #warmup
            for _ in range(5):
                _ = embed.forward(tokens)

            #time the lookup
            start_time = time.time()
            iterations = 100

            for _ in range(iterations):
                output = embed.forward(tokens)

            end_time = time.time()

            #Calculating metrics 
            total_time = end_time - start_time 
            avg_time_ms = (total_time/iterations)*1000
            total_tokens = batch_size * seq_len * iterations 
            throughput = total_tokens /total_time

            print(f"{vocab_size:<12,} {batch_size:<12} {avg_time_ms:<18.2f} {throughput:<20,.0f}")

if __name__ == "__main__":
    analyze_embedding_performance()