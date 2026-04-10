import os
import sys


 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 

from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from embeddings import Embedding,PositionalEncoding,_compute_sinusoidal_table,create_sinusoidal_embeddings
from embeddings import EmbeddingLayer
import numpy as np
import math 

def analyze_embedding_memory_scaling():
    """
    This function compares memory requirements across 
    different model scales
    """
    print("Analysis of Embedding Memory Requirements..")
    print("="*60)

    #vocabulary and embedding dimension scenarios
    scenarios = [
        ("Small Model",10_000,256),
        ("Medium Model",50_000,512),
        ("Large Model",100_000,1024),
        ("GPT-3 Scale",50_257,12_288),

    ]

    print(f"{'Model':<15} {'Vocab Size':<12} {'Embed Dim':<12} {'Memory (MB)':<15} {'Parameters (M)':<15}")
    print("-" * 80)

    for name,vocab_size,embed_dim in scenarios:
        #calculate memory for FP32 (4 bytes per parameter)
        params = vocab_size * embed_dim
        memory_mb = params * BYTES_PER_FLOAT32 /MB_TO_BYTES
        params_m = params / 1_000_000

        print(f"{name:<15} {vocab_size:<12,} {embed_dim:<12} {memory_mb:<15.1f} {params_m:<15.2f}")

    print("\n Key Insights:")
    print("• Embedding tables often dominate model memory (especially for large vocabularies)")
    print("• Memory scales linearly with vocab_size × embed_dim")
    print("• Consider vocabulary pruning for memory-constrained environments")

    # Positional encoding memory comparison
    print(f"\n Positional Encoding Memory Comparison (embed_dim=512, max_seq_len=2048):")

    learned_params = 2048 * 512
    learned_memory = learned_params * 4 / (1024 * 1024)

    print(f"Learned PE:     {learned_memory:.1f} MB ({learned_params:,} parameters)")
    print(f"Sinusoidal PE:  0.0 MB (0 parameters - computed on-the-fly)")
    print(f"No PE:          0.0 MB (0 parameters)")

    print("\n Production Implications:")
    print("• GPT-3's embedding table: ~2.4GB (50K vocab × 12K dims)")
    print("• Learned PE adds memory but may improve task-specific performance")
    print("• Sinusoidal PE saves memory and allows longer sequences")

if __name__ == "__main__":
    analyze_embedding_memory_scaling()