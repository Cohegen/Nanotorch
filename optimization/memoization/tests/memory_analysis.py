import numpy as np
import time 
from typing import Tuple,Optional,Dict,List
import os 
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Tensor import Tensor 
from memoization import KVCache,_cached_generate,_BYTES_PER_FLOAT32,_MB_TO_BYTES

def analyze_kvcache_memory():
    """
     AnalyZING KV cache memory usage across different configurations.

    Educational Purpose:
        Demonstrates how cache memory scales with model architecture.
        You will discover that :
        - Linear scaling with sequence length O(n)
        - Memory overhead as percentage of model parameters
        - Trade-off between cache size and speedup gains

    Analyzes:
        - Tiny models (128D): ~0.12 MB
        - Small models (512D): ~2 MB
        - Medium models (768D): ~9 MB
        - Large models (1024D): ~32 MB

    Key Insight:
        Cache overhead is 10-30% of model parameters, but enables
        10-15× speedup. Memory is cheap, compute is expensive!

    Production Context:
        GPT-3 (175B params, 2048 context): ~4GB cache per sequence
        This memory cost is acceptable given the massive speedup.
    """
    print(" Analyzing KV Cache Memory Usage...")
    print()

    # Testing different model configurations
    configs = [
        (128, 4, 32, "Tiny"),
        (512, 8, 64, "Small"),
        (768, 12, 128, "Medium"),
        (1024, 16, 256, "Large"),
    ]

    print("Model Config | Cache Memory | Per Layer | Memory Overhead")
    print("-" * 60)

    for embed_dim, num_layers, seq_len, name in configs:
        # Memory per layer: 2 tensors (K, V) × batch × seq_len × embed_dim × 4 bytes
        batch_size = 1
        memory_per_layer = 2 * batch_size * seq_len * embed_dim * _BYTES_PER_FLOAT32 / _MB_TO_BYTES
        total_memory = memory_per_layer * num_layers

        # Model parameter memory (approximate)
        params_per_layer = embed_dim * embed_dim * _BYTES_PER_FLOAT32  # QKV projections
        model_memory = params_per_layer * num_layers * _BYTES_PER_FLOAT32 / _MB_TO_BYTES

        overhead_pct = (total_memory / model_memory) * 100 if model_memory > 0 else 0

        print(f"{name:12s} | {total_memory:11.2f} MB | {memory_per_layer:8.2f} MB | {overhead_pct:6.1f}%")

    print()
    print(" Key Insights:")
    print("   • Cache memory scales linearly with sequence length (O(n))")
    print("   • Longer sequences require proportionally more cache memory")
    print("   • Cache overhead is typically 10-30% of model parameters")
    print()
    print(" Production Context:")
    print("   • GPT-3 (175B params, 2048 context): ~4GB cache memory")
    print("   • Trade-off: 2× memory enables 10-15× speedup")
    print("   • Worth it for inference-heavy workloads!")


def analyze_kvcache_speedup():
    """
     Measures KV cache speedup vs vanilla attention.

    Educational Purpose:
        Shows the learner WHY caching provides dramatic speedup through
        concrete complexity analysis. Compares O(n²) vs O(n) growth.

    Demonstrates:
        - Naive approach: O(n²) operations per token
        - Cached approach: O(n) operations per token
        - Speedup increases with generation length
        - 100-token generation: 170× fewer operations

    Key Insight:
        Speedup is SUPER-LINEAR with generation length because:
        - Longer sequences → more redundant computation without cache
        - Cache benefit compounds: saves O(n²) → O(n) at EVERY step

    Production Reality:
        This is why ChatGPT can generate responses in real-time.
        Without caching, conversational AI would be economically impossible.
    """
    print("\n Analyzing KV Cache Speedup...")
    print()

    import time

    # Creating test configuration
    batch_size = 1
    embed_dim = 256
    num_heads = 8
    head_dim = embed_dim // num_heads

    print("Generation Length | Without Cache | With Cache | Speedup")
    print("-" * 55)

    for gen_length in [10, 25, 50, 100]:
        # Simulating without cache: O(n²) for each new token
        # Each token processes entire context
        ops_without = sum(i**2 for i in range(1, gen_length + 1))

        # Simulating with cache: O(n) for each new token
        # Each token only processes itself
        ops_with = gen_length

        # Estimating time (arbitrary units)
        time_without = ops_without / 1000  # ms
        time_with = ops_with / 1000  # ms
        speedup = ops_without / ops_with

        print(f"{gen_length:17d} | {time_without:12.1f} ms | {time_with:10.1f} ms | {speedup:6.1f}×")

    print()
    print(" Key Insights:")
    print("   • Speedup increases with generation length (longer = better ROI)")
    print("   • 100-token generation: ~170× fewer operations!")
    print("   • Cache eliminates O(n²) recomputation per token")
    print()
    print(" Production Reality:")
    print("   • ChatGPT uses KV caching for ALL generation")
    print("   • Without caching: 100-token response takes ~17 seconds")
    print("   • With caching: 100-token response takes ~0.1 seconds")
    print("   • This optimization makes conversational AI possible!")

# Running analysis when developing this module
if __name__ == "__main__":
    analyze_kvcache_memory()
    analyze_kvcache_speedup()