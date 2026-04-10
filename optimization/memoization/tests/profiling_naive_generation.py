import numpy as np
import time 
from typing import Tuple,Optional,Dict,List
import os 
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Tensor import Tensor 
from optimization.profiling.profiling import Profiler

def profile_naive_generation():
    """
    This function profiles the transformer generation to discover the O(n**2) bottleneck.

    It intends to demonstrate why KV caching is necessary by showing concrete measurements of quadratic
    growth in generation time
    """
    profiler = Profiler()

    def naive_attention_step(seq_len,hidden_dim=64):
        """
        Simulates one steps of attention computation
        Without caching this process ALL prevois token everytime.

        """
        #Q,K,V for entire sequence
        q = Tensor(np.random.randn(1,seq_len,hidden_dim))
        k = Tensor(np.random.randn(1, seq_len, hidden_dim))
        v = Tensor(np.random.randn(1, seq_len, hidden_dim))

        #Attention:Q @K.T then @ V
        #this is O(seq_len**2) in complexity
        scores = q @k.T 
        output = scores @ v

        return output 

    #profiling at increasing sequence lengths
    print("Profiling Transformer Generation (Without Caching):\n")
    print("   Seq Len  |  Latency (ms)  |  Growth")
    print("   ---------|----------------|----------")

    sequence_lengths = [10,20,40,80,160]
    latencies = []

    for seq_len in sequence_lengths:
        #measuring latency for this sequence length
        latency = profiler.measure_latency(
            lambda _:naive_attention_step(seq_len),
            None,
            warmup=5,
            iterations=20
        )
        latencies.append(latency)

        #calculating growth rate 
        if len(latencies) > 1:
            growth = latencies[-1] / latencies[-2]
            print(f"   {seq_len:3d}      |  {latency:6.2f}        |  {growth:.2f}×")
        else:
            print(f"   {seq_len:3d}      |  {latency:6.2f}        |  baseline")


if __name__ == "__main__":
    profile_naive_generation()


