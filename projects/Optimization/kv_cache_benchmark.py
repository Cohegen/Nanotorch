import sys
import time
from pathlib import Path

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Tensor import Tensor
from layers.layers import Linear
from optimization.memoization.memoization import KVCache, _cached_generation_step
from common import save_json, seed_everything


class ToySelfAttention:
    def __init__(self, embed_dim=64, num_heads=4):
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)


def naive_generation_step(prefix, attention):
    batch_size, seq_len, embed_dim = prefix.shape
    num_heads = attention.num_heads
    head_dim = attention.head_dim

    q = attention.q_proj.forward(prefix)
    k = attention.k_proj.forward(prefix)
    v = attention.v_proj.forward(prefix)

    q_heads = q.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k_heads = k.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    v_heads = v.reshape(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

    q_last = q_heads[:, :, -1:, :]
    scores = np.matmul(q_last.data, np.transpose(k_heads.data, (0, 1, 3, 2))) / np.sqrt(head_dim)
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    attended = np.matmul(weights, v_heads.data)
    attended = np.transpose(attended, (0, 2, 1, 3)).reshape(batch_size, 1, embed_dim)
    return attention.out_proj.forward(Tensor(attended))


def benchmark_generation(batch_size=1, seq_len=48, embed_dim=64, num_heads=4, iterations=25, seed=7):
    seed_everything(seed)
    attention = ToySelfAttention(embed_dim=embed_dim, num_heads=num_heads)
    sequence = Tensor(np.random.randn(batch_size, seq_len, embed_dim).astype(np.float32))

    naive_outputs = []
    for step in range(seq_len):
        prefix = Tensor(sequence.data[:, : step + 1, :])
        naive_outputs.append(naive_generation_step(prefix, attention))

    cache = KVCache(
        batch_size=batch_size,
        max_seq_len=seq_len,
        num_layers=1,
        num_heads=num_heads,
        head_dim=embed_dim // num_heads,
    )
    cached_outputs = []
    for step in range(seq_len):
        token = Tensor(sequence.data[:, step : step + 1, :])
        cached_outputs.append(_cached_generation_step(token, attention, cache, layer_idx=0))
        cache.advance()

    max_abs_diff = max(
        float(np.max(np.abs(naive_outputs[i].data - cached_outputs[i].data))) for i in range(seq_len)
    )

    naive_times = []
    cached_times = []
    for _ in range(iterations):
        start = time.perf_counter()
        for step in range(seq_len):
            prefix = Tensor(sequence.data[:, : step + 1, :])
            _ = naive_generation_step(prefix, attention)
        naive_times.append((time.perf_counter() - start) * 1000.0)

        cache = KVCache(
            batch_size=batch_size,
            max_seq_len=seq_len,
            num_layers=1,
            num_heads=num_heads,
            head_dim=embed_dim // num_heads,
        )
        start = time.perf_counter()
        for step in range(seq_len):
            token = Tensor(sequence.data[:, step : step + 1, :])
            _ = _cached_generation_step(token, attention, cache, layer_idx=0)
            cache.advance()
        cached_times.append((time.perf_counter() - start) * 1000.0)

    naive_latency = float(np.median(np.array(naive_times)))
    cached_latency = float(np.median(np.array(cached_times)))
    speedup = naive_latency / max(cached_latency, 1e-9)

    results = {
        "project": "kv_cache_benchmark",
        "batch_size": batch_size,
        "sequence_length": seq_len,
        "embed_dim": embed_dim,
        "num_heads": num_heads,
        "iterations": iterations,
        "naive_latency_ms": naive_latency,
        "cached_latency_ms": cached_latency,
        "speedup_x": speedup,
        "max_abs_output_diff": max_abs_diff,
        "cache_memory": cache.get_memory_usage(),
    }
    return results


def main():
    project_dir = Path(__file__).resolve().parent
    output_path = project_dir / "reports" / "kv_cache_benchmark.json"
    results = benchmark_generation()
    save_json(output_path, results)

    print("KV cache benchmark complete")
    print(f"Naive latency: {results['naive_latency_ms']:.3f} ms")
    print(f"Cached latency: {results['cached_latency_ms']:.3f} ms")
    print(f"Speedup: {results['speedup_x']:.2f}x")
    print(f"Max output diff: {results['max_abs_output_diff']:.6f}")
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    main()
