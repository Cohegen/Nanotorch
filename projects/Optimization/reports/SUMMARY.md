# Optimization Reports Summary

## compression_profiling_pipeline

| Metric | Baseline | Compressed | Delta/Ratio |
| --- | --- | --- | --- |
| Accuracy | 0.98 | 0.785 | -0.19499999999999995 |
| Latency (ms) | 149.73580000150832 | 180.18914999993285 | 30.453349998424528 |
| Memory/Size | 0.1515350341796875 | 0.6115932464599609 | - |

![compression_profiling_pipeline.json](plots\compression_profiling_pipeline.png)

---

## kv_cache_benchmark

- Naive Latency: 36.52 ms
- Cached Latency: 30.03 ms
- Speedup: 1.22x

![kv_cache_benchmark.json](plots\kv_cache_benchmark.png)

---

## quantized_nanodigits

| Metric | Baseline | Quantized | Delta/Ratio |
| --- | --- | --- | --- |
| Accuracy | N/A | N/A | 0.0 |
| Latency (ms) | 38.953649998802575 | 58.84999999761931 | N/A |
| Memory/Size | 68904 | 17250 | - |

![quantized_nanodigits.json](plots\quantized_nanodigits.png)

---

