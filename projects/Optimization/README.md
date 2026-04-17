# Optimization Projects

This folder contains three runnable project scripts built on top of the repo's `optimization` utilities.

## Included projects

- `kv_cache_benchmark.py`
  Compares naive autoregressive attention against `KVCache`-based cached generation and writes a latency report.

- `quantized_nanodigits.py`
  Trains a small NanoDigits MLP, quantizes its linear layers with the quantization module, and compares accuracy, latency, and estimated model size.

- `compression_profiling_pipeline.py`
  Trains the same NanoDigits MLP, applies compression techniques, profiles the model before and after compression, and records the tradeoffs.

## Reports

Each script writes a JSON report under:

- `projects/Optimization/reports/kv_cache_benchmark.json`
- `projects/Optimization/reports/quantized_nanodigits.json`
- `projects/Optimization/reports/compression_profiling_pipeline.json`

## Notes

- These projects use the local NanoDigits dataset already present in the repo.
- The quantization and compression scripts are intentionally built around linear layers because the current optimization utilities operate on `Sequential`/`Linear` patterns most directly.
