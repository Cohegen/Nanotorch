# NanoGPT2 In NanoTorch

This directory contains a CPU-friendly GPT-2 style decoder-only language model implemented against NanoTorch primitives.

Files:
- `nanogpt2_nanotorch.py`: trains a GPT-2 style model on `datasets/nanodigits/corpus.txt` using the repo BPE tokenizer
- `benchmark_nanogpt2.py`: runs a few small benchmark configurations and saves stats in `projects/Transformers/artifacts`

Notes:
- The current repo dataset file at `datasets/nanodigits/corpus.txt` is empty, so the scripts can optionally fall back to a tiny built-in demo corpus.
- Pass `--no-fallback` if you want the training run to fail instead of using the fallback text.
- Training artifacts are written to `projects/Transformers/artifacts`:
  - `nanogpt2_train_stats.json`
  - `nanogpt2_training_curve.csv`
  - `nanogpt2_results_table.md`
  - `nanogpt2_loss_curve.png`
  - `nanogpt2_training_stats.png`

Examples:

```powershell
python projects\Transformers\nanogpt2_nanotorch.py
python projects\Transformers\nanogpt2_nanotorch.py --max-steps 80 --bpe-vocab-size 48
python projects\Transformers\benchmark_nanogpt2.py
```
