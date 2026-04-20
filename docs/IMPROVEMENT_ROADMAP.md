# NanoTorch Improvement Roadmap

This repo hardening pass is organized from major issues to minor ones so future work stays aligned with framework reliability first.

## 1. Core Reliability

- keep `nanotorch` as the recommended package entrypoint
- keep behavioral fixes in the original implementation modules
- strengthen tensor, autograd, optimizer, and trainer validation
- fail early on non-finite loss, gradients, and parameters

Implemented in this pass:

- validation helpers in `nanotorch.utils.validation`
- reproducibility helpers in `nanotorch.utils.reproducibility`
- stricter `Trainer` checks for non-finite values

## 2. Environment Reliability

- standardize installation through `pyproject.toml`
- keep minimal runtime and dev dependency files
- provide a smoke test after setup changes
- avoid relying on stale local venv launchers

Implemented in this pass:

- `pyproject.toml`
- `requirements.txt`
- `requirements-dev.txt`
- `pytest.ini`
- `scripts/smoke_test.py`

## 3. Testing and Validation

- keep tests focused on behavior, not only import layout
- add regression coverage for new utility helpers
- expand trainer tests for validation behavior

Implemented in this pass:

- `tests/test_utils.py`
- expanded `tests/test_training.py`

## 4. Shared Project Infrastructure

Recommended next:

- central experiment logging helpers
- shared metrics and plotting utilities
- common dataset transform utilities
- checkpoint resume examples in `projects/`

## 5. Performance and Model Breadth

Recommended after the reliability layers are stable:

- more vectorized tensor and convolution paths
- more efficient attention implementations
- self-supervised learning helpers for DINO, BYOL, MAE, and SimCLR
- stronger generative model infrastructure around VAE, UNet, and diffusion
