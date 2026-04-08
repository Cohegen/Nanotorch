# NanoTorchVision

`nanotorchvision/` is the vision-model companion package for NanoTorch.

It is intended to hold:
- small image datasets and dataset loaders
- CPU-friendly reference vision architectures
- benchmark summary readers and leaderboard generation

The package is deliberately conservative about architecture claims.
If the current NanoTorch core is missing an operation needed for a faithful model
implementation, the corresponding model is marked experimental or unavailable
instead of being silently approximated.

## Current Scope

Available now:
- `MiniResNetDigits`
- `AlexNetTinyDigits`
- `MobileNetStyleTinyDigits`
- `ViTTinyDigits` (experimental)
- NanoDigits dataset helpers
- leaderboard generation from benchmark summaries

Not fully supported yet:
- `DenseNetTinyDigits`
  because the current framework does not yet expose a clean differentiable
  channel-concatenation path for dense feature reuse.
