"""PyTorch-like compatibility package built on the existing NanoTorch modules."""

from Tensor.tensor import Tensor

from . import autograd, nn, optim, optimization, tokenization, training, utils


def tensor(data, requires_grad=False):
    """Create a tensor using the existing Tensor implementation."""
    return Tensor(data, requires_grad=requires_grad)


__all__ = [
    "Tensor",
    "autograd",
    "nn",
    "optim",
    "optimization",
    "tensor",
    "tokenization",
    "training",
    "utils",
]
