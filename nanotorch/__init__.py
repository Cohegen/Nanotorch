"""PyTorch-like compatibility package built on the existing NanoTorch modules."""

from Tensor.tensor import Tensor

from . import autograd, nn, optim, optimization, tokenization, training, utils
from .utils import load_checkpoint, load_state_dict, save_checkpoint, save_state_dict
from .utils import manual_seed, seed_everything


def tensor(data, requires_grad=False):
    """Create a tensor using the existing Tensor implementation."""
    return Tensor(data, requires_grad=requires_grad)


__all__ = [
    "Tensor",
    "autograd",
    "nn",
    "optim",
    "optimization",
    "load_checkpoint",
    "load_state_dict",
    "manual_seed",
    "save_checkpoint",
    "save_state_dict",
    "seed_everything",
    "tensor",
    "tokenization",
    "training",
    "utils",
]
