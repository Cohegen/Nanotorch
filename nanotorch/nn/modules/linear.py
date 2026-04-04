"""PyTorch-style linear module exports."""

from layers.layers import Dropout, Layer, Linear

Module = Layer

__all__ = ["Dropout", "Layer", "Linear", "Module"]
