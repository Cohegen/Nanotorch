"""PyTorch-style nn.modules namespace."""

from activations.activations import ELU, GELU, LeakyReLU, Mish, PReLU, ReLU, SiLU, Sigmoid, Softmax, SwiGLU, Tanh
from attention.attention import MultiHeadAttention, scaled_dot_product_attention
from convolution.convolutions import AvgPool2d, BatchNorm2d, Conv2d, MaxPool2d
from embeddings.embeddings import (
    Embedding,
    EmbeddingLayer,
    PositionalEncoding,
    create_sinusoidal_embeddings,
)
from layers.layers import Dropout, Layer, Linear, Sequential
from losses.losses import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss
from transformers.transformers import GPT, LayerNorm, MLP, TransformerBlock, create_causal_maks

from . import (
    activation,
    attention,
    container,
    conv,
    linear,
    loss,
    normalization,
    pooling,
    sparse,
    transformer,
)

Module = Layer

__all__ = [
    "activation",
    "attention",
    "AvgPool2d",
    "BatchNorm2d",
    "BinaryCrossEntropyLoss",
    "container",
    "Conv2d",
    "CrossEntropyLoss",
    "Dropout",
    "Embedding",
    "EmbeddingLayer",
    "ELU",
    "GELU",
    "GPT",
    "Layer",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "linear",
    "loss",
    "MaxPool2d",
    "Mish",
    "MLP",
    "Module",
    "MSELoss",
    "MultiHeadAttention",
    "normalization",
    "PReLU",
    "PositionalEncoding",
    "pooling",
    "ReLU",
    "SiLU",
    "scaled_dot_product_attention",
    "Sequential",
    "Sigmoid",
    "Softmax",
    "sparse",
    "SwiGLU",
    "Tanh",
    "transformer",
    "TransformerBlock",
    "create_causal_maks",
    "create_sinusoidal_embeddings",
]
