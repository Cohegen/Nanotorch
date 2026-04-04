"""Neural-network compatibility exports."""

from activations.activations import GELU, ReLU, Sigmoid, Softmax, Tanh
from attention.attention import MultiHeadAttention, scaled_dot_product_attention
from convolution.convolutions import AvgPool2d, BatchNorm2d, Conv2d, MaxPool2d
from embeddings.embeddings import (
    Embedding,
    EmbeddingLayer,
    PositionalEncoding,
    create_sinusoidal_embeddings,
)
from layers.layers import Dropout, Layer, Linear, Sequential
from losses.losses import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss, log_softmax
from transformers.transformers import GPT, LayerNorm, MLP, TransformerBlock, create_causal_maks

from . import activations, attention, embeddings, functional, losses, modules, transformer

Module = Layer

__all__ = [
    "activations",
    "attention",
    "AvgPool2d",
    "BatchNorm2d",
    "BinaryCrossEntropyLoss",
    "Conv2d",
    "CrossEntropyLoss",
    "Dropout",
    "Embedding",
    "EmbeddingLayer",
    "embeddings",
    "functional",
    "GELU",
    "GPT",
    "Layer",
    "LayerNorm",
    "Linear",
    "log_softmax",
    "losses",
    "MaxPool2d",
    "MLP",
    "modules",
    "Module",
    "MSELoss",
    "MultiHeadAttention",
    "PositionalEncoding",
    "ReLU",
    "scaled_dot_product_attention",
    "Sequential",
    "Sigmoid",
    "Softmax",
    "Tanh",
    "transformer",
    "TransformerBlock",
    "create_causal_maks",
    "create_sinusoidal_embeddings",
]
