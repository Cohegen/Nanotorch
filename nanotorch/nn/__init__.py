"""Neural-network compatibility exports."""
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from activations.activations import ELU, GELU, LeakyReLU, Mish, PReLU, ReLU, SiLU, Sigmoid, Softmax, SwiGLU, Tanh
from attention.attention import (
    FlashMultiHeadAttention,
    FlashMultiHeadAttentionV2,
    FlashMultiHeadAttentionV3,
    GroupedQueryAttention,
    LinearMultiHeadAttention,
    MultiHeadAttention,
    MultiLatentAttention,
    MultiQueryAttention,
    PagedMultiHeadAttention,
    SparseMultiHeadAttention,
    flash_attention,
    flash_attention_v2,
    flash_attention_v3,
    linear_attention,
    paged_attention,
    scaled_dot_product_attention,
    sparse_attention,
)
from convolution.convolutions import AvgPool2d, BatchNorm2d, Conv2d, MaxPool2d
from embeddings.embeddings import (
    Embedding,
    EmbeddingLayer,
    PositionalEncoding,
    create_sinusoidal_embeddings,
)
from layers.layers import Dropout, Layer, Linear, Sequential
from losses.losses import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss, log_softmax
from transformers.transformers import GPT, LayerNorm, MLP, TransformerBlock, create_causal_mask
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
    "ELU",
    "functional",
    "GELU",
    "GPT",
    "Layer",
    "LayerNorm",
    "Linear",
    "log_softmax",
    "LeakyReLU",
    "losses",
    "MaxPool2d",
    "Mish",
    "MLP",
    "modules",
    "Module",
    "MSELoss",
    "MultiHeadAttention",
    "FlashMultiHeadAttention",
    "FlashMultiHeadAttentionV2",
    "FlashMultiHeadAttentionV3",
    "GroupedQueryAttention",
    "MultiQueryAttention",
    "MultiLatentAttention",
    "SparseMultiHeadAttention",
    "LinearMultiHeadAttention",
    "PagedMultiHeadAttention",
    "PReLU",
    "PositionalEncoding",
    "ReLU",
    "SiLU",
    "flash_attention",
    "flash_attention_v2",
    "flash_attention_v3",
    "sparse_attention",
    "linear_attention",
    "paged_attention",
    "scaled_dot_product_attention",
    "Sequential",
    "Sigmoid",
    "Softmax",
    "SwiGLU",
    "Tanh",
    "transformer",
    "TransformerBlock",
    "create_causal_mask",
    "create_sinusoidal_embeddings",
]
