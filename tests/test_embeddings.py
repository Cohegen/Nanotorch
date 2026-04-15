"""Unit tests for the embeddings module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from Tensor.tensor import Tensor
from embeddings.embeddings import (
    Embedding,
    PositionalEncoding,
    EmbeddingLayer,
    _compute_sinusoidal_table,
    create_sinusoidal_embeddings,
)


class TestEmbedding:
    def test_output_shape(self):
        emb = Embedding(vocab_size=100, embed_dim=32)
        indices = Tensor(np.array([0, 5, 10], dtype=np.float32))
        result = emb(indices)
        assert result.shape == (3, 32)

    def test_weight_shape(self):
        emb = Embedding(vocab_size=50, embed_dim=16)
        assert emb.weight.shape == (50, 16)

    def test_single_index(self):
        emb = Embedding(vocab_size=10, embed_dim=8)
        indices = Tensor(np.array([3], dtype=np.float32))
        result = emb(indices)
        assert result.shape == (1, 8)

    def test_parameters(self):
        emb = Embedding(vocab_size=10, embed_dim=8)
        params = emb.parameters()
        assert len(params) == 1
        assert params[0].shape == (10, 8)

    def test_repr(self):
        emb = Embedding(vocab_size=10, embed_dim=8)
        r = repr(emb)
        assert "Embedding" in r
        assert "10" in r
        assert "8" in r


class TestPositionalEncoding:
    def test_output_shape(self):
        pe = PositionalEncoding(max_seq_len=50, embed_dim=32)
        x = Tensor(np.random.randn(2, 10, 32).astype(np.float32))
        result = pe(x)
        assert result.shape == (2, 10, 32)

    def test_adds_position_info(self):
        pe = PositionalEncoding(max_seq_len=50, embed_dim=16)
        x = Tensor(np.zeros((1, 5, 16), dtype=np.float32))
        result = pe(x)
        # Result should not be all zeros since positional encoding was added
        assert not np.allclose(result.data, 0.0)

    def test_parameters(self):
        pe = PositionalEncoding(max_seq_len=50, embed_dim=16)
        params = pe.parameters()
        assert len(params) >= 1

    def test_repr(self):
        pe = PositionalEncoding(max_seq_len=50, embed_dim=16)
        r = repr(pe)
        assert "PositionalEncoding" in r


class TestEmbeddingLayer:
    def test_output_shape(self):
        layer = EmbeddingLayer(vocab_size=100, embed_dim=32, max_seq_len=50)
        indices = Tensor(np.array([[0, 1, 2, 3]], dtype=np.float32))
        result = layer(indices)
        assert result.shape[-1] == 32

    def test_parameters(self):
        layer = EmbeddingLayer(vocab_size=100, embed_dim=32, max_seq_len=50)
        params = layer.parameters()
        assert len(params) >= 2

    def test_repr(self):
        layer = EmbeddingLayer(vocab_size=100, embed_dim=32, max_seq_len=50)
        r = repr(layer)
        assert "EmbeddingLayer" in r


class TestSinusoidalTable:
    def test_output_shape(self):
        table = _compute_sinusoidal_table(10, 16)
        assert table.shape == (10, 16)

    def test_values_bounded(self):
        table = _compute_sinusoidal_table(20, 32)
        assert np.all(table >= -1.0)
        assert np.all(table <= 1.0)

    def test_create_sinusoidal_embeddings(self):
        emb = create_sinusoidal_embeddings(10, 16)
        assert isinstance(emb, Tensor)
        assert emb.shape == (10, 16)
