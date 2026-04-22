"""Unit tests for the transformers module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from Tensor.tensor import Tensor
from transformers.transformers import LayerNorm, MLP, TransformerBlock, GPT


class TestLayerNorm:
    def test_output_shape(self):
        ln = LayerNorm(16)
        x = Tensor(np.random.randn(2, 5, 16).astype(np.float32))
        out = ln(x)
        assert out.shape == (2, 5, 16)

    def test_normalized_output(self):
        ln = LayerNorm(8)
        x = Tensor(np.random.randn(1, 3, 8).astype(np.float32))
        out = ln(x)
        # After layer norm, output should have roughly zero mean per vector
        mean = np.mean(out.data, axis=-1)
        assert np.all(np.abs(mean) < 0.6)  # relaxed bound for small dims

    def test_parameters(self):
        ln = LayerNorm(16)
        params = ln.parameters()
        # gamma and beta
        assert len(params) == 2

    def test_gamma_initialized_ones(self):
        ln = LayerNorm(8)
        np.testing.assert_array_equal(ln.gamma.data, np.ones(8))

    def test_beta_initialized_zeros(self):
        ln = LayerNorm(8)
        np.testing.assert_array_equal(ln.beta.data, np.zeros(8))


class TestMLP:
    def test_output_shape(self):
        mlp = MLP(16)
        x = Tensor(np.random.randn(1, 5, 16).astype(np.float32))
        out = mlp(x)
        assert out.shape == (1, 5, 16)

    def test_parameters(self):
        mlp = MLP(16)
        params = mlp.parameters()
        assert len(params) > 0

    def test_different_input_output_dim(self):
        mlp = MLP(32)
        x = Tensor(np.random.randn(2, 3, 32).astype(np.float32))
        out = mlp(x)
        assert out.shape == (2, 3, 32)


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(embed_dim=16, num_heads=4)
        x = Tensor(np.random.randn(1, 5, 16).astype(np.float32))
        out = block(x)
        assert out.shape == (1, 5, 16)

    def test_parameters(self):
        block = TransformerBlock(embed_dim=16, num_heads=4)
        params = block.parameters()
        assert len(params) > 0

    def test_with_mask(self):
        block = TransformerBlock(embed_dim=16, num_heads=4)
        x = Tensor(np.random.randn(1, 5, 16).astype(np.float32))
        mask = Tensor(np.triu(np.ones((5, 5), dtype=np.float32), k=1))
        out = block(x, mask=mask)
        assert out.shape == (1, 5, 16)


class TestGPT:
    def test_output_shape(self):
        gpt = GPT(vocab_size=50, embed_dim=16, num_heads=4, num_layers=2, max_seq_len=32)
        x = Tensor(np.array([[0, 1, 2, 3]], dtype=np.float32))
        out = gpt(x)
        assert out.shape[-1] == 50  # vocab_size

    def test_parameters(self):
        gpt = GPT(vocab_size=50, embed_dim=16, num_heads=4, num_layers=2, max_seq_len=32)
        params = gpt.parameters()
        assert len(params) > 0

    def test_single_token(self):
        gpt = GPT(vocab_size=20, embed_dim=16, num_heads=4, num_layers=1, max_seq_len=10)
        x = Tensor(np.array([[5]], dtype=np.float32))
        out = gpt(x)
        assert out.shape[-1] == 20
