"""Unit tests for the attention module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from Tensor.tensor import Tensor
from attention.attention import (
    _compute_attention_scores,
    _scale_scores,
    _apply_mask,
    scaled_dot_product_attention,
    MultiHeadAttention,
)


class TestComputeAttentionScores:
    def test_output_shape(self):
        q = Tensor(np.random.randn(2, 4, 8).astype(np.float32))
        k = Tensor(np.random.randn(2, 4, 8).astype(np.float32))
        scores = _compute_attention_scores(q, k)
        assert scores.shape == (2, 4, 4)

    def test_identity_scores(self):
        # Q == K should give high diagonal scores
        x = Tensor(np.eye(3, dtype=np.float32).reshape(1, 3, 3))
        scores = _compute_attention_scores(x, x)
        # Diagonal should be highest
        diag = np.diag(scores.data[0])
        assert np.all(diag >= scores.data[0])


class TestScaleScores:
    def test_scaling_factor(self):
        scores = Tensor(np.ones((1, 4, 4), dtype=np.float32))
        d_model = 64
        scaled = _scale_scores(scores, d_model)
        expected = 1.0 / np.sqrt(d_model)
        np.testing.assert_almost_equal(scaled.data[0, 0, 0], expected, decimal=5)


class TestApplyMask:
    def test_mask_application(self):
        scores = Tensor(np.ones((1, 3, 3), dtype=np.float32))
        # Causal mask: 1 = keep, 0 = mask out
        # Lower triangular mask where upper triangle positions get masked
        mask = Tensor(np.tril(np.ones((3, 3), dtype=np.float32)))
        masked = _apply_mask(scores, mask)
        # Masked positions (upper triangle, where mask=0) should have very negative values
        assert masked.data[0, 0, 1] < -1e4

    def test_identity_mask(self):
        scores = Tensor(np.ones((1, 3, 3), dtype=np.float32))
        # All-ones mask means keep everything
        mask = Tensor(np.ones((3, 3), dtype=np.float32))
        result = _apply_mask(scores, mask)
        np.testing.assert_array_almost_equal(result.data, scores.data)


class TestScaledDotProductAttention:
    def test_output_shape(self):
        q = Tensor(np.random.randn(1, 4, 8).astype(np.float32))
        k = Tensor(np.random.randn(1, 4, 8).astype(np.float32))
        v = Tensor(np.random.randn(1, 4, 8).astype(np.float32))
        out, weights = scaled_dot_product_attention(q, k, v)
        assert out.shape == (1, 4, 8)

    def test_with_mask(self):
        q = Tensor(np.random.randn(1, 4, 8).astype(np.float32))
        k = Tensor(np.random.randn(1, 4, 8).astype(np.float32))
        v = Tensor(np.random.randn(1, 4, 8).astype(np.float32))
        mask = Tensor(np.tril(np.ones((4, 4), dtype=np.float32)))
        out, weights = scaled_dot_product_attention(q, k, v, mask=mask)
        assert out.shape == (1, 4, 8)


class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(embed_dim=16, num_heads=4)
        x = Tensor(np.random.randn(1, 5, 16).astype(np.float32))
        out = mha(x)
        assert out.shape == (1, 5, 16)

    def test_parameters_count(self):
        mha = MultiHeadAttention(embed_dim=16, num_heads=4)
        params = mha.parameters()
        # Q, K, V, and output projection weights
        assert len(params) >= 4

    def test_single_head(self):
        mha = MultiHeadAttention(embed_dim=8, num_heads=1)
        x = Tensor(np.random.randn(1, 3, 8).astype(np.float32))
        out = mha(x)
        assert out.shape == (1, 3, 8)

    def test_with_mask(self):
        mha = MultiHeadAttention(embed_dim=16, num_heads=4)
        x = Tensor(np.random.randn(1, 5, 16).astype(np.float32))
        mask = Tensor(np.tril(np.ones((5, 5), dtype=np.float32)))
        out = mha(x, mask=mask)
        assert out.shape == (1, 5, 16)
