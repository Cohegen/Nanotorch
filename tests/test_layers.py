"""Unit tests for the layers module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from Tensor.tensor import Tensor
from layers.layers import Layer, Linear, Dropout, Sequential, Parameter


class TestParameter:
    def test_creation(self):
        p = Parameter(Tensor([1.0, 2.0, 3.0]))
        assert isinstance(p, Tensor)
        assert hasattr(p, 'data')

    def test_repr(self):
        p = Parameter(Tensor([1.0]))
        r = repr(p)
        assert "Tensor" in r


class TestLinear:
    def test_output_shape(self):
        linear = Linear(4, 3)
        x = Tensor(np.random.randn(2, 4).astype(np.float32))
        out = linear(x)
        assert out.shape == (2, 3)

    def test_no_bias(self):
        linear = Linear(4, 3, bias=False)
        assert linear.bias is None
        x = Tensor(np.random.randn(2, 4).astype(np.float32))
        out = linear(x)
        assert out.shape == (2, 3)

    def test_parameters_with_bias(self):
        linear = Linear(4, 3, bias=True)
        params = linear.parameters()
        assert len(params) == 2

    def test_parameters_without_bias(self):
        linear = Linear(4, 3, bias=False)
        params = linear.parameters()
        assert len(params) == 1

    def test_weight_shape(self):
        linear = Linear(5, 3)
        assert linear.weight.data.shape == (3, 5)

    def test_bias_shape(self):
        linear = Linear(5, 3, bias=True)
        assert linear.bias.data.shape == (3,)

    def test_repr(self):
        linear = Linear(5, 3)
        r = repr(linear)
        assert "Linear" in r
        assert "5" in r
        assert "3" in r

    def test_single_sample(self):
        linear = Linear(4, 2)
        x = Tensor(np.random.randn(1, 4).astype(np.float32))
        out = linear(x)
        assert out.shape == (1, 2)


class TestDropout:
    def test_forward(self):
        dropout = Dropout(p=0.5)
        x = Tensor(np.ones((100, 100)).astype(np.float32))
        out = dropout(x)
        # Some values should be zero during dropout
        assert np.any(out.data == 0.0)

    def test_zero_dropout(self):
        dropout = Dropout(p=0.0)
        x = Tensor(np.ones((10, 10)).astype(np.float32))
        out = dropout(x)
        np.testing.assert_array_equal(out.data, x.data)

    def test_parameters_empty(self):
        dropout = Dropout(p=0.5)
        assert dropout.parameters() == []


class TestSequential:
    def test_forward(self):
        model = Sequential(
            Linear(4, 3),
            Linear(3, 2),
        )
        x = Tensor(np.random.randn(1, 4).astype(np.float32))
        out = model(x)
        assert out.shape == (1, 2)

    def test_parameters(self):
        model = Sequential(
            Linear(4, 3, bias=True),
            Linear(3, 2, bias=True),
        )
        params = model.parameters()
        # 2 layers * 2 params each (weight + bias)
        assert len(params) == 4

    def test_empty_sequential(self):
        model = Sequential()
        assert model.parameters() == []

    def test_repr(self):
        model = Sequential(Linear(4, 3))
        r = repr(model)
        assert "Sequential" in r

    def test_forwards_optional_args_to_layers_that_accept_them(self):
        class AddMask(Layer):
            def forward(self, x, mask=None):
                return x if mask is None else x + mask

        model = Sequential(
            AddMask(),
            Linear(4, 4, bias=False),
        )
        x = Tensor(np.ones((1, 4), dtype=np.float32))
        mask = Tensor(np.ones((1, 4), dtype=np.float32))

        out = model(x, mask=mask)

        expected = Tensor(np.full((1, 4), 2.0, dtype=np.float32)).matmul(
            model.layers[1].weight.transpose(-2, -1)
        )
        np.testing.assert_allclose(out.data, expected.data)


class TestLayer:
    def test_base_layer_parameters(self):
        layer = Layer()
        assert layer.parameters() == []
