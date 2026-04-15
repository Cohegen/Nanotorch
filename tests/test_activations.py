"""Unit tests for the activations module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from Tensor.tensor import Tensor
from activations.activations import Sigmoid, ReLU, Tanh, GELU, Softmax


class TestReLU:
    def test_positive_values(self):
        relu = ReLU()
        x = Tensor([1.0, 2.0, 3.0])
        result = relu(x)
        np.testing.assert_array_equal(result.data, [1.0, 2.0, 3.0])

    def test_negative_values(self):
        relu = ReLU()
        x = Tensor([-1.0, -2.0, -3.0])
        result = relu(x)
        np.testing.assert_array_equal(result.data, [0.0, 0.0, 0.0])

    def test_mixed_values(self):
        relu = ReLU()
        x = Tensor([-2.0, 0.0, 3.0])
        result = relu(x)
        np.testing.assert_array_equal(result.data, [0.0, 0.0, 3.0])

    def test_zero(self):
        relu = ReLU()
        x = Tensor([0.0])
        result = relu(x)
        assert result.data[0] == 0.0

    def test_parameters_empty(self):
        relu = ReLU()
        assert relu.parameters() == []

    def test_2d_input(self):
        relu = ReLU()
        x = Tensor([[-1, 2], [3, -4]])
        result = relu(x)
        np.testing.assert_array_equal(result.data, [[0, 2], [3, 0]])


class TestSigmoid:
    def test_zero_input(self):
        sigmoid = Sigmoid()
        x = Tensor([0.0])
        result = sigmoid(x)
        np.testing.assert_almost_equal(result.data[0], 0.5)

    def test_large_positive(self):
        sigmoid = Sigmoid()
        x = Tensor([100.0])
        result = sigmoid(x)
        np.testing.assert_almost_equal(result.data[0], 1.0, decimal=5)

    def test_large_negative(self):
        sigmoid = Sigmoid()
        x = Tensor([-100.0])
        result = sigmoid(x)
        np.testing.assert_almost_equal(result.data[0], 0.0, decimal=5)

    def test_output_range(self):
        sigmoid = Sigmoid()
        x = Tensor(np.linspace(-10, 10, 100))
        result = sigmoid(x)
        assert np.all(result.data >= 0.0)
        assert np.all(result.data <= 1.0)

    def test_parameters_empty(self):
        sigmoid = Sigmoid()
        assert sigmoid.parameters() == []

    def test_symmetry(self):
        sigmoid = Sigmoid()
        x_pos = Tensor([2.0])
        x_neg = Tensor([-2.0])
        r_pos = sigmoid(x_pos)
        r_neg = sigmoid(x_neg)
        np.testing.assert_almost_equal(r_pos.data[0] + r_neg.data[0], 1.0, decimal=5)


class TestTanh:
    def test_zero_input(self):
        tanh = Tanh()
        x = Tensor([0.0])
        result = tanh(x)
        np.testing.assert_almost_equal(result.data[0], 0.0)

    def test_output_range(self):
        tanh = Tanh()
        x = Tensor(np.linspace(-10, 10, 100))
        result = tanh(x)
        assert np.all(result.data >= -1.0)
        assert np.all(result.data <= 1.0)

    def test_antisymmetry(self):
        tanh = Tanh()
        x = Tensor([1.5])
        r_pos = tanh(x)
        r_neg = tanh(Tensor([-1.5]))
        np.testing.assert_almost_equal(r_pos.data[0], -r_neg.data[0], decimal=5)

    def test_parameters_empty(self):
        tanh = Tanh()
        assert tanh.parameters() == []


class TestGELU:
    def test_zero_input(self):
        gelu = GELU()
        x = Tensor([0.0])
        result = gelu(x)
        np.testing.assert_almost_equal(result.data[0], 0.0, decimal=5)

    def test_positive_input(self):
        gelu = GELU()
        x = Tensor([1.0])
        result = gelu(x)
        assert result.data[0] > 0.0

    def test_negative_input_near_zero(self):
        gelu = GELU()
        x = Tensor([-0.5])
        result = gelu(x)
        assert result.data[0] < 0.0

    def test_large_positive_approaches_identity(self):
        gelu = GELU()
        x = Tensor([10.0])
        result = gelu(x)
        np.testing.assert_almost_equal(result.data[0], 10.0, decimal=1)


class TestSoftmax:
    def test_basic(self):
        softmax = Softmax()
        x = Tensor([[1.0, 2.0, 3.0]])
        result = softmax(x)
        np.testing.assert_almost_equal(np.sum(result.data), 1.0)

    def test_sum_to_one(self):
        softmax = Softmax()
        x = Tensor([[0.5, 1.5, 2.5, 3.5]])
        result = softmax(x)
        np.testing.assert_almost_equal(np.sum(result.data, axis=-1), [1.0])

    def test_all_equal(self):
        softmax = Softmax()
        x = Tensor([[1.0, 1.0, 1.0]])
        result = softmax(x)
        np.testing.assert_almost_equal(result.data[0], [1 / 3, 1 / 3, 1 / 3], decimal=5)

    def test_numerical_stability(self):
        softmax = Softmax()
        x = Tensor([[1000.0, 1001.0, 1002.0]])
        result = softmax(x)
        assert not np.any(np.isnan(result.data))
        np.testing.assert_almost_equal(np.sum(result.data), 1.0)

    def test_batch(self):
        softmax = Softmax()
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        result = softmax(x, dim=-1)
        row_sums = np.sum(result.data, axis=-1)
        np.testing.assert_almost_equal(row_sums, [1.0, 1.0])

    def test_parameters_empty(self):
        softmax = Softmax()
        assert softmax.parameters() == []
