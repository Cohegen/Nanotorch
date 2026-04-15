"""Unit tests for the losses module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from Tensor.tensor import Tensor
from losses.losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss, log_softmax


class TestMSELoss:
    def test_zero_loss(self):
        loss_fn = MSELoss()
        pred = Tensor([[1.0, 2.0, 3.0]])
        target = Tensor([[1.0, 2.0, 3.0]])
        loss = loss_fn(pred, target)
        np.testing.assert_almost_equal(loss.data, 0.0, decimal=5)

    def test_known_loss(self):
        loss_fn = MSELoss()
        pred = Tensor([[1.0, 0.0]])
        target = Tensor([[0.0, 0.0]])
        loss = loss_fn(pred, target)
        # MSE = mean((1-0)^2, (0-0)^2) = 0.5
        np.testing.assert_almost_equal(loss.data, 0.5, decimal=5)

    def test_positive_loss(self):
        loss_fn = MSELoss()
        pred = Tensor([[1.0, 2.0]])
        target = Tensor([[3.0, 4.0]])
        loss = loss_fn(pred, target)
        assert loss.data > 0


class TestCrossEntropyLoss:
    def test_output_positive(self):
        loss_fn = CrossEntropyLoss()
        logits = Tensor([[2.0, 1.0, 0.1]])
        targets = Tensor([[1, 0, 0]])
        loss = loss_fn(logits, targets)
        assert loss.data > 0

    def test_confident_correct_prediction(self):
        loss_fn = CrossEntropyLoss()
        logits = Tensor([[10.0, -10.0, -10.0]])
        targets = Tensor([[0]])  # index of correct class
        loss = loss_fn(logits, targets)
        # Very confident correct prediction should have low loss
        assert loss.data < 1.0


class TestBinaryCrossEntropyLoss:
    def test_perfect_prediction(self):
        loss_fn = BinaryCrossEntropyLoss()
        pred = Tensor([[0.99]])
        target = Tensor([[1.0]])
        loss = loss_fn(pred, target)
        assert loss.data < 0.1

    def test_worst_prediction(self):
        loss_fn = BinaryCrossEntropyLoss()
        pred = Tensor([[0.01]])
        target = Tensor([[1.0]])
        loss = loss_fn(pred, target)
        assert loss.data > 1.0

    def test_output_positive(self):
        loss_fn = BinaryCrossEntropyLoss()
        pred = Tensor([[0.5]])
        target = Tensor([[1.0]])
        loss = loss_fn(pred, target)
        assert loss.data > 0



class TestLogSoftmax:
    def test_output_negative(self):
        x = Tensor([[1.0, 2.0, 3.0]])
        result = log_softmax(x)
        # log_softmax values should all be <= 0
        assert np.all(result.data <= 0.0 + 1e-6)

    def test_exp_sums_to_one(self):
        x = Tensor([[1.0, 2.0, 3.0]])
        result = log_softmax(x)
        exp_sum = np.sum(np.exp(result.data), axis=-1)
        np.testing.assert_almost_equal(exp_sum, [1.0], decimal=5)

    def test_uniform_input(self):
        x = Tensor([[1.0, 1.0, 1.0]])
        result = log_softmax(x)
        expected = np.log(1.0 / 3.0)
        np.testing.assert_almost_equal(result.data[0], [expected, expected, expected], decimal=5)
