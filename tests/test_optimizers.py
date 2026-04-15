"""Unit tests for the optimizers module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from Tensor.tensor import Tensor
from optimizers.optimizers import SGD, Adam, AdamW


class TestSGD:
    def _make_param_with_grad(self, data, grad):
        t = Tensor(data, requires_grad=True)
        t.grad = Tensor(grad)
        return t

    def test_step_updates_params(self):
        param = self._make_param_with_grad([1.0, 2.0, 3.0], [0.1, 0.2, 0.3])
        optimizer = SGD([param], lr=1.0)
        optimizer.step()
        np.testing.assert_array_almost_equal(param.data, [0.9, 1.8, 2.7])

    def test_step_with_learning_rate(self):
        param = self._make_param_with_grad([1.0], [1.0])
        optimizer = SGD([param], lr=0.1)
        optimizer.step()
        np.testing.assert_almost_equal(param.data[0], 0.9)

    def test_zero_grad(self):
        param = self._make_param_with_grad([1.0], [1.0])
        optimizer = SGD([param], lr=0.1)
        optimizer.zero_grad()
        assert param.grad is None

    def test_multiple_steps(self):
        param = self._make_param_with_grad([10.0], [1.0])
        optimizer = SGD([param], lr=1.0)
        optimizer.step()
        assert param.data[0] == 9.0
        param.grad = Tensor([1.0])
        optimizer.step()
        assert param.data[0] == 8.0

    def test_multiple_params(self):
        p1 = self._make_param_with_grad([1.0], [0.5])
        p2 = self._make_param_with_grad([2.0], [0.5])
        optimizer = SGD([p1, p2], lr=1.0)
        optimizer.step()
        np.testing.assert_almost_equal(p1.data[0], 0.5)
        np.testing.assert_almost_equal(p2.data[0], 1.5)


class TestAdam:
    def _make_param_with_grad(self, data, grad):
        t = Tensor(data, requires_grad=True)
        t.grad = Tensor(grad)
        return t

    def test_step_updates_params(self):
        param = self._make_param_with_grad([1.0, 2.0], [0.1, 0.2])
        optimizer = Adam([param], lr=0.01)
        initial = param.data.copy()
        optimizer.step()
        # Params should change after step
        assert not np.array_equal(param.data, initial)

    def test_zero_grad(self):
        param = self._make_param_with_grad([1.0], [1.0])
        optimizer = Adam([param], lr=0.01)
        optimizer.zero_grad()
        assert param.grad is None

    def test_multiple_steps_converge(self):
        param = self._make_param_with_grad([5.0], [1.0])
        optimizer = Adam([param], lr=0.1)
        for _ in range(10):
            param.grad = Tensor([1.0])
            optimizer.step()
        # After many steps with constant gradient, param should decrease
        assert param.data[0] < 5.0


class TestAdamW:
    def _make_param_with_grad(self, data, grad):
        t = Tensor(data, requires_grad=True)
        t.grad = Tensor(grad)
        return t

    def test_step_updates_params(self):
        param = self._make_param_with_grad([1.0, 2.0], [0.1, 0.2])
        optimizer = AdamW([param], lr=0.01)
        initial = param.data.copy()
        optimizer.step()
        assert not np.array_equal(param.data, initial)

    def test_weight_decay_effect(self):
        # With weight decay, parameters should be pushed toward zero
        param = self._make_param_with_grad([10.0], [0.0])
        optimizer = AdamW([param], lr=0.01, weight_decay=0.1)
        optimizer.step()
        # Weight decay should reduce the parameter value
        assert param.data[0] < 10.0

    def test_zero_grad(self):
        param = self._make_param_with_grad([1.0], [1.0])
        optimizer = AdamW([param], lr=0.01)
        optimizer.zero_grad()
        assert param.grad is None
