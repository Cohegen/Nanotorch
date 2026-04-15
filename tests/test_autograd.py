"""Unit tests for the autograd module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from Tensor.tensor import Tensor
from autograd.autograd import (
    AddBackward,
    SubBackward,
    MulBackward,
    DivBackward,
    MatMulBackward,
    SumBackward,
    MeanBackward,
    ReshapeBackward,
    enable_autograd,
    no_grad,
)


class TestAddBackward:
    def test_gradient_shapes(self):
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        fn = AddBackward(a, b)
        grad_output = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        grads = fn.apply(grad_output)
        assert len(grads) == 2

    def test_gradient_values(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        fn = AddBackward(a, b)
        grad_output = np.array([1.0, 1.0], dtype=np.float32)
        grads = fn.apply(grad_output)
        np.testing.assert_array_equal(grads[0], [1.0, 1.0])
        np.testing.assert_array_equal(grads[1], [1.0, 1.0])


class TestSubBackward:
    def test_gradient_values(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        fn = SubBackward(a, b)
        grad_output = np.array([1.0, 1.0], dtype=np.float32)
        grads = fn.apply(grad_output)
        np.testing.assert_array_equal(grads[0], [1.0, 1.0])
        np.testing.assert_array_equal(grads[1], [-1.0, -1.0])


class TestMulBackward:
    def test_gradient_values(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        fn = MulBackward(a, b)
        grad_output = np.array([1.0, 1.0], dtype=np.float32)
        grads = fn.apply(grad_output)
        np.testing.assert_array_equal(grads[0], [4.0, 5.0])
        np.testing.assert_array_equal(grads[1], [2.0, 3.0])


class TestDivBackward:
    def test_gradient_values(self):
        a = Tensor([6.0, 8.0], requires_grad=True)
        b = Tensor([2.0, 4.0], requires_grad=True)
        fn = DivBackward(a, b)
        grad_output = np.array([1.0, 1.0], dtype=np.float32)
        grads = fn.apply(grad_output)
        np.testing.assert_array_almost_equal(grads[0], [0.5, 0.25])
        np.testing.assert_array_almost_equal(grads[1], [-1.5, -0.5])


class TestMatMulBackward:
    def test_gradient_shapes(self):
        a = Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
        b = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
        fn = MatMulBackward(a, b)
        grad_output = np.ones((2, 4), dtype=np.float32)
        grads = fn.apply(grad_output)
        assert grads[0].shape == (2, 3)
        assert grads[1].shape == (3, 4)


class TestSumBackward:
    def test_gradient_broadcast(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        fn = SumBackward(a)
        grad_output = np.array(1.0, dtype=np.float32)
        grads = fn.apply(grad_output)
        np.testing.assert_array_equal(grads[0], [[1.0, 1.0], [1.0, 1.0]])


class TestMeanBackward:
    def test_gradient_values(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        fn = MeanBackward(a, axis=None, keepdims=False)
        grad_output = np.array(1.0, dtype=np.float32)
        grads = fn.apply(grad_output)
        expected = 1.0 / 4.0
        np.testing.assert_array_almost_equal(grads[0], [[expected, expected], [expected, expected]])


class TestReshapeBackward:
    def test_gradient_restores_shape(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        original_shape = a.shape
        fn = ReshapeBackward(a, original_shape)
        grad_output = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        grads = fn.apply(grad_output)
        assert grads[0].shape == (2, 2)


class TestNoGrad:
    def test_no_grad_context(self):
        with no_grad():
            a = Tensor([1.0, 2.0], requires_grad=True)
            b = Tensor([3.0, 4.0], requires_grad=True)
            c = a + b
            assert not getattr(c, 'requires_grad', False) or not hasattr(c, '_grad_fn') or c._grad_fn is None


class TestEndToEndBackward:
    def test_simple_add_backward(self):
        a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        c = a + b
        s = c.sum()
        s.backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_simple_mul_backward(self):
        a = Tensor([2.0, 3.0], requires_grad=True)
        b = Tensor([4.0, 5.0], requires_grad=True)
        c = a * b
        s = c.sum()
        s.backward()
        assert a.grad is not None
        assert b.grad is not None

    def test_chain_backward(self):
        a = Tensor([1.0, 2.0], requires_grad=True)
        b = Tensor([3.0, 4.0], requires_grad=True)
        c = a + b
        d = c * Tensor([2.0, 2.0])
        s = d.sum()
        s.backward()
        assert a.grad is not None
