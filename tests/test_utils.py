"""Tests for NanoTorch utility helpers."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

import nanotorch as nt
from Tensor.tensor import Tensor
from nanotorch.utils import (
    assert_finite_tensor,
    collect_gradient_issues,
    manual_seed,
    seed_everything,
    summarize_gradients,
)


def test_manual_seed_reproducible():
    manual_seed(11)
    first = np.random.randn(4)
    manual_seed(11)
    second = np.random.randn(4)
    np.testing.assert_allclose(first, second)


def test_seed_everything_alias():
    seed = seed_everything(5)
    assert seed == 5


def test_assert_finite_tensor_raises_on_nan():
    with pytest.raises(ValueError):
        assert_finite_tensor(Tensor([1.0, np.nan]), name="bad_tensor")


def test_collect_gradient_issues_detects_missing_and_nonfinite():
    good = Tensor([1.0], requires_grad=True)
    good.grad = Tensor([0.5])
    missing = Tensor([1.0], requires_grad=True)
    bad = Tensor([1.0], requires_grad=True)
    bad.grad = Tensor([np.inf])

    issues = collect_gradient_issues([good, missing, bad])

    assert issues["missing_grad_indices"] == [1]
    assert issues["nonfinite_grad_indices"] == [2]


def test_summarize_gradients_reports_global_norm():
    p1 = Tensor([1.0], requires_grad=True)
    p1.grad = Tensor([3.0])
    p2 = Tensor([1.0], requires_grad=True)
    p2.grad = Tensor([4.0])

    summary = summarize_gradients([p1, p2])

    assert summary["gradients_present"] == 2
    np.testing.assert_allclose(summary["global_norm"], 5.0, atol=1e-6)
    np.testing.assert_allclose(summary["max_abs"], 4.0, atol=1e-6)


def test_nanotorch_exports_seed_helpers():
    assert nt.manual_seed is not None
    assert nt.seed_everything is not None
