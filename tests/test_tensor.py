"""Unit tests for the Tensor module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from Tensor.tensor import Tensor


class TestTensorCreation:
    def test_scalar_creation(self):
        t = Tensor(5.0)
        assert t.data == 5.0
        assert t.shape == ()
        assert t.num_elements == 1
        assert t.dtype == np.float32

    def test_vector_creation(self):
        t = Tensor([1, 2, 3])
        np.testing.assert_array_equal(t.data, np.array([1, 2, 3], dtype=np.float32))
        assert t.shape == (3,)
        assert t.num_elements == 3

    def test_matrix_creation(self):
        t = Tensor([[1, 2], [3, 4]])
        assert t.shape == (2, 2)
        assert t.num_elements == 4

    def test_3d_tensor_creation(self):
        t = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        assert t.shape == (2, 2, 2)
        assert t.num_elements == 8

    def test_requires_grad(self):
        t = Tensor([1, 2, 3], requires_grad=True)
        assert t.requires_grad is True
        assert t.grad is None

    def test_from_tensor(self):
        t1 = Tensor([1, 2, 3], requires_grad=True)
        t2 = Tensor(t1)
        np.testing.assert_array_equal(t1.data, t2.data)

    def test_from_list_of_tensors(self):
        t1 = Tensor([1, 2])
        t2 = Tensor([3, 4])
        t3 = Tensor([t1, t2])
        assert t3.shape == (2, 2)

    def test_device_default(self):
        t = Tensor([1])
        assert t.device == "cpu"


class TestTensorProperties:
    def test_size_no_dim(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        assert t.size() == (2, 3)

    def test_size_with_dim(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        assert t.size(0) == 2
        assert t.size(1) == 3

    def test_numel(self):
        t = Tensor([[1, 2], [3, 4], [5, 6]])
        assert t.numel() == 6

    def test_ndim(self):
        assert Tensor([1, 2, 3]).ndim == 1
        assert Tensor([[1, 2], [3, 4]]).ndim == 2
        assert Tensor([[[1]]]).ndim == 3

    def test_dim(self):
        t = Tensor([[1, 2], [3, 4]])
        assert t.dim() == 2

    def test_len(self):
        assert len(Tensor([1, 2, 3])) == 3
        assert len(Tensor([[1, 2], [3, 4]])) == 2

    def test_len_scalar(self):
        assert len(Tensor(5.0)) == 1

    def test_contiguous(self):
        t = Tensor([1, 2, 3])
        assert t.contiguous() is t


class TestTensorConversions:
    def test_numpy(self):
        t = Tensor([1, 2, 3])
        arr = t.numpy()
        assert isinstance(arr, np.ndarray)
        np.testing.assert_array_equal(arr, [1, 2, 3])

    def test_tolist(self):
        t = Tensor([1, 2, 3])
        assert t.tolist() == [1.0, 2.0, 3.0]

    def test_array_protocol(self):
        t = Tensor([1, 2, 3])
        arr = np.array(t)
        np.testing.assert_array_equal(arr, [1, 2, 3])

    def test_array_protocol_with_dtype(self):
        t = Tensor([1, 2, 3])
        arr = t.__array__(dtype=np.float64)
        assert arr.dtype == np.float64

    def test_repr(self):
        t = Tensor([1, 2])
        r = repr(t)
        assert "Tensor" in r

    def test_str(self):
        t = Tensor([1, 2])
        s = str(t)
        assert "Tensor" in s


class TestTensorArithmetic:
    def test_add_tensors(self):
        a = Tensor([1, 2, 3])
        b = Tensor([4, 5, 6])
        c = a + b
        np.testing.assert_array_equal(c.data, [5, 7, 9])

    def test_add_scalar(self):
        a = Tensor([1, 2, 3])
        c = a + 10
        np.testing.assert_array_equal(c.data, [11, 12, 13])

    def test_sub_tensors(self):
        a = Tensor([5, 6, 7])
        b = Tensor([1, 2, 3])
        c = a - b
        np.testing.assert_array_equal(c.data, [4, 4, 4])

    def test_sub_scalar(self):
        a = Tensor([5, 6, 7])
        c = a - 1
        np.testing.assert_array_equal(c.data, [4, 5, 6])

    def test_mul_tensors(self):
        a = Tensor([2, 3, 4])
        b = Tensor([5, 6, 7])
        c = a * b
        np.testing.assert_array_equal(c.data, [10, 18, 28])

    def test_mul_scalar(self):
        a = Tensor([2, 3, 4])
        c = a * 3
        np.testing.assert_array_equal(c.data, [6, 9, 12])

    def test_div_tensors(self):
        a = Tensor([10, 20, 30])
        b = Tensor([2, 4, 5])
        c = a / b
        np.testing.assert_array_equal(c.data, [5, 5, 6])

    def test_div_scalar(self):
        a = Tensor([10, 20, 30])
        c = a / 10
        np.testing.assert_array_equal(c.data, [1, 2, 3])


class TestTensorComparisons:
    def test_eq(self):
        a = Tensor([1, 2, 3])
        b = Tensor([1, 0, 3])
        result = (a == b)
        np.testing.assert_array_equal(result.data, [1, 0, 1])

    def test_lt(self):
        a = Tensor([1, 2, 3])
        b = Tensor([2, 2, 2])
        result = (a < b)
        np.testing.assert_array_equal(result.data, [1, 0, 0])

    def test_gt(self):
        a = Tensor([1, 2, 3])
        b = Tensor([2, 2, 2])
        result = (a > b)
        np.testing.assert_array_equal(result.data, [0, 0, 1])

    def test_le(self):
        a = Tensor([1, 2, 3])
        result = (a <= 2)
        np.testing.assert_array_equal(result.data, [1, 1, 0])

    def test_ge(self):
        a = Tensor([1, 2, 3])
        result = (a >= 2)
        np.testing.assert_array_equal(result.data, [0, 1, 1])

    def test_ne(self):
        a = Tensor([1, 2, 3])
        b = Tensor([1, 0, 3])
        result = (a != b)
        np.testing.assert_array_equal(result.data, [0, 1, 0])


class TestTensorMatmul:
    def test_matmul_2d(self):
        a = Tensor([[1, 2], [3, 4]])
        b = Tensor([[5, 6], [7, 8]])
        c = a.matmul(b)
        expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
        np.testing.assert_array_almost_equal(c.data, expected)

    def test_matmul_batched(self):
        a = Tensor(np.arange(12, dtype=np.float32).reshape(2, 2, 3))
        b = Tensor(np.arange(24, dtype=np.float32).reshape(2, 3, 4))
        c = a.matmul(b)
        expected = np.matmul(a.data, b.data)
        np.testing.assert_array_almost_equal(c.data, expected)

    def test_matmul_operator(self):
        a = Tensor([[1, 0], [0, 1]])
        b = Tensor([[5, 6], [7, 8]])
        c = a @ b
        np.testing.assert_array_almost_equal(c.data, b.data)

    def test_matmul_type_error(self):
        a = Tensor([[1, 2]])
        with pytest.raises(TypeError):
            a.matmul(5)

    def test_matmul_dimension_mismatch(self):
        a = Tensor([[1, 2, 3]])
        b = Tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            a.matmul(b)

    def test_matmul_scalar(self):
        a = Tensor(3.0)
        b = Tensor(4.0)
        c = a.matmul(b)
        assert c.data == 12.0


class TestTensorReshape:
    def test_reshape_basic(self):
        t = Tensor([1, 2, 3, 4, 5, 6])
        r = t.reshape(2, 3)
        assert r.shape == (2, 3)

    def test_reshape_infer_dim(self):
        t = Tensor([1, 2, 3, 4, 5, 6])
        r = t.reshape(2, -1)
        assert r.shape == (2, 3)

    def test_reshape_tuple(self):
        t = Tensor([1, 2, 3, 4])
        r = t.reshape((2, 2))
        assert r.shape == (2, 2)

    def test_reshape_invalid_size(self):
        t = Tensor([1, 2, 3, 4, 5])
        with pytest.raises(ValueError):
            t.reshape(2, 3)

    def test_reshape_multiple_infer(self):
        t = Tensor([1, 2, 3, 4])
        with pytest.raises(ValueError):
            t.reshape(-1, -1)

    def test_view_alias(self):
        t = Tensor([1, 2, 3, 4])
        r = t.view(2, 2)
        assert r.shape == (2, 2)


class TestTensorTranspose:
    def test_transpose_2d(self):
        t = Tensor([[1, 2, 3], [4, 5, 6]])
        tr = t.transpose()
        assert tr.shape == (3, 2)
        np.testing.assert_array_equal(tr.data, [[1, 4], [2, 5], [3, 6]])

    def test_transpose_with_dims(self):
        t = Tensor([[1, 2], [3, 4]])
        tr = t.transpose(0, 1)
        assert tr.shape == (2, 2)
        np.testing.assert_array_equal(tr.data, [[1, 3], [2, 4]])

    def test_transpose_1d_copy(self):
        t = Tensor([1, 2, 3])
        tr = t.transpose()
        np.testing.assert_array_equal(tr.data, t.data)

    def test_transpose_partial_dim_error(self):
        t = Tensor([[1, 2], [3, 4]])
        with pytest.raises(ValueError):
            t.transpose(0, None)


class TestTensorReductions:
    def test_sum_all(self):
        t = Tensor([[1, 2], [3, 4]])
        s = t.sum()
        assert s.data == 10.0

    def test_sum_axis(self):
        t = Tensor([[1, 2], [3, 4]])
        s = t.sum(axis=0)
        np.testing.assert_array_equal(s.data, [4, 6])

    def test_sum_keepdims(self):
        t = Tensor([[1, 2], [3, 4]])
        s = t.sum(axis=1, keepdims=True)
        assert s.shape == (2, 1)

    def test_mean_all(self):
        t = Tensor([[1, 2], [3, 4]])
        m = t.mean()
        assert m.data == 2.5

    def test_mean_axis(self):
        t = Tensor([[1, 2], [3, 4]])
        m = t.mean(axis=1)
        np.testing.assert_array_equal(m.data, [1.5, 3.5])

    def test_max_all(self):
        t = Tensor([[1, 5], [3, 2]])
        m = t.max()
        assert m.data == 5.0

    def test_max_axis(self):
        t = Tensor([[1, 5], [3, 2]])
        m = t.max(axis=1)
        np.testing.assert_array_equal(m.data, [5, 3])


class TestTensorIndexing:
    def test_getitem_single(self):
        t = Tensor([10, 20, 30])
        assert t[0].data == 10.0

    def test_getitem_slice(self):
        t = Tensor([10, 20, 30, 40])
        s = t[1:3]
        np.testing.assert_array_equal(s.data, [20, 30])

    def test_getitem_2d(self):
        t = Tensor([[1, 2], [3, 4]])
        row = t[0]
        np.testing.assert_array_equal(row.data, [1, 2])


class TestTensorSplit:
    def test_split_basic(self):
        t = Tensor([1, 2, 3, 4, 5, 6])
        parts = t.split(3)
        assert len(parts) == 2
        np.testing.assert_array_equal(parts[0].data, [1, 2, 3])
        np.testing.assert_array_equal(parts[1].data, [4, 5, 6])


class TestTensorMaskedFill:
    def test_masked_fill(self):
        t = Tensor([1, 2, 3, 4])
        mask = Tensor([1, 0, 1, 0])
        result = t.masked_fill(mask, -999)
        np.testing.assert_array_equal(result.data, [-999, 2, -999, 4])
