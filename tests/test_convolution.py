"""Unit tests for the convolution module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from Tensor.tensor import Tensor
from convolution.convolutions import (
    im2col,
    col2im,
    validate_4d_input,
    Conv2d,
    MaxPool2d,
    AvgPool2d,
    BatchNorm2d,
)


class TestIm2Col:
    def test_output_shape(self):
        # Input: (batch=1, channels=1, height=4, width=4)
        x = np.random.randn(1, 1, 4, 4).astype(np.float32)
        result = im2col(x, 3, 3, 1, 0)
        # Output should be (kernel_h*kernel_w*channels, out_h*out_w*batch)
        # out_h = (4-3)/1+1 = 2, out_w = 2
        assert result.shape[0] == 9  # 3*3*1
        assert result.shape[1] == 4  # 2*2*1

    def test_with_padding(self):
        x = np.random.randn(1, 1, 3, 3).astype(np.float32)
        result = im2col(x, 3, 3, 1, 1)
        # With pad=1: out_h = (3+2-3)/1+1 = 3, out_w = 3
        assert result.shape[1] == 9  # 3*3*1


class TestCol2Im:
    def test_roundtrip_shape(self):
        x = np.random.randn(1, 1, 4, 4).astype(np.float32)
        cols = im2col(x, 2, 2, 1, 0)
        reconstructed = col2im(cols, (1, 1, 4, 4), 2, 2, 1, 0)
        assert reconstructed.shape == (1, 1, 4, 4)


class TestValidate4dInput:
    def test_valid_input(self):
        x = Tensor(np.random.randn(1, 3, 8, 8).astype(np.float32))
        # Should not raise
        validate_4d_input(x, "Conv2d")

    def test_invalid_input_3d(self):
        x = Tensor(np.random.randn(3, 8, 8).astype(np.float32))
        with pytest.raises(ValueError):
            validate_4d_input(x, "Conv2d")


class TestConv2d:
    def test_output_shape(self):
        conv = Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=0)
        x = Tensor(np.random.randn(1, 1, 8, 8).astype(np.float32))
        out = conv(x)
        assert out.shape == (1, 4, 6, 6)

    def test_output_shape_with_padding(self):
        conv = Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1)
        x = Tensor(np.random.randn(1, 1, 8, 8).astype(np.float32))
        out = conv(x)
        assert out.shape == (1, 4, 8, 8)

    def test_output_shape_with_stride(self):
        conv = Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=2, padding=0)
        x = Tensor(np.random.randn(1, 1, 8, 8).astype(np.float32))
        out = conv(x)
        assert out.shape == (1, 2, 3, 3)

    def test_parameters(self):
        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        params = conv.parameters()
        assert len(params) >= 1

    def test_multiple_channels(self):
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 6, 6).astype(np.float32))
        out = conv(x)
        assert out.shape == (2, 8, 6, 6)


class TestMaxPool2d:
    def test_output_shape(self):
        pool = MaxPool2d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(1, 1, 4, 4).astype(np.float32))
        out = pool(x)
        assert out.shape == (1, 1, 2, 2)

    def test_max_values(self):
        pool = MaxPool2d(kernel_size=2, stride=2)
        data = np.array([[[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]]]], dtype=np.float32)
        x = Tensor(data)
        out = pool(x)
        expected = np.array([[[[6, 8], [14, 16]]]], dtype=np.float32)
        np.testing.assert_array_equal(out.data, expected)

    def test_parameters_empty(self):
        pool = MaxPool2d(kernel_size=2)
        assert pool.parameters() == []


class TestAvgPool2d:
    def test_output_shape(self):
        pool = AvgPool2d(kernel_size=2, stride=2)
        x = Tensor(np.random.randn(1, 1, 4, 4).astype(np.float32))
        out = pool(x)
        assert out.shape == (1, 1, 2, 2)

    def test_average_values(self):
        pool = AvgPool2d(kernel_size=2, stride=2)
        data = np.array([[[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]]]], dtype=np.float32)
        x = Tensor(data)
        out = pool(x)
        # Average of [1,2,5,6]=3.5, [3,4,7,8]=5.5, [9,10,13,14]=11.5, [11,12,15,16]=13.5
        expected = np.array([[[[3.5, 5.5], [11.5, 13.5]]]], dtype=np.float32)
        np.testing.assert_array_almost_equal(out.data, expected)


class TestBatchNorm2d:
    def test_output_shape(self):
        bn = BatchNorm2d(num_features=3)
        x = Tensor(np.random.randn(2, 3, 4, 4).astype(np.float32))
        out = bn(x)
        assert out.shape == (2, 3, 4, 4)

    def test_parameters(self):
        bn = BatchNorm2d(num_features=8)
        params = bn.parameters()
        assert len(params) >= 2
