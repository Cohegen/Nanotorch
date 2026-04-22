"""Unit tests for the quantization module."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from Tensor.tensor import Tensor
from layers.layers import Linear, Sequential
from optimization.quantization.quantization import (
    QuantizedLinear,
    dequantize_int8_per_channel,
    quantize_int8_per_channel,
    quantize_model,
)


class TestPerChannelQuantization:
    def test_quantize_int8_per_channel_preserves_channel_metadata(self):
        weight = Tensor(
            np.array(
                [
                    [-2.0, -0.5, 0.1, 0.9],
                    [0.01, 0.02, 0.03, 0.04],
                    [3.0, -3.5, 1.5, 0.2],
                ],
                dtype=np.float32,
            )
        )

        q_weight, scales, zero_points = quantize_int8_per_channel(weight, axis=0)

        assert q_weight.data.dtype == np.int8
        assert scales.shape == (3,)
        assert zero_points.shape == (3,)

        restored = dequantize_int8_per_channel(q_weight, scales, zero_points, axis=0)
        max_error = np.max(np.abs(restored.data - weight.data))
        assert max_error < 0.05

    def test_quantized_linear_per_channel_forward_stays_close(self):
        linear = Linear(4, 3)
        linear.weight = Tensor(
            np.array(
                [
                    [-2.0, -0.5, 0.2, 1.0],
                    [0.01, 0.02, 0.03, 0.04],
                    [3.0, -3.5, 1.5, 0.2],
                ],
                dtype=np.float32,
            )
        )
        linear.bias = Tensor(np.array([0.2, -0.1, 0.05], dtype=np.float32))

        quantized = QuantizedLinear(linear, weight_strategy="per_channel")
        x = Tensor(np.array([[0.4, -0.2, 0.8, 0.1], [1.0, 0.5, -0.4, 0.3]], dtype=np.float32))

        original_output = linear(x)
        quantized_output = quantized(x)

        assert quantized.q_weight.data.dtype == np.int8
        assert np.asarray(quantized.weight_scale).shape == (3,)
        mean_error = np.mean(np.abs(original_output.data - quantized_output.data))
        assert mean_error < 0.05


class TestDynamicQuantization:
    def test_quantized_linear_dynamic_activations_collect_runtime_params(self):
        linear = Linear(4, 3)
        linear.weight = Tensor(
            np.array(
                [
                    [0.8, -0.2, 0.3, 0.1],
                    [-0.4, 0.7, -0.5, 0.2],
                    [0.05, 0.15, -0.25, 0.35],
                ],
                dtype=np.float32,
            )
        )
        linear.bias = Tensor(np.array([0.1, -0.05, 0.02], dtype=np.float32))

        quantized = QuantizedLinear(
            linear,
            weight_strategy="per_tensor",
            activation_strategy="dynamic",
        )
        x = Tensor(np.array([[1.5, -2.0, 0.75, 3.25]], dtype=np.float32))

        original_output = linear(x)
        quantized_output = quantized(x)

        assert quantized.last_dynamic_input_scale is not None
        assert quantized.last_dynamic_input_zero_point is not None
        mean_error = np.mean(np.abs(original_output.data - quantized_output.data))
        assert mean_error < 0.05

    def test_quantize_model_applies_strategies_to_linear_layers(self):
        model = Sequential(
            Linear(4, 5),
            Linear(5, 2),
        )
        x = Tensor(np.array([[0.5, -0.25, 1.0, 0.75]], dtype=np.float32))
        original_output = model(x)
        calibration_data = [
            Tensor(np.array([[0.2, -0.1, 0.8, 0.3]], dtype=np.float32)),
            Tensor(np.array([[1.0, 0.5, -0.7, 0.4]], dtype=np.float32)),
        ]

        quantize_model(
            model,
            calibration_data=calibration_data,
            weight_strategy="per_channel",
            activation_strategy="dynamic",
        )

        assert isinstance(model.layers[0], QuantizedLinear)
        assert isinstance(model.layers[1], QuantizedLinear)
        assert model.layers[0].weight_strategy == "per_channel"
        assert model.layers[1].activation_strategy == "dynamic"

        quantized_output = model(x)
        mean_error = np.mean(np.abs(original_output.data - quantized_output.data))
        assert mean_error < 0.1

