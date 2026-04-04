from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nanotorch as nt
import nanotorch.nn as nn


def assert_allclose(actual, expected, message, atol=1e-6):
    if not np.allclose(actual, expected, atol=atol):
        raise AssertionError(f"{message}\nactual={actual}\nexpected={expected}")


def main():
    linear = nn.Linear(2, 3)
    linear.weight.data = np.array(
        [[1.0, -1.0, 0.5], [0.0, 2.0, -0.5]],
        dtype=np.float32,
    )
    linear.bias.data = np.array([0.1, 0.2, 0.3], dtype=np.float32)

    x = nt.tensor([[2.0, 3.0]])
    linear_out = linear(x)
    expected_linear = np.array([[2.1, 4.2, -0.2]], dtype=np.float32)
    assert_allclose(linear_out.data, expected_linear, "linear forward pass failed")

    model = nn.Sequential(
        linear,
        nn.ReLU(),
    )
    seq_out = model(x)
    assert_allclose(seq_out.data, np.array([[2.1, 4.2, 0.0]], dtype=np.float32), "sequential forward pass failed")

    loss_fn = nn.CrossEntropyLoss()
    logits = nt.tensor([[2.0, 0.5, -1.0], [0.1, 1.7, -0.4]])
    targets = nt.tensor([0, 1])
    loss = loss_fn(logits, targets)

    expected_loss = np.array(0.2611176, dtype=np.float32)
    assert_allclose(loss.data, expected_loss, "cross entropy output drifted", atol=1e-5)

    print("test_nn_pipeline: PASS")


if __name__ == "__main__":
    main()
