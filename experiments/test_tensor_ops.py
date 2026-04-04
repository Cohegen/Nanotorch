from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import nanotorch as nt


def assert_allclose(actual, expected, message):
    if not np.allclose(actual, expected):
        raise AssertionError(f"{message}\nactual={actual}\nexpected={expected}")


def main():
    left = nt.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = nt.tensor([[0.5, 1.5], [2.5, 3.5]])

    assert left.shape == (2, 2)
    assert left.dtype == np.float32

    added = left + right
    multiplied = left * right
    reshaped = left.reshape(4)
    transposed = left.transpose()
    product = left @ nt.tensor([[1.0, 0.0], [0.0, 1.0]])

    assert_allclose(added.data, np.array([[1.5, 3.5], [5.5, 7.5]], dtype=np.float32), "tensor addition failed")
    assert_allclose(multiplied.data, np.array([[0.5, 3.0], [7.5, 14.0]], dtype=np.float32), "tensor multiplication failed")
    assert reshaped.shape == (4,)
    assert_allclose(reshaped.data, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), "tensor reshape failed")
    assert_allclose(transposed.data, np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32), "tensor transpose failed")
    assert_allclose(product.data, left.data, "matrix multiplication with identity failed")

    print("test_tensor_ops: PASS")


if __name__ == "__main__":
    main()
