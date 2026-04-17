import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Tensor import Tensor
from activations import ELU, LeakyReLU, Mish, PReLU, SiLU, SwiGLU, TOLERANCE


def testing_extended_activations():
    print("Testing extended activation functions")

    x = Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    leaky_relu = LeakyReLU(negative_slope=0.1)
    leaky_result = leaky_relu(x)
    assert np.allclose(leaky_result.data, [-0.2, -0.1, 0.0, 1.0, 2.0], atol=TOLERANCE)

    silu = SiLU()
    silu_result = silu(Tensor([0.0, 1.0]))
    expected_silu = np.array([0.0, 1.0 / (1.0 + np.exp(-1.0))], dtype=np.float32)
    assert np.allclose(silu_result.data, expected_silu, atol=1e-6)

    mish = Mish()
    mish_zero = mish(Tensor([0.0]))
    assert np.allclose(mish_zero.data, [0.0], atol=TOLERANCE)
    mish_positive = mish(Tensor([1.0]))
    assert mish_positive.data[0] > 0.0

    prelu = PReLU(init=0.2)
    prelu_result = prelu(Tensor([-2.0, 3.0]))
    assert np.allclose(prelu_result.data, [-0.4, 3.0], atol=TOLERANCE)
    assert len(prelu.parameters()) == 1

    elu = ELU(alpha=1.0)
    elu_result = elu(Tensor([-1.0, 0.0, 2.0]))
    expected_elu = np.array([np.exp(-1.0) - 1.0, 0.0, 2.0], dtype=np.float32)
    assert np.allclose(elu_result.data, expected_elu, atol=1e-6)

    swiglu = SwiGLU()
    swiglu_input = Tensor([[1.0, 2.0, 3.0, 4.0]])
    swiglu_result = swiglu(swiglu_input)
    gate = np.array([3.0, 4.0], dtype=np.float32)
    expected_swiglu = np.array([[1.0, 2.0]], dtype=np.float32) * (gate / (1.0 + np.exp(-gate)))
    assert np.allclose(swiglu_result.data, expected_swiglu, atol=1e-6)

    try:
        swiglu(Tensor([[1.0, 2.0, 3.0]]))
        raise AssertionError("SwiGLU should require an even feature dimension")
    except ValueError:
        pass

    print("Extended activations work correctly")


if __name__ == "__main__":
    testing_extended_activations()
