import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Tensor import Tensor
from autograd.autograd import enable_autograd
from optimizers import Muon

enable_autograd()


def testing_muon():
    print("Testing Muon optimizer")

    matrix_param = Tensor([[1.0, -2.0], [0.5, 3.0]], requires_grad=True)
    vector_param = Tensor([0.25, -0.75], requires_grad=True)

    optimizer = Muon([matrix_param, vector_param], lr=0.05, momentum=0.9)

    matrix_param.grad = Tensor([[0.3, -0.1], [0.2, 0.4]])
    vector_param.grad = Tensor([0.2, -0.4])

    matrix_before = matrix_param.data.copy()
    vector_before = vector_param.data.copy()

    optimizer.step()

    assert optimizer.step_count == 1
    assert not np.allclose(matrix_param.data, matrix_before)
    assert not np.allclose(vector_param.data, vector_before)
    assert 'momentum_buffer' in optimizer.state[id(matrix_param)]
    assert 'momentum_buffer' in optimizer.state[id(vector_param)]

    # Muon should preserve matrix shape after flatten/orthogonalize/reshape.
    assert matrix_param.data.shape == matrix_before.shape
    assert vector_param.data.shape == vector_before.shape

    matrix_param.grad = Tensor([[0.3, -0.1], [0.2, 0.4]])
    vector_param.grad = Tensor([0.2, -0.4])
    optimizer.step()

    assert optimizer.step_count == 2
    assert np.linalg.norm(matrix_param.data - matrix_before) > 0
    assert np.linalg.norm(vector_param.data - vector_before) > 0

    print("Muon optimizer works correctly")


if __name__ == "__main__":
    testing_muon()
