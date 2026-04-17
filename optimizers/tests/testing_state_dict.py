import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Tensor import Tensor
from optimizers import Adam


def testing_optimizer_state_dict_roundtrip():
    param1 = Tensor([1.0, 2.0], requires_grad=True)
    param2 = Tensor([3.0, 4.0], requires_grad=True)
    optimizer = Adam([param1, param2], lr=0.01)

    param1.grad = Tensor([0.1, 0.2])
    param2.grad = Tensor([0.3, 0.4])
    optimizer.step()

    saved_state = optimizer.state_dict()

    new_param1 = Tensor([1.0, 2.0], requires_grad=True)
    new_param2 = Tensor([3.0, 4.0], requires_grad=True)
    new_optimizer = Adam([new_param1, new_param2], lr=0.5)
    new_optimizer.load_state_dict(saved_state)

    assert new_optimizer.step_count == optimizer.step_count
    assert new_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]

    original_state = optimizer.state[id(param1)]
    restored_state = new_optimizer.state[id(new_param1)]
    assert np.allclose(restored_state["exp_avg"], original_state["exp_avg"])
    assert np.allclose(restored_state["exp_avg_sq"], original_state["exp_avg_sq"])
    assert restored_state["step"] == original_state["step"]

    print("Optimizer state_dict works correctly")


if __name__ == "__main__":
    testing_optimizer_state_dict_roundtrip()
