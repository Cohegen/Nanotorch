import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from Tensor import Tensor
from layers.layers import Linear, Sequential
from nanotorch.utils.checkpointing import load_checkpoint, save_checkpoint


def testing_state_dict_roundtrip():
    model = Sequential(Linear(4, 3), Linear(3, 2))
    original_state = model.state_dict()

    for param in model.parameters():
        param.data.fill(123.0)

    load_result = model.load_state_dict(original_state)
    assert load_result["missing_keys"] == []
    assert load_result["unexpected_keys"] == []

    restored_state = model.state_dict()
    for key in original_state:
        assert np.allclose(restored_state[key], original_state[key]), f"Mismatch for {key}"


def testing_checkpoint_save_and_load():
    model = Sequential(Linear(4, 3), Linear(3, 2))
    reference_state = model.state_dict()

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pkl")
        save_checkpoint(
            checkpoint_path,
            model,
            epoch=3,
            metadata={"tag": "unit-test"},
        )

        for param in model.parameters():
            param.data.fill(-5.0)

        checkpoint = load_checkpoint(checkpoint_path, model=model)
        assert checkpoint["epoch"] == 3
        assert checkpoint["metadata"]["tag"] == "unit-test"

        restored_state = model.state_dict()
        for key in reference_state:
            assert np.allclose(restored_state[key], reference_state[key]), f"Checkpoint mismatch for {key}"

    print("State dict and checkpoint helpers work correctly")


if __name__ == "__main__":
    testing_state_dict_roundtrip()
    testing_checkpoint_save_and_load()
