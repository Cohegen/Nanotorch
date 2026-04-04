# NanoTorch Experiments

This folder contains small runnable scripts that exercise the packaged `nanotorch` API.

Each script uses plain assertions so it can work as a lightweight smoke test without introducing a separate test framework.

## Scripts

- `test_tensor_ops.py`: checks tensor creation, arithmetic, reshape, transpose, and matrix multiplication.
- `test_nn_pipeline.py`: checks a deterministic linear layer forward pass, a small sequential model, and cross-entropy loss output.
- `test_data_and_optim.py`: checks `TensorDataset`, `DataLoader`, image transforms, and a simple SGD parameter update.
- `run_all.py`: discovers and runs all `test_*.py` scripts in this directory.

## Run

From the project root:

```powershell
python experiments\run_all.py
```

If `python` is not available on your machine, fix the local interpreter or virtualenv first and then rerun the command above.
