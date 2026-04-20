import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import nanotorch as nt
import nanotorch.nn as nn
from nanotorch.optim import SGD
from nanotorch.utils import assert_finite_parameters, seed_everything, summarize_gradients


def main():
    seed_everything(7)

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    optimizer = SGD(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    x = nt.tensor(np.random.randn(6, 4).astype(np.float32))
    target = nt.tensor(np.random.randn(6, 2).astype(np.float32))

    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, target)
    loss.backward()
    optimizer.step()

    summary = summarize_gradients(model.parameters())
    assert_finite_parameters(model)

    print("NanoTorch smoke test passed")
    print(f"loss={float(loss.data):.6f}")
    print(f"grads_present={summary['gradients_present']}")
    print(f"grad_global_norm={summary['global_norm']:.6f}")


if __name__ == "__main__":
    main()

