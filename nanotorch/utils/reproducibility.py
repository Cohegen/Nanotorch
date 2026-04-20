"""Reproducibility helpers for NanoTorch projects."""

import os
import random

import numpy as np


def manual_seed(seed):
    """Seed Python and NumPy RNGs and return the applied seed."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    return seed


def seed_everything(seed):
    """Alias with a more experiment-oriented name."""
    return manual_seed(seed)


__all__ = ["manual_seed", "seed_everything"]

