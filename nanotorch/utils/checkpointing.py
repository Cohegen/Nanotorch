"""Checkpoint helpers for NanoTorch models and optimizers."""

import pickle
from pathlib import Path


def save_state_dict(path, state_dict):
    """Persist a state dictionary to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        pickle.dump(state_dict, handle)
    return path


def load_state_dict(path):
    """Load a serialized state dictionary from disk."""
    with open(path, "rb") as handle:
        return pickle.load(handle)


def save_checkpoint(path, model, optimizer=None, epoch=None, metadata=None):
    """Save a full training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metadata": {} if metadata is None else dict(metadata),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    return save_state_dict(path, checkpoint)


def load_checkpoint(path, model=None, optimizer=None, strict=True):
    """Load a checkpoint and optionally restore model and optimizer state."""
    checkpoint = load_state_dict(path)

    if model is not None and "model_state_dict" in checkpoint:
        checkpoint["model_load_result"] = model.load_state_dict(
            checkpoint["model_state_dict"], strict=strict
        )

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
