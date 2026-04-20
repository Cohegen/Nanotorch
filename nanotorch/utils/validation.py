"""Validation helpers for safer NanoTorch training loops."""

import numpy as np


def _to_numpy(value):
    if hasattr(value, "data"):
        return np.asarray(value.data, dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def is_finite_tensor(value):
    """Return True when all values are finite."""
    return bool(np.all(np.isfinite(_to_numpy(value))))


def assert_finite_tensor(value, name="tensor"):
    """Raise when the provided tensor-like value contains NaN or inf."""
    array = _to_numpy(value)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values")
    return value


def assert_finite_parameters(model):
    """Validate that every model parameter remains finite."""
    for index, parameter in enumerate(model.parameters()):
        assert_finite_tensor(parameter, name=f"parameter[{index}]")
    return model


def collect_gradient_issues(parameters):
    """Summarize missing and non-finite gradients across a parameter list."""
    issues = {
        "missing_grad_indices": [],
        "nonfinite_grad_indices": [],
    }
    for index, parameter in enumerate(parameters):
        grad = getattr(parameter, "grad", None)
        if grad is None:
            issues["missing_grad_indices"].append(index)
            continue
        if not is_finite_tensor(grad):
            issues["nonfinite_grad_indices"].append(index)
    return issues


def summarize_gradients(parameters):
    """Return a compact gradient summary for logging and tests."""
    gradients_present = 0
    squared_norm = 0.0
    max_abs = 0.0

    for parameter in parameters:
        grad = getattr(parameter, "grad", None)
        if grad is None:
            continue
        grad_array = _to_numpy(grad)
        gradients_present += 1
        squared_norm += float(np.sum(grad_array ** 2))
        max_abs = max(max_abs, float(np.max(np.abs(grad_array))))

    return {
        "gradients_present": gradients_present,
        "global_norm": float(np.sqrt(squared_norm)),
        "max_abs": float(max_abs),
    }


def assert_no_gradient_issues(parameters):
    """Raise when required gradients are missing or non-finite."""
    issues = collect_gradient_issues(parameters)
    if issues["missing_grad_indices"] or issues["nonfinite_grad_indices"]:
        raise ValueError(
            "Gradient validation failed: "
            f"missing={issues['missing_grad_indices']}, "
            f"nonfinite={issues['nonfinite_grad_indices']}"
        )
    return parameters


__all__ = [
    "assert_finite_parameters",
    "assert_finite_tensor",
    "assert_no_gradient_issues",
    "collect_gradient_issues",
    "is_finite_tensor",
    "summarize_gradients",
]
