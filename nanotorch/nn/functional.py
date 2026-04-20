"""Functional neural-network helpers."""

import numpy as np
from Tensor import Tensor
from activations.activations import Sigmoid, ReLU, Tanh, GELU, Softmax
from losses.losses import log_softmax as _log_softmax

__all__ = [
    "relu",
    "sigmoid",
    "softmax",
    "log_softmax",
    "tanh",
    "gelu",
    "mse_loss",
    "cross_entropy",
    "binary_cross_entropy",
]

def relu(x):
    """Rectified Linear Unit: max(0, x)"""
    return ReLU().forward(x)

def sigmoid(x):
    """Sigmoid activation: 1 / (1 + exp(-x))"""
    return Sigmoid().forward(x)

def softmax(x, dim=-1):
    """Softmax activation: exp(x) / sum(exp(x))"""
    return Softmax().forward(x, dim=dim)

def log_softmax(x, dim=-1):
    """Log-Softmax for numerical stability"""
    return _log_softmax(x, dim=dim)

def tanh(x):
    """Hyperbolic tangent activation"""
    return Tanh().forward(x)

def gelu(x):
    """Gaussian Error Linear Unit"""
    return GELU().forward(x)

def mse_loss(predictions, targets):
    """Mean Squared Error loss"""
    from autograd.autograd import MSEBackward
    diff = predictions.data - targets.data
    mse = np.mean(diff**2)
    result = Tensor(mse)
    if predictions.requires_grad:
        result.requires_grad = True
        result._grad_fn = MSEBackward(predictions, targets)
    return result

def cross_entropy(logits, targets):
    """Cross Entropy loss for multi-class classification"""
    from autograd.autograd import CrossEntropyBackward
    log_probs = log_softmax(logits, dim=-1)
    batch_size = logits.shape[0]
    target_indices = targets.data.astype(int)
    selected_log_probs = log_probs.data[np.arange(batch_size), target_indices]
    ce_loss = -np.mean(selected_log_probs)
    result = Tensor(ce_loss)
    if logits.requires_grad:
        result.requires_grad = True
        result._grad_fn = CrossEntropyBackward(logits, targets)
    return result

def binary_cross_entropy(predictions, targets, eps=1e-7):
    """Binary Cross Entropy loss"""
    from autograd.autograd import BCEBackward
    p = np.clip(predictions.data, eps, 1-eps)
    y = targets.data
    bce = -np.mean(y * np.log(p) + (1-y) * np.log(1-p))
    result = Tensor(bce)
    if predictions.requires_grad:
        result.requires_grad = True
        result._grad_fn = BCEBackward(predictions, targets)
    return result
