# Optimization in NanoTorch

This guide explores how we enable neural networks to learn from gradients using sophisticated algorithms. In the training loop, after we compute gradients via backpropagation, the optimizer's job is to decide exactly how to update the model's parameters to minimize loss.

Note: `optimizers/optimizers.py` calls `enable_autograd()` at import time, so Tensor operations used for training will have gradient tracking enabled when you use these optimizers.

## The Optimization Lifecycle

A typical training iteration in `nano-torch` follows this path:

```python
import numpy as np
from Tensor import Tensor
from optimizers.optimizers import AdamW

# 1. Initialize parameters and optimizer
params = [Tensor(np.random.randn(10, 10), requires_grad=True)]
optimizer = AdamW(params, lr=1e-3, weight_decay=0.01)

# --- Inside Training Loop ---
# 2. Clear previous gradients
optimizer.zero_grad()

# 3. Forward pass & compute loss (e.g., MSE)
output = model(input_data)
loss = ((output - target) ** 2).mean()

# 4. Backward pass (calculates .grad for all params)
loss.backward()

# 5. Optimizer step (updates .data using .grad)
optimizer.step()
```

---

## Why Optimizers Matter

1.  **Convergence**: Faster and more stable training.
2.  **Generalization**: Proper regularization (like Weight Decay) helps the model perform well on unseen data.
3.  **Efficiency**: Adaptive algorithms handle varying gradient scales across different layers.

---
## Conceptual View: what `step()` is really doing
In this project, `loss.backward()` computes gradients and stores them in each parameter’s `.grad`.
Then the optimizer’s `step()` turns those gradients into a concrete parameter update by applying an algorithm-specific rule.

You can think of an optimizer as having three ingredients:

- **Gradient signal**: `param.grad` (the direction and relative magnitude of “how to change” the loss).
- **State** (optional): running estimates or buffers that smooth or rescale the gradient over time (e.g., `momentum_buffers`, `m_buffers`, `v_buffers`).
- **Learning-rate scaling**: `lr` decides how big each step is.

So the conceptual goal is: make updates that are
- stable (won’t blow up when gradients are noisy or large)
- scale-aware (different parameters can have very different gradient magnitudes)
- regularized (weight decay discourages overly large weights)

In `nano-torch`, `step()` also uses a simple contract:
- it skips parameters where `param.grad is None`
- it updates `param.data` in-place based on the algorithm rule

## The Optimizer Base Class

All optimizers in `nano-torch` inherit from the `Optimizer` class, which defines the standard interface:

*   `zero_grad()`: Clears the `.grad` attribute of all tracked parameters (sets it back to `None`). Gradients in `nano-torch` accumulate (they are added together) during backprop, which is useful for "Gradient Accumulation" (simulating larger batches). Failing to call this will lead to gradients from previous steps interfering with the current update.
*   `step()`: Performs the actual parameter update based on the specific algorithm (SGD, Adam, etc.).

In this repo’s `Optimizer` base class, `step()` implementations skip parameters whose `grad is None`. They also handle `param.grad` being either a `Tensor` or a raw numpy array (normalizing to `grad.data` when needed).

---

## Stochastic Gradient Descent (SGD)

SGD is the foundation of neural network optimization. It follows the simple principle: **"Move in the direction opposite to the gradient."**

### Intuition: The Rolling Ball
Imagine a ball on a hilly landscape. The gradient tells you which way is "up." To reach the bottom (minimum loss), you move "down."

### Momentum: Adding Mass
Pure SGD often suffers from oscillations in narrow "valleys." Adding **Momentum** is like giving the ball mass. It builds up speed in consistent directions and "plows through" small bumps or noisy gradients.

Conceptually, momentum turns “raw gradients” into a smoother update direction:
- the optimizer keeps a **velocity buffer** `v` for each parameter
- each step updates `v` as an exponential moving average of past gradients
- the parameter update uses this smoothed `v` instead of the instantaneous gradient

In this repo’s `SGD.step()`, if `weight_decay != 0`, it applies weight decay by adding `weight_decay * param.data` into the gradient before the momentum/parameter update.

**Formula:**
1.  `v_t = β * v_{t-1} + g_t` (Velocity update)
2.  `θ_t = θ_{t-1} - η * v_t` (Parameter update)

*   `β` (beta): Usually 0.9. It represents how much of the previous direction to keep.

**Code Example:**
```python
from optimizers.optimizers import SGD

# Simple SGD
optimizer = SGD(model.parameters(), lr=0.01)

# SGD with Momentum (recommended for CNNs)
optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
```

---

## Adam: Adaptive Moment Estimation

Adam is the "Swiss Army Knife" of optimizers. It automatically adjusts the learning rate for *each individual parameter*.

### Why Adaptive?
In a large network, a final layer weight might need a tiny learning rate (high precision), while a rarely-updated embedding weight might need a huge step. Adam handles this diversity automatically.

### The Two-Memory System
1.  **First Moment (m)**: Tracks the "Direction" (Momentum).
2.  **Second Moment (v)**: Tracks the "Scale" (Variance). If a gradient is consistently large, `v` becomes large, which *divides* the learning rate, making the step smaller and safer.

Conceptually, Adam is doing two things at once:
- it smooths gradients over time (through the first moment `m`)
- it normalizes the step using an estimate of gradient variability (through the second moment `v`)

That normalization is what makes Adam robust when different parameters experience gradients of very different magnitudes.

**Formula:**
`θ_t = θ_{t-1} - η * m_hat / (√v_hat + ε)`

*   **`eps` (ε)**: A tiny number (1e-8) to prevent division by zero if `v` is zero.
*   **`betas`**: The decay rates for `m` and `v`. Usually `(0.9, 0.999)`.

In this repo, `step_count` is used to compute bias-corrected moments (`m_hat`, `v_hat`). Early in training, the moving averages are biased toward zero; bias correction makes the first few steps behave sensibly.

**Code Example:**
```python
from optimizers.optimizers import Adam

# Standard Adam initialization
optimizer = Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
```

---

## AdamW: Fixing Weight Decay

AdamW is a specialized version of Adam that has become the standard for training **Transformers** and modern Large Language Models (LLMs).

### The "Bug" in Adam
In standard Adam implementations (and in this repo’s `Adam`), weight decay (L2 regularization) is folded into the gradient as:

`grad += weight_decay * param`

This happens *before* the Adam moment/adaptive update, so the effect of weight decay interacts with the adaptive scaling.

Conceptually, that means the regularization strength is no longer “constant”: parameters that get re-scaled by Adam’s adaptive step can experience weaker or stronger effective weight decay than you intended.

### The AdamW Fix
AdamW **decouples** the weight decay from the gradient update. It applies the Adam step first, then shrinks the parameters by a percentage each step:

`param.data *= (1 - lr * weight_decay)`

So the learner can think of AdamW as:
- Adam decides the gradient-based direction and size using moments
- then AdamW applies an extra “pull to zero” on the parameters, independent of the gradient statistics

**Correct Update:**
`θ_t = θ_{t-1} - η * Δθ - (η * λ * θ_{t-1})`

**Code Example:**
```python
from optimizers.optimizers import AdamW

# Standard choice for modern deep learning
optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

---

## Summary Analysis

| Optimizer | Memory | Complexity | Best For |
| :--- | :--- | :--- | :--- |
| **SGD** | 1x Params | Low | Basic ML, Fine-tuning |
| **SGD+M** | 2x Params | Low | Computer Vision (CNNs) |
| **Adam** | 3x Params | Medium | Generative Models, RNNs |
| **AdamW** | 3x Params | Medium | Transformers, LLMs, General DL |

---
## Practical debugging checklist (conceptual)
When training diverges (loss blows up) or becomes `NaN`, it is usually one of these conceptual issues:

- **Learning rate too high**: even a “smart” optimizer can’t rescue steps that are too large.
- **Forgetting `zero_grad()`**: gradients accumulate, so the optimizer may update as if you trained on multiple batches at once.
- **Bad numerical regime**: some losses/activations can produce extremely large gradients; adaptive methods help, but `eps` still matters.
- **Weight decay misconceptions**: if you expect AdamW-style regularization, but you’re using Adam (coupled weight decay), the regularization behavior may differ.

### Pro-Tips for Choosing:
*   **Start with AdamW**: It's robust and usually works well with its default settings (`lr=1e-3`, `weight_decay=0.01`).
*   **Use SGD+Momentum**: If you are training a very deep CNN and have the time to tune the learning rate carefully; it often achieves slightly better final accuracy than Adam on ImageNet-style tasks.
*   **Monitor your Gradients**: If you see "NaN" losses, check if your learning rate is too high or if your `eps` in Adam is too small.
