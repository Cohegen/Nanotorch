# Optimization in Nano-Torch: A Deep Dive

Optimization is the heartbeat of neural network training. While the **Autograd** engine provides the "compass" (gradients), the **Optimizer** is the "engine" that decides how to move the parameters across a complex, high-dimensional loss landscape to find the global minimum.

In this guide, we explore the implementation, mathematics, and system-level implications of the optimizers available in `nano-torch`.

---

## 1. The Optimization Lifecycle

A typical training iteration in `nano-torch` is an orchestrated dance between the model, the data, and the optimizer. 

### Implementation Pattern
```python
import numpy as np
from Tensor import Tensor
from optimizers.optimizers import AdamW

# 1. Setup: Initialize parameters and optimizer
# We use AdamW with a standard learning rate and weight decay
params = [Tensor(np.random.randn(512, 512), requires_grad=True)]
optimizer = AdamW(params, lr=3e-4, weight_decay=0.01)

# --- Inside Training Loop ---
for epoch in range(num_epochs):
    for batch in dataloader:
        # 2. Clear previous gradients
        # Crucial: nano-torch gradients accumulate by default!
        optimizer.zero_grad()

        # 3. Forward pass
        # The computation graph is built dynamically here
        output = model(batch.data)
        loss = criterion(output, batch.target)

        # 4. Backward pass
        # This triggers the Autograd engine to populate .grad attributes
        loss.backward()

        # 5. Optimizer step
        # The optimizer looks at each param.grad and updates param.data
        optimizer.step()
        
        print(f"Step {optimizer.step_count}: Loss = {loss.data}")
```

---

## 2. The Optimizer Base Class: The Interface

All optimizers in `nano-torch` inherit from the `Optimizer` base class. This ensures a consistent API for any training loop, regardless of the underlying algorithm.

### Core Implementation Details:
*   **Initialization**: When an optimizer is initialized, it automatically ensures that all passed parameters participate in autograd by setting `requires_grad = True` and `grad = None`.
*   **State Management**: It tracks `step_count` (essential for algorithms like Adam that use bias correction) and maintains "buffers" (like momentum or variance) specific to each parameter.
*   **Gradient Zeroing**: The `zero_grad()` method iterates through all parameters and resets their `.grad` to `None`. This is critical because `nano-torch` accumulates gradients during the backward pass.
*   **Dual-Type Support**: The `step()` implementations are designed to handle gradients whether they are stored as `Tensor` objects or raw `numpy` arrays (which often happens when coming directly from the autograd engine).

---

## 3. Stochastic Gradient Descent (SGD)

SGD is the foundational optimization algorithm. In `nano-torch`, it is enhanced with **Momentum** and **Weight Decay**.

### The Math
1.  **Weight Decay (L2 Regularization)**: If $\lambda > 0$, we first adjust the gradient:
    $g_t = \nabla \theta + \lambda \theta$
2.  **Momentum Update**:
    $v_t = \beta \cdot v_{t-1} + g_t$
3.  **Parameter Update**:
    $\theta_{t+1} = \theta_t - \eta \cdot v_t$

### Checkpointing API
Unlike basic implementations, our `SGD` class provides an explicit API for state management:
*   `has_momentum()`: Returns `True` if $\beta > 0$.
*   `get_momentum()`: Safely retrieves the current momentum buffers (velocity vectors) for all parameters.
*   `set_momentum_state(state)`: Restores buffers from a saved checkpoint, with built-in validation to ensure the architecture matches.

**Default Hyperparameters:**
*   `lr`: 0.01
*   `momentum`: 0.0 (Vanilla SGD)
*   `weight_decay`: 0.0

---

## 4. Adam: Adaptive Moment Estimation

Adam is a "hybrid" optimizer that combines **Momentum** (first moment) and **RMSProp** (second moment). It calculates individual adaptive learning rates for every parameter.

### The Adaptive Mechanism
Adam tracks the "moving average" of both the gradients and their squares:

1.  **First Moment (m)**: Estimate of the mean gradient.
    $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
2.  **Second Moment (v)**: Estimate of the uncentered variance.
    $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$

### Bias Correction
Since $m$ and $v$ start at zero, they are biased toward zero early in training. Adam corrects this using the current `step_count` ($t$):
*   $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$
*   $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$

**Final Update**: $\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$

The $\epsilon$ (epsilon) parameter is a small constant (default `1e-8`) added for numerical stability to prevent division by zero.

---

## 5. AdamW: The Modern Standard

AdamW fixes a fundamental flaw in how Adam handles L2 regularization (Weight Decay).

### The "Decoupling" Fix
In standard Adam, weight decay is added to the gradient *before* the adaptive scaling. This means parameters with large gradients (and thus large $v_t$) receive *less* effective weight decay.

**AdamW** decouples them:
1.  Calculate the Adam update using only the "pure" gradient.
2.  Apply the update to the parameter.
3.  Apply weight decay directly: $\theta_{new} = \theta_{new} \cdot (1 - \eta \cdot \lambda)$

This ensures that regularization is applied uniformly relative to the learning rate, which is why AdamW is the preferred choice for **Transformers** and **Large Language Models**.

**Default Hyperparameters:**
*   `lr`: 0.001
*   `betas`: (0.9, 0.999)
*   `eps`: 1e-8
*   `weight_decay`: 0.01

---

## 6. System Analysis: Memory & Compute

Choosing an optimizer is a trade-off between convergence speed and hardware constraints.

### A. Memory Overhead
Optimizers require significant additional storage per parameter to store their state (buffers):

| Optimizer | Buffers | Total Bytes per Param (FP32) | Overhead vs Weight |
| :--- | :--- | :--- | :--- |
| **Vanilla SGD** | 0 | 4 | 1x |
| **SGD + Momentum**| 1 ($v$) | 8 | 2x |
| **Adam / AdamW** | 2 ($m, v$) | 12 | 3x |

**Scaling Example**: A 7B parameter LLM requires ~28GB for weights. Using AdamW adds another **56GB** of optimizer state, bringing the total to ~84GB (excluding gradients and activations).

### B. Computational Complexity
While all three are $O(N)$ where $N$ is the number of parameters, the constant factors differ:
*   **SGD**: Fast. Simple additions and subtractions.
*   **Adam/AdamW**: Slower per step. Requires power-of-two, square root, and division operations for every single parameter. However, they usually converge in significantly fewer steps than SGD.

---

## 7. Troubleshooting & Numerical Stability

### Symptom: Loss is NaN
*   **Likely Cause**: Learning rate is too high, causing weight explosions.
*   **Implementation Note**: Check the `eps` value in Adam. If your gradients are extremely small, `sqrt(v_t)` might vanish, making $\epsilon$ the only thing preventing division by zero.

### Symptom: Vanishing Gradients
*   **Likely Cause**: Deep architectures without residual connections or poor initialization.
*   **Optimizer Role**: Adam can sometimes help "rescue" training by scaling up the updates for parameters with tiny gradients, but it's not a substitute for good architecture.

### Symptom: Poor Generalization
*   **Likely Cause**: Adam/AdamW might be finding "sharp" minima that don't generalize well to test data.
*   **Fix**: Try switching to **SGD + Momentum** for the final stage of training, or increase `weight_decay` in AdamW.

---

## 8. Summary Comparison

| Feature | SGD | Adam | AdamW |
| :--- | :--- | :--- | :--- |
| **Learning Rate** | Global | Per-parameter (Adaptive) | Per-parameter (Adaptive) |
| **Momentum** | Optional | Built-in | Built-in |
| **Weight Decay** | Integrated L2 | Integrated L2 | **Decoupled** |
| **Best For** | CNNs, Simple models | RNNs, Sparse data | Transformers, LLMs |
| **Reliability** | Needs tuning | "Set and forget" | "Set and forget" + Regularization |
