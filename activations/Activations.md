# Activation Functions

Activation functions are mathematical equations that determine the output of a neural network node. They introduce **non-linearity** into the network, allowing it to learn complex patterns in data. Without non-linear activation functions, a multi-layer neural network would behave like a single-layer linear model, regardless of how many layers it has.

This module implements several key activation functions used in modern deep learning architectures.

---

## Implemented Activations

### 1. Sigmoid
The Sigmoid function maps any real-valued number into a range between 0 and 1. It is traditionally used in the output layer of binary classification models.

**Formula:**
$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Key Characteristics:**
- **Range:** (0, 1)
- **Use Case:** Binary classification, gating mechanisms (e.g., LSTMs).
- **Implementation Note:** To prevent numerical overflow from large negative values of $x$, our implementation clips inputs to $[-500, 500]$ and uses a numerically stable formulation.

**Usage Example:**
```python
from activations.activations import Sigmoid
from Tensor import Tensor
import numpy as np

sigmoid = Sigmoid()
x = Tensor(np.array([-1.0, 0.0, 1.0]))
output = sigmoid(x)
print(output.data) 
# Output: [0.26894142, 0.5, 0.73105858]
```

---

### 2. ReLU (Rectified Linear Unit)
ReLU is the most widely used activation function for hidden layers. It outputs the input directly if it is positive; otherwise, it outputs zero.

**Formula:**
$$f(x) = \max(0, x)$$

**Key Characteristics:**
- **Range:** [0, $\infty$)
- **Use Case:** Hidden layers in CNNs and MLP.
- **Pros:** Computationally efficient, helps mitigate the vanishing gradient problem.

**Usage Example:**
```python
from activations.activations import ReLU
from Tensor import Tensor
import numpy as np

relu = ReLU()
x = Tensor(np.array([-2.0, -0.5, 1.0, 2.0]))
output = relu(x)
print(output.data)
# Output: [0.0, 0.0, 1.0, 2.0]
```

---

### 3. Tanh (Hyperbolic Tangent)
Tanh is similar to Sigmoid but maps inputs to a range between -1 and 1. It is zero-centered, which often makes optimization easier than Sigmoid.

**Formula:**
$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

**Key Characteristics:**
- **Range:** (-1, 1)
- **Use Case:** RNN hidden states, hidden layers where zero-centered output is preferred.

**Usage Example:**
```python
from activations.activations import Tanh
from Tensor import Tensor
import numpy as np

tanh = Tanh()
x = Tensor(np.array([-1.0, 0.0, 1.0]))
output = tanh(x)
print(output.data)
# Output: [-0.76159416, 0.0, 0.76159416]
```

---

### 4. GELU (Gaussian Error Linear Unit)
GELU is a smooth approximation of ReLU that weights inputs by their percentile rather than a hard threshold. It is the standard activation function in modern Transformer models (GPT, BERT).

**Formula (Approximation):**
$$GELU(x) \approx x \cdot \sigma(1.702x)$$

**Key Characteristics:**
- **Range:** $\approx$ [-0.17, $\infty$)
- **Use Case:** Modern Transformers and LLMs.

**Usage Example:**
```python
from activations.activations import GELU
from Tensor import Tensor
import numpy as np

gelu = GELU()
x = Tensor(np.array([-1.0, 0.0, 1.0]))
output = gelu(x)
print(output.data)
# Output: [-0.1542426, 0.0, 0.8457574]
```

---

### 5. Softmax
Softmax turns a vector of numbers into a vector of probabilities that sum to 1. It is almost exclusively used in the final layer of multi-class classification models.

**Formula:**
$$\text{Softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}$$

**Key Characteristics:**
- **Range:** (0, 1), with $\sum \text{outputs} = 1.0$.
- **Use Case:** Multi-class classification output layers.
- **Implementation Note:** Uses the "Max Subtraction" trick for numerical stability.

**Usage Example:**
```python
from activations.activations import Softmax
from Tensor import Tensor
import numpy as np

softmax = Softmax()
# Logits for 3 classes
x = Tensor(np.array([2.0, 1.0, 0.1]))
output = softmax(x, dim=0)
print(output.data)
# Output: [0.65900114, 0.24243297, 0.09856589]
print(np.sum(output.data)) # Always 1.0
```

---

## Computational Cost Analysis

Different activations have different computational profiles:

- **ReLU: O(n) comparisons** (Fastest)
- **Sigmoid/Tanh: O(n) exponentials** (3-4× slower than ReLU)
- **GELU: O(n) exponentials + multiplications** (4-5× slower than ReLU)
- **Softmax: O(n) exponentials + O(n) sum + O(n) divisions** (Most expensive)

## Numerical Stability Considerations

Activations can fail catastrophically without proper handling:

- **Sigmoid/Tanh overflow:** Large inputs lead to `exp(x)` exceeding float limits. We use clipping and stable math to avoid this.
- **Softmax overflow:** Large positive logits cause `exp(x)` to return `inf`. We use the "Max Subtraction" trick to keep exponents $\le 0$.
- **ReLU dying neurons:** Monitoring is required to ensure a significant portion of the network doesn't "die" (permanently output 0).

## Gradient Behavior Preview

Understanding gradient characteristics helps diagnose training issues:

- **ReLU:** Constant gradient (1) for $x > 0$, but zero for $x < 0$. No vanishing gradients, but neurons can die.
- **Sigmoid/Tanh:** Gradients "vanish" (approach zero) for large absolute inputs, making deep networks hard to train.
- **GELU:** Smooth gradient everywhere, avoiding the sharp discontinuity of ReLU while maintaining similar benefits.
- **Softmax:** Gradients are coupled across the entire dimension (Jacobian matrix), making the backward pass more complex.

---

## Selection Guide Summary

| Activation | Typical Layer | Primary Benefit |
| :--- | :--- | :--- |
| **ReLU** | Hidden (CNN/MLP) | Speed and sparsity |
| **GELU** | Hidden (Transformers) | State-of-the-art performance |
| **Tanh** | Hidden (RNN) | Zero-centered outputs |
| **Sigmoid** | Output (Binary Class) | Probability mapping (0,1) |
| **Softmax** | Output (Multi-class) | Probability distribution (sums to 1) |
