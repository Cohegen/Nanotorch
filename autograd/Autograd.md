
## Introduction to Automatic Differentiation

Automatic differentiation (autograd) is the magic that makes neural networks learn.
Instead of manually computing gradients for every parameter, autograd tracks operations and automatically compute gradients via the chain rule.

We have so far implemented layers and loss functions. To train a model, we need:

```
Loss = f(W₃, f(W₂, f(W₁, x)))
∂Loss/∂W₁ = ?  ∂Loss/∂W₂ = ?  ∂Loss/∂W₃ = ?
```

Manual gradient computation becomes impossible for complex models with millions of parameters.
The solution is to sue computational graphs
```
Forward Pass:  x → Linear₁ → ReLU → Linear₂ → Loss
Backward Pass: ∇x ← ∇Linear₁ ← ∇ReLU ← ∇Linear₂ ← ∇Loss
```

**Complete Autograd Process Visualization:**
```
┌─ FORWARD PASS ─────────────────────────────────────────────────┐
│                                                                │
│ x ──┬── W₁ ──┐                                                 │
│     │        ├──[Linear₁]──→ z₁ ──[ReLU]──→ a₁ ──┬── W₂ ──┐    │
│     └── b₁ ──┘                               │        ├─→ Loss │
│                                              └── b₂ ──┘        │
│                                                                │
└─ COMPUTATION GRAPH BUILT ──────────────────────────────────────┘
                             │
                             ▼
┌─ BACKWARD PASS ─────────────────────────────────────────────┐
│                                                             │
│∇x ←┬← ∇W₁ ←┐                                                │
│    │       ├←[Linear₁]←─ ∇z₁ ←[ReLU]← ∇a₁ ←┬← ∇W₂ ←┐        │
│    └← ∇b₁ ←┘                             │       ├← ∇Loss   │
│                                          └← ∇b₂ ←┘          │
│                                                             │
└─ GRADIENTS COMPUTED ────────────────────────────────────────┘

Key Insight: Each [operation] stores how to compute its backward pass.
The chain rule automatically flows gradients through the entire graph.
```

Each operation records how to compute its backward pass. The chain rule connects them all.



## Mathematical Intuition of Chain Rule

For composite function : f(g(x)), the derivative is:
```
df/dx = (df/dg) x (dg/dx)

```

### Computational Graph example

```
Simple computation : L = (x*y + 5)**2
Forward Pass:
  x=2 ──┐
        ├──[×]──→ z=6 ──[+5]──→ w=11 ──[²]──→ L=121
  y=3 ──┘

  Backward Pass (Chain Rule in Action):
  ∂L/∂x = ∂L/∂w × ∂w/∂z × ∂z/∂x
        = 2w  ×  1  ×  y
        = 2(11) × 1 × 3 = 66

  ∂L/∂y = ∂L/∂w × ∂w/∂z × ∂z/∂y
        = 2w  ×  1  ×  x
        = 2(11) × 1 × 2 = 44

  Gradient Flow Visualization:
  ∇x=66 ←──┐
           ├──[×]←── ∇z=22 ←──[+]←── ∇w=22 ←──[²]←── ∇L=1
  ∇y=44 ←──┘
```

### Memory Layout During Backpropagation
```
Computation Graph Memory Structure:
┌─────────────────────────────────────────────────────────┐
│ Forward Pass (stored for backward)                      │
├─────────────────────────────────────────────────────────┤
│ Node 1: x=2 (leaf, requires_grad=True) │ grad: None→66  │
│ Node 2: y=3 (leaf, requires_grad=True) │ grad: None→44  │
│ Node 3: z=x*y (MulFunction)            │ grad: None→22  │
│         saved: (x=2, y=3)              │ inputs: [x,y]  │
│ Node 4: w=z+5 (AddFunction)            │ grad: None→22  │
│         saved: (z=6, 5)                │ inputs: [z]    │
│ Node 5: L=w² (PowFunction)             │ grad: 1        │
│         saved: (w=11)                  │ inputs: [w]    │
└─────────────────────────────────────────────────────────┘

Memory Cost: 2× parameters (data + gradients) + graph overhead
```


## Implementation phase: Building the Autograd Engine
We will enhance the existing Tensor class and create a supporting infrastructure.

### The function Architecture

Every differentiable operation needs two things:
1. **Forward pass** : Compute the result
2. **Backward pass** : computes the gradient for inputs

```
Function Class Design:
┌─────────────────────────────────────┐
│ Function (Base Class)               │
├─────────────────────────────────────┤
│ • saved_tensors    ← Store data     │
│ • apply()          ← Compute grads  │
└─────────────────────────────────────┘
          ↑
    ┌─────┴─────┬─────────┬──────────┐
    │           │         │          │
┌───▼────┐ ┌────▼───┐ ┌───▼────┐ ┌───▼────┐
│  Add   │ │  Mul   │ │ Matmul │ │  Sum   │
│Backward│ │Backward│ │Backward│ │Backward│
└────────┘ └────────┘ └────────┘ └────────┘
```
Each operation inherits from Fuction and implements specific gradients rules.

## Function Base Class 
This class is the foundation that makes autograd possible.
Every differentiable operation (addition,multiplication) inherits from this class.

**Importance Function Base Class**
- They remember inputs needed for backward pass.
- They remember gradient computation via apply()
- They connect from computation graphs
- They enable the chain rule to flow gradients

**The Pattern:**
```
Forward:  inputs → Function.forward() → output
Backward: grad_output → Function.apply() → grad_inputs

This pattern enables the chain rule to flow gradients through complex computations.
```
The code of this class is in **autograd.py**


### Operation Functions / implement Gradients rules
Here we will implement specific operations that compute gradients correctly.
Each operation has mathematical rules for how gradients flow backward.

**Gradient Flow Visualization:**
```
Addition (z = a + b):
    ∂z/∂a = 1    ∂z/∂b = 1

    a ──┐           grad_a ←──┐
        ├─[+]─→ z          ├─[+]←── grad_z
    b ──┘           grad_b ←──┘

Multiplication (z = a * b):
    ∂z/∂a = b    ∂z/∂b = a

    a ──┐           grad_a = grad_z * b
        ├─[×]─→ z
    b ──┘           grad_b = grad_z * a

Matrix Multiplication (Z = A @ B):
    ∂Z/∂A = grad_Z @ B.T
    ∂Z/∂B = A.T @ grad_Z

    A ──┐           grad_A = grad_Z @ B.T
        ├─[@]─→ Z
    B ──┘           grad_B = A.T @ grad_Z
```
Each operation stores the inputs it needs for computing gradients.




## AddBackward - Gradient Rules for Addition

Addition is the simplest gradiet operation: gradients flow unchanged to both inputs.

**Mathematical Principle:**
```
If z = a + b, then:
∂z/∂a = 1  (gradient of z w.r.t. a)
∂z/∂b = 1  (gradient of z w.r.t. b)

By chain rule:
∂Loss/∂a = ∂Loss/∂z × ∂z/∂a = grad_output × 1 = grad_output
∂Loss/∂b = ∂Loss/∂z × ∂z/∂b = grad_output × 1 = grad_output
```

**BroadCasting Challenge:**
When tensors have different shapes, Numpy broadcasts automatically in forward pass, but we must "unbroadcast" gradients in backward pass to match original shapes.

Add backward is in **autograd.py**

### MulBackward 
These are gradient rules for Element-wise multiplication

Element-wise multiplication follows the product rule of calculus.

**Mathematical Principle:**

```
If z = a * b (element-wise), then:
∂z/∂a = b  (gradient w.r.t. a equals the other input)
∂z/∂b = a  (gradient w.r.t. b equals the other input)

By chain rule:
∂Loss/∂a = grad_output * b
∂Loss/∂b = grad_output * a
```
**Visual Example:**
```
Forward:  a=[2,3] * b=[4,5] = z=[8,15]
Backward: grad_z=[1,1]
          grad_a = grad_z * b = [1,1] * [4,5] = [4,5]
          grad_b = grad_z * a = [1,1] * [2,3] = [2,3]
```



## SubBackward
These are gradient rules for subtraction

Subtraction is mathematically simple but important for operations like normalization

**Mathematical Principle:**
```
If z = a - b, then:
∂z/∂a = 1
∂z/∂b = -1

Gradient flow forward to the first operand, but **negated** to the second.
```

### DivBackward
They are gradient rules for division

Division requires the quotient rule from calculus

**Mathematical Principle:**

```
If z = a / b, then:
∂z/∂a = 1/b
∂z/∂b = -a/b²
```
**Quotient Rule:** For z = f/g, dz = (g·df - f·dg)/g²



## MatmulBackward 
Gradient Rules for Matrix multiplication

Matrix multiplication has more complex gradient rules based on matrix calculus.

**Mathematical Principle:**
```
If Z = A @ B (matrix multiplication), then:
∂Z/∂A = grad_Z @ B.T
∂Z/∂B = A.T @ grad_Z
```

**Why These Rules Work:**
```
For element Z[i,j] = Σ_k A[i,k] * B[k,j]
∂Z[i,j]/∂A[i,k] = B[k,j]  ← This gives us grad_Z @ B.T
∂Z[i,j]/∂B[k,j] = A[i,k]  ← This gives us A.T @ grad_Z
```
We transpose the matrices so as to ensure ∂Z/∂A and ∂Z/∂B have the same shapes as A and B respectively.

**Dimension Analysis:**
```
Forward:  A(m×k) @ B(k×n) = Z(m×n)
Backward: grad_Z(m×n) @ B.T(n×k) = grad_A(m×k) ✓
          A.T(k×m) @ grad_Z(m×n) = grad_B(k×n) ✓
```

## SumBackward
Sum operations reduce tensor dimensions, so gradients must be broadcast back.

**Mathematical Principle:**
```
If z = sum(a), then ∂z/∂a[i] = 1 for all i
Gradient is broadcasted from scalar result back to input shape.
```

**Gradient Broadcasting Examples:**
```
Case 1: Full sum
  Forward:  a=[1,2,3] → sum() → z=6 (scalar)
  Backward: grad_z=1 → broadcast → grad_a=[1,1,1]

Case 2: Axis sum
  Forward:  a=[[1,2],[3,4]] → sum(axis=0) → z=[4,6]
  Backward: grad_z=[1,1] → broadcast → grad_a=[[1,1],[1,1]]
```

## The Heart of Autograd: The `backward()` Method

The `backward()` method is the engine that drives the learning process. It implements **Reverse-Mode Automatic Differentiation**, a two-step process:

1.  **Forward Pass**: As you perform operations, a "Computation Graph" is built. Each `Tensor` resulting from an operation stores a reference to a `_grad_fn` (a `Function` object like `AddBackward`).
2.  **Backward Pass**: When you call `loss.backward()`, the engine traverses this graph in reverse order (from the output back to the inputs), applying the chain rule at each step.

### How `backward()` works internally:
```python
def backward(self, gradient=None):
    # 1. Initialize gradient for the output (usually 1.0 for scalars)
    if gradient is None:
        gradient = np.ones_like(self.data)

    # 2. Accumulate the gradient in self.grad
    if self.grad is None:
        self.grad = np.zeros_like(self.data)
    self.grad += gradient

    # 3. Propagate to parents
    if hasattr(self, '_grad_fn'):
        # Apply the specific gradient rule for this operation
        grads = self._grad_fn.apply(gradient)
        
        # Recursively call backward on each input tensor
        for tensor, grad in zip(self._grad_fn.saved_tensors, grads):
            if isinstance(tensor, Tensor) and tensor.requires_grad:
                tensor.backward(grad)
```

## Managing Gradients: `requires_grad` and `zero_grad()`

### `requires_grad=True`
Not all tensors need gradients (e.g., input data or constants). We use `requires_grad=True` to tell the engine to start tracking operations for a specific tensor.
- **Leaf Tensors**: Tensors you create (like weights `W`).
- **Non-Leaf Tensors**: Tensors created by operations (like `z = x @ W`). They inherit `requires_grad=True` if any of their inputs require it.

### `zero_grad()`
In training loops, gradients are **accumulated** (added) into the `.grad` attribute every time `backward()` is called. This is useful for some advanced architectures, but normally, we want to start fresh for each batch.
```python
# Standard Training Pattern
for epoch in range(epochs):
    for x, y in dataloader:
        optimizer.zero_grad()  # Reset gradients to None/Zero
        loss = model(x).compute_loss(y)
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update weights
```

## Advanced Gradient Operations

Our autograd engine supports complex operations beyond basic arithmetic:

### 1. Reshape and Transpose
Moving data around doesn't change its values, but we must "move the gradients back" the same way.
- **Reshape**: If you reshaped `(4, 3)` to `(12,)`, the backward pass reshapes the gradient from `(12,)` back to `(4, 3)`.
- **Transpose**: The gradient of a transpose is simply the transpose of the gradient.

### 2. Slicing and Indexing (`SliceBackward`)
When you take a slice `y = x[0:5]`, only those 5 elements contribute to the output.
- **Backward**: Gradients flow back to those specific 5 positions in `x`, while all other positions get a gradient of `0`.

### 3. Embeddings (`EmbeddingBackward`)
Embeddings are like a massive lookup table.
- **Forward**: Pick rows `[1, 5, 2]` from the weight matrix.
- **Backward**: Accumulate gradients into rows `1, 5, and 2` of the weight matrix. If the same index is picked multiple times (e.g., `[1, 1, 2]`), the gradients for index `1` are **summed**.

## Handling Broadcasting in Backward Pass

Broadcasting allows operations on tensors of different shapes (e.g., `(10, 5) + (5,)`). During the backward pass, we must "un-broadcast" the gradient to match the original smaller shape.
```python
# If gradient shape is (10, 5) but input was (5,)
# We sum across the broadcasted dimension (axis 0)
summed_grad = grad.sum(axis=0) # Result shape: (5,)
```

## Activation and Loss Function Gradients

Autograd isn't limited to `+` and `*`. We've integrated it directly into our activation functions:

| Function | Forward | Backward Derivative |
| :--- | :--- | :--- |
| **ReLU** | `max(0, x)` | `1 if x > 0 else 0` |
| **Sigmoid** | `1 / (1 + e^-x)` | `σ(x) * (1 - σ(x))` |
| **Softmax** | `e^x / Σe^x` | `softmax * (grad - sum(grad * softmax))` |
| **MSE Loss** | `mean((p-y)²)` | `2 * (p - y) / N` |

## Practical Example: A Simple Neuron

Here is how all these concepts come together to train a single neuron:

```python
import numpy as np
from autograd.autograd import enable_autograd
from Tensor import Tensor

enable_autograd()

# 1. Setup inputs and weights
x = Tensor([2.0, 3.0])
w = Tensor([0.5, -0.5], requires_grad=True)
b = Tensor([0.1], requires_grad=True)
target = Tensor([1.0])

# 2. Forward Pass (Graph built automatically)
z = (x * w).sum() + b
prediction = z.sigmoid() # Assuming sigmoid is patched

# 3. Compute Loss
loss = ((prediction - target) ** 2)

# 4. Backward Pass
loss.backward()

# 5. Check Gradients
print(f"Weight grads: {w.grad}")
print(f"Bias grad: {b.grad}")

# 6. Update (Manual Optimization)
lr = 0.01
w.data -= lr * w.grad
b.data -= lr * b.grad

# 7. Reset for next step
w.zero_grad()
b.zero_grad()
```
