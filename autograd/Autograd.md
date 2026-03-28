# Autograd in NanoTorch

## Introduction

Automatic differentiation (autograd) is the system that allows neural networks to compute gradients automatically. It tracks operations and uses the chain rule to calculate derivatives, so you don't have to compute them manually.

## Why Autograd Exists

Training a neural network requires gradient descent:

1. Forward pass: compute the output and loss `L`.
2. Compute gradients `dL/dW` for all learnable parameters.
3. Update parameters to minimize the loss.

Manual gradient computation is infeasible for complex models with millions of parameters. Autograd solves this problem using **computational graphs** and **reverse-mode automatic differentiation (backpropagation)**.

## How Autograd Works

Autograd operates by recording operations during the forward pass and applying the chain rule during the backward pass.

### Forward Pass

* Operations on tensors produce new tensors.
* If a tensor requires a gradient, a `_grad_fn` is attached, which stores the operation and its inputs.
* This builds a computation graph that encodes dependencies between tensors.

### Backward Pass

* Initiated by calling `tensor.backward()` on a scalar output.
* The gradient is propagated recursively through `_grad_fn` using local gradient rules.
* Gradients are accumulated in `.grad` of leaf tensors (those with `requires_grad=True`).
* Broadcasting is handled by summing gradients over broadcasted axes.

### Core Components in `autograd/autograd.py`

1. **Function Class**

   * Base class for differentiable operations.
   * Stores `saved_tensors` needed for backward computation.
   * Each subclass implements `apply(grad_output)` to return gradients for the saved inputs.

2. **Backward Subclasses**

   * `AddBackward`, `MulBackward`, `DivBackward` for arithmetic.
   * `MatMulBackward`, `TransposeBackward`, `ReshapeBackward` for matrix operations.
   * `SliceBackward`, `SumBackward` for indexing and reductions.
   * `ReLUBackward`, `SigmoidBackward`, `SoftmaxBckward`, `MSEBackward`, `BCEBackward`, `CrossEntropyBackward` for activations and losses.

3. **enable_autograd()**

   * Patches `Tensor` operations to build the computation graph.
   * Adds `_grad_fn` and sets `requires_grad=True` when needed.
   * Installs `backward()` and `zero_grad()` methods.

4. **Tensor.backward()**

   * Initializes upstream gradient (1 for scalars if not provided).
   * Accumulates gradients in `self.grad`.
   * Handles broadcasting.
   * Recursively calls backward on saved inputs using `_grad_fn.apply()`.

## Mathematical Intuition (Chain Rule)

For a composite function `f(g(x))`:

```
df/dx = (df/dg) * (dg/dx)
```

### Example: L = (x * y + 5)^2

```
x = 2, y = 3
z = x*y = 6
w = z+5 = 11
L = w^2 = 121

Backward Pass:
∂L/∂x = ∂L/∂w * ∂w/∂z * ∂z/∂x = 2*w * 1 * y = 66
∂L/∂y = ∂L/∂w * ∂w/∂z * ∂z/∂y = 2*w * 1 * x = 44
```

Gradient flow:

```
∇x=66 ←──┐
           ├──[×]←── ∇z=22 ←──[+]←── ∇w=22 ←── [²]←── ∇L=1
∇y=44 ←──┘
```

### Recap of Calculus Essentials

* **Derivative of addition:** d(u+v)/dx = du/dx + dv/dx
* **Derivative of multiplication (product rule):** d(u*v)/dx = u*dv/dx + v*du/dx
* **Derivative of division (quotient rule):** d(u/v)/dx = (v*du/dx - u*dv/dx)/v^2
* **Chain rule for composition:** d(f(g(x)))/dx = f'(g(x)) * g'(x)
* **Common function derivatives:**

  * d(x^n)/dx = n*x^(n-1)
  * d(exp(x))/dx = exp(x)
  * d(log(x))/dx = 1/x
  * d(sin(x))/dx = cos(x)
  * d(cos(x))/dx = -sin(x)

### Tiny Autograd Example

```python
x = Tensor([2.0], requires_grad=True)
y = x * 3
z = y + 1
z.backward()
print(x.grad)  # Output: 3
```

Forward attaches `_grad_fn` to `y` and `z`, backward propagates the gradient using the chain rule.

## Key Insights

* Autograd is essentially **local derivatives composed via the chain rule**.
* Leaf tensors accumulate gradients; intermediate tensors provide backward rules.
* Broadcasting and reshaping are reversed during backward to match input shapes.
* Scalability is achieved by reverse-mode differentiation, efficient for scalar outputs and many parameters.

## Current Limitations

* Recursion drives backward propagation; no explicit topological sort.
* Only operations with `_Backward` classes are supported.
* Consistency in `_grad_fn` attachment is required for custom operations.

## Where to Look in Code

* `autograd/autograd.py`

  * `Function` class and `*Backward` subclasses
  * `enable_autograd()` for patching tensors
  * `Tensor.backward()` logic for gradient accumulation and propagation
