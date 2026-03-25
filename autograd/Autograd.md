## Autograd (Automatic Differentiation) in nano-torch

### Why autograd exists (background)
Training a neural network uses gradient descent:

1. Run a forward pass to compute a loss `L`.
2. Compute gradients `dL/dW` for all learnable parameters.
3. Update parameters to reduce the loss.

Step (2) is the hard part: for anything more complex than a toy model, manually deriving and coding gradients for every operation becomes error-prone and infeasible.

Autograd fixes this by using the **chain rule** and a **computational graph**: instead of writing gradients by hand, we let the system build a graph of operations during the forward pass, then automatically backpropagate gradients through that graph.

### The essence: reverse-mode automatic differentiation
This repo implements the reverse-mode flavor of autograd (the same idea used by “backprop”):

- **Forward pass**: operations create new `Tensor` values; if a value needs gradients, we record *how to compute its backward*.
- **Backward pass**: starting from the loss, gradients flow backward through the recorded operations using the chain rule.

Reverse-mode is efficient when you have a single scalar output loss and many parameters.

### Under the hood in this repo: `autograd/autograd.py`
`autograd/autograd.py` builds a lightweight autograd engine by:

1. Defining a `Function` base class and many `*Backward` subclasses.
2. Installing (“patching”) gradient-tracking behavior into the `Tensor` class via `enable_autograd()`.
3. Implementing the reverse-mode propagation logic inside `Tensor.backward()`.

#### 1. `Function`: “how to compute backward”
Each differentiable operation corresponds to a `Function` subclass.

- `Function` stores `saved_tensors`: the inputs needed to compute gradients later.
- Each subclass implements `apply(self, grad_output)` and returns gradients for the saved inputs.

Examples of backward rules implemented in this file:
- `AddBackward`, `MulBackward`, `SubBackward`, `DivBackward`
- `MatMulBackward`, `TransposeBackward`, `ReshapeBackward`
- `SliceBackward`, `SumBackward`
- activations / losses gradients like `ReLUBackward`, `SigmoidBackward`, `SoftmaxBckward`, `MSEBackward`, `BCEBackward`, `CrossEntropyBackward`

#### 2. `enable_autograd()`: building the computation graph
Calling `enable_autograd()` turns on gradient tracking by patching `Tensor` methods so they:

1. Compute the normal forward result (using the existing numpy-based `Tensor` operations).
2. If any input has `requires_grad=True`, mark the result as `requires_grad=True`.
3. Attach a `_grad_fn` object onto the result tensor (e.g. `AddBackward`, `MulBackward`, etc.).

Concretely, `enable_autograd()` replaces key operations on `Tensor`, including:
- arithmetic: `__add__`, `__sub__`, `__mul__`, `__truediv__`
- indexing: `__getitem__`
- linear algebra: `matmul`
- shape ops: `transpose`, `reshape`, `sum`
- engine methods: `backward`, `zero_grad`

So the “graph” is encoded directly in the tensors via:
- `tensor._grad_fn`: the operation that produced the tensor
- `tensor._grad_fn.saved_tensors`: the inputs that operation needs for backward

#### 3. `Tensor.backward()`: gradient propagation engine
The actual backprop logic is implemented inside `Tensor.backward()` (installed in `enable_autograd()`).

Conceptually, it does:

1. Initialize the starting gradient:
   - If `gradient` is not provided and the output is scalar (`self.data.size == 1`), it uses `ones_like`.
   - Otherwise, for non-scalars it requires an explicit upstream gradient.
2. Accumulate gradients in `self.grad`:
   - If `self.grad` is `None`, initialize zeros.
   - Then add: `self.grad += gradient`.
3. Handle broadcasting:
   - If shapes differ due to broadcasting, it sums axes and reshapes so the gradient matches the tensor’s shape.
4. Propagate through the graph:
   - If the tensor has `_grad_fn`, call `self._grad_fn.apply(gradient)` to get gradients for saved inputs.
   - Recursively call `backward()` on saved tensors that require gradients.

That recursive loop is how the chain rule is applied across the entire recorded computation.

### How the code’s concepts map to theory
Below is how the important pieces in `autograd.py` demonstrate autograd’s core ideas.

#### Basic arithmetic ops
- `AddBackward`: passes `grad_output` to both inputs.
- `MulBackward`: product rule:
  - `grad_a = grad_output * b`
  - `grad_b = grad_output * a`
- `SubBackward`: conceptually subtraction should route gradients as:
  - `grad_a = grad_output`
  - `grad_b = -grad_output`
  In `autograd.py`, `SubBackward` is currently incomplete (it computes `grad_a` but does not return properly), so subtraction backward may not behave correctly until that is fixed.
- `DivBackward`: quotient rule:
  - `grad_a = grad_output / b`
  - `grad_b = -grad_output * a / (b^2)`

#### Matrix ops and shape ops
- `MatMulBackward`: matrix calculus with transposes (including batched matmul via last-two-dim transpose).
- `TransposeBackward`: gradient is the transpose of the upstream gradient.
- `ReshapeBackward`: gradient is reshaped back to the original shape.

These show a general rule: operations that only reorder or reshape values transform gradients in the inverse way.

#### Indexing and reduction
- `SliceBackward`: scatters `grad_output` back to the original tensor positions (unsliced entries get 0).
- `SumBackward`: broadcasts the scalar gradient back to all summed elements.

This demonstrates how autograd handles operations that change tensor size.

#### Activations and losses
Inside `enable_autograd()`, the file attempts to patch activation/loss `forward` methods so they:
- compute the normal forward result
- attach the correct `_grad_fn` for backward (e.g. `ReLUBackward`, `SoftmaxBckward`, `MSEBackward`, `BCEBackward`)

The backward rule implementations show standard gradient patterns:
- `SoftmaxBckward`: gradients are coupled across classes because softmax normalization mixes them.
- `BCEBackward`: clamps probabilities to avoid unstable `log`/division behavior.
- `CrossEntropyBackward`: uses the closed-form gradient with softmax + one-hot targets.

### Tiny “connect-the-dots” example
If you do:

```python
x = Tensor([2.0], requires_grad=True)
y = x * 3
z = y + 1
z.backward()
```

Then:
- forward attaches `_grad_fn` to `y` and `z`
- backward starts from the scalar output gradient
- the engine calls each `_grad_fn.apply(...)` and recursively computes gradients for parents, accumulating into `.grad`

### The deeper mental model: local derivatives compose
Autograd doesn’t “differentiate the whole program” at once. Instead, it relies on a local rule at each operation.

For any operation `y = op(x1, x2, ...)`, the backward rule implements:

- a **local gradient** that tells how changing `y` would change each `xi`
- then multiplies (conceptually) by the incoming gradient `grad_output`

Mathematically, the chain rule is often expressed as a sequence of **vector-Jacobian products (VJPs)**:

`grad_x = grad_output @ (dy/dx)`

That’s exactly what `Function.apply(grad_output)` is doing: it receives the upstream gradient and returns gradients for the saved inputs.

So the engine is basically:
- forward: remember what each output depends on (`saved_tensors`)
- backward: for each output, ask its `_grad_fn` to map `grad_output` to input gradients
- repeat recursively until you reach “leaf” tensors

### Leaf tensors, `requires_grad`, and gradient accumulation
In this implementation, gradients exist only for tensors with `requires_grad=True`.

Two important behaviors to understand:

- `Tensor.backward()` exits early if `requires_grad` is false, so it does not compute gradients for that tensor.
- Gradients are accumulated via `self.grad += gradient`, so if a tensor contributes to the loss through multiple paths, those gradient contributions add up.

This accumulation is why training loops typically call `zero_grad()` between batches. In this repo, `zero_grad()` sets `tensor.grad = None` (not zero), so the next backward pass will re-initialize gradients cleanly.

### Broadcasting: “unbroadcast” gradients back to the input shape
Many numpy ops broadcast shapes in the forward pass. During backward, gradients must match the original tensor shape.

This is handled inside `Tensor.backward()`: if `gradient.shape != self.grad.shape`, it repeatedly reduces extra dimensions by summing until rank matches, then sums along axes where the original tensor had dimension `1` (because the forward broadcast duplicated that value).

So the learner should think of it as:
- forward: value gets reused across broadcasted copies
- backward: those copies each “vote” for the gradient, so we sum them back

### The starting gradient: why scalars are special
In reverse-mode, `backward()` always needs an initial upstream gradient for the output it is differentiating.

This repo follows the common convention:
- If you call `backward()` with `gradient=None` and the output is a scalar (`self.data.size == 1`), it initializes the gradient to `1` (implemented as `ones_like`).
- If the output is non-scalar, `backward()` requires you to pass the upstream gradient explicitly, because there is no single “dL/doutput” without more information.

### Backward traversal: recursion instead of an explicit topological sort
Most autograd engines traverse the computation graph in a topologically sorted order.

Here, the engine is simpler:
- each tensor stores a `_grad_fn`
- backward recursively calls `tensor.backward(grad)` on the saved parents

This means the traversal order is driven by the dependency structure stored in `_grad_fn.saved_tensors`.

It’s educational and works well for small graphs, but you should be aware that it’s not a fully general, optimized “real framework” scheduler.

### The `Function.apply()` contract
Each `Function` knows which tensors it depends on (its `saved_tensors`). The backward rule must return a tuple of gradients whose elements correspond 1:1 to those saved tensors, in the same order.

That is why `Tensor.backward()` does:
- `grads = self._grad_fn.apply(gradient)`
- then `for tensor, grad in zip(self._grad_fn.saved_tensors, grads): ...`

### Current limitations / things to be aware of
This autograd engine is intentionally educational and lightweight. Common limitations you may see:
- no full topological sort; recursion drives the backward propagation
- only operations with corresponding `*Backward` rules and tracked forward patches are supported
- attribute naming consistency matters when patching losses/activations (the code should attach backward handlers using the `_grad_fn` convention)

### Where to look in code
- `autograd/autograd.py`
  - `class Function` and each `*Backward` subclass
  - `enable_autograd()` where `Tensor` methods are replaced/installed
  - `Tensor.backward()` logic that accumulates gradients and propagates them

<!--
OLD/UNFINISHED DRAFT CONTENT BELOW

Everything from here to the end of this file is an earlier draft that was left behind.
It is commented out so the rewritten, correct explanation above is the one that renders.
-->
## Autograd (Automatic Differentiation) in nano-torch

### Why autograd exists (the background)
Training a neural network typically uses gradient descent:

1. Run a forward pass to compute a loss `L`.
2. Compute gradients `dL/dW` for all learnable parameters.
3. Update parameters to reduce the loss.

The hard part is step (2). For a complex model with many parameters, manually deriving and implementing gradients for every operation is both error-prone and infeasible.

Autograd solves this by using the **chain rule** plus a **computational graph**:

```
L = f(W3, f(W2, f(W1, x)))
dL/dW1, dL/dW2, dL/dW3  are computed automatically
by chaining local derivatives of each operation.
```

### The essence: reverse-mode AD (backprop)
nano-torch’s autograd implements **reverse-mode automatic differentiation**, the same idea used by backpropagation:

- During the **forward pass**, we record which operations produced each `Tensor` that may require gradients.
- During the **backward pass**, we start from the loss and propagate gradients *backwards* through the recorded graph.

In reverse-mode, a single scalar loss produces gradients for many inputs/parameters efficiently.

### Under the hood in this repo: `autograd/autograd.py`
This file builds a small autograd engine on top of your existing `Tensor` implementation.

#### 1. `Function`: “how to compute backward”
All differentiable operations use a `Function` object.

- `Function` stores `saved_tensors`, i.e., the inputs needed to compute gradients later.
- Each subclass implements `apply(self, grad_output)`, returning gradients for each saved input.

Example subclasses in this repo:
- `AddBackward`
- `MulBackward`
- `DivBackward`
- `MatMulBackward`
- `TransposeBackward`
- `ReshapeBackward`
- `SliceBackward`
- `SumBackward`
- activation/loss gradient rules like `ReLUBackward`, `SigmoidBackward`, `SoftmaxBckward`, `MSEBackward`, `BCEBackward`, `CrossEntropyBackward`

#### 2. `enable_autograd()`: patching `Tensor` to build the graph
`enable_autograd()` is the switch that turns gradient tracking on.

What it does:

1. Adds gradient fields to `Tensor`:
   - `tensor.requires_grad`
   - `tensor.grad` (initialized to `None`)
2. Replaces key `Tensor` methods with “tracked” versions that build computation graph metadata:
   - `Tensor.__add__` -> `tracked_add`
   - `Tensor.__sub__` -> `tracked_sub`
   - `Tensor.__mul__` -> `tracked_mul`
   - `Tensor.__truediv__` -> `tracked_div`
   - `Tensor.__getitem__` -> `tracked_getitem`
   - `Tensor.matmul` -> `tracked_matmul`
   - `Tensor.transpose` -> `tracked_transpose`
   - `Tensor.reshape` -> `tracked_reshape`
   - `Tensor.sum` -> `sum_op`
   - `Tensor.backward` -> engine logic (defined in `enable_autograd` as `backward`)
   - `Tensor.zero_grad` -> resets accumulation
3. When a tracked operation produces a result where `requires_grad` is needed, it attaches a `_grad_fn`:
   - e.g. for addition: `result._grad_fn = AddBackward(self, other)`
   - similarly for multiplication, matmul, etc.

So the “graph” is encoded in the tensors themselves via `._grad_fn` and each function’s `saved_tensors`.

#### 3. `Tensor.backward()`: the actual gradient propagation engine
The backward engine is implemented in `Tensor.backward()` (installed inside `enable_autograd()`).

Conceptually, it does:

1. If `gradient` is not provided:
   - If the tensor is scalar (`self.data.size == 1`), it uses `ones_like` as the starting gradient.
   - Otherwise it raises an error (you must pass the upstream gradient for non-scalars).
2. Accumulate gradients into `self.grad`:
   - If `self.grad` is `None`, initialize it to zeros.
   - Then do `self.grad += gradient`.
3. Handle broadcasting:
   - If shapes differ due to broadcasting in forward, it sums the gradient back to match `self.grad.shape`.
4. If the tensor has a `_grad_fn`:
   - Call `grads = self._grad_fn.apply(gradient)`
   - For each `(saved_tensor, grad)` pair, recursively call `tensor.backward(grad)` if the saved tensor requires gradients.

This is the chain rule working in reverse order.

### How the concepts in `autograd.py` map to the theory
Below are the core operations this repo supports and the key autograd concept each demonstrates.

#### Basic arithmetic ops
- `AddBackward`: passes `grad_output` to both inputs (with shape considerations handled by `Tensor.backward`).
- `MulBackward`: uses the product rule:
  - `grad_a = grad_output * b`
  - `grad_b = grad_output * a`
- `SubBackward`: forwards `grad_output` to the left input and negates it for the right input (the file currently returns only `grad_a` in the snippet; the intended concept is sign handling).
- `DivBackward`: uses the quotient rule:
  - `grad_a = grad_output / b`
  - `grad_b = -grad_output * a / (b^2)`

These all rely on the same engine mechanics:
- forward stores references to inputs in `saved_tensors`
- backward produces gradients for those inputs

#### Matrix operations and dimension reshaping
- `MatMulBackward` computes gradients via matrix calculus using transposes:
  - `grad_A = grad_output @ B.T`
  - `grad_B = A.T @ grad_output`
  It also supports batched matmul by transposing only the last two dimensions.
- `TransposeBackward` returns `grad_output` transposed back.
- `ReshapeBackward` reshapes `grad_output` back to the original input shape.

These demonstrate a common autograd pattern:
if the forward pass only rearranges values (transpose/reshape), the backward pass rearranges gradients in the inverse way.

#### Indexing and reduction
- `SliceBackward` scatters `grad_output` back into an array of the original input shape (unsliced positions get 0).
- `SumBackward` broadcasts the scalar gradient back to every element that contributed to the sum.

These demonstrate how autograd handles operations that change tensor size:
- reduction (sum) duplicates gradient back
- slicing/indexing routes gradient only to used elements

#### Activations and losses
Inside `enable_autograd()`, the code attempts to patch:
- activations (`Sigmoid`, `ReLU`, `Softmax`, `GELU`)
- losses (`BinaryCrossEntropyLoss`, `MSELoss`, `CrossEntropyLoss`)

Each patched forward method attaches the corresponding `_grad_fn`, for example:
- `ReLU` forward attaches `ReLUBackward`
- `Sigmoid` forward attaches `SigmoidBackward`
- `Softmax` forward attaches `SoftmaxBckward`
- `MSELoss` forward attaches `MSEBackward`
- `BCE` forward attaches `BCEBackward`

The gradient rules shown in this file illustrate how autograd works for more complex non-linear functions:
- `SoftmaxBckward` computes gradients with the normalization coupling across classes.
- `BCEBackward` clamps probabilities to avoid log/divide numerical issues.
- `CrossEntropyBackward` uses the known closed-form derivative with softmax and one-hot targets.

### Tiny “connect-the-dots” example (what happens when you call backward)
Suppose you do:

```python
x = Tensor([2.0], requires_grad=True)
y = x * 3
z = y + 1
z.backward()
```

Then:
- During forward:
  - `y` gets `requires_grad=True` and `y._grad_fn = MulBackward(x, 3)`
  - `z` gets `requires_grad=True` and `z._grad_fn = AddBackward(y, 1)`
- During backward:
  - `z.backward()` initializes gradient to `1` (scalar output)
  - calls `AddBackward.apply(grad_output)` to get grad for `y`
  - recursively calls `y.backward(grad_for_y)`
  - `MulBackward.apply` produces grad for `x`, stored in `x.grad`

### Current limitations / things to be aware of
This autograd implementation is intentionally lightweight and educational.

Common limitations you may see:
- The “graph” is encoded via recursive `_grad_fn` links without a full topological sort.
- No explicit support for all ops; only the ops with `*_Backward` rules and tracked forward patches work.
- Careful attention to attribute naming is required when patching losses (the file’s CrossEntropy tracking hook should be consistent with `_grad_fn` and requires-grad flags).

### Where to look in code
- `autograd/autograd.py`
  - `class Function` and each `*Backward` subclass
  - `enable_autograd()` where `Tensor` methods are replaced/installed
  - `Tensor.backward()` logic that accumulates gradients and propagates them


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
"""

"""
## Mathematical Intuition of Chain Rule

For composite function : f(g(x)), the derivative is:
```
df/dx = (df/dg) x (dg/dx)

```

###Computational Graph example

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

###Memory Layout During Backpropagation
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

"""

"""
##Implementation phase: Building the Autograd Engine
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
"""

"""
## Function Base Class 
This class is the foundation that makes autograd possible.
Every differentiable operation (addition,multiplication) inherits from this class.

**Importance Function Base Class**
- They remember inputs needed for backward pass.
- They remember gradient computation via apply()\
- They connect from computation graphs
- They enable the chain rule to flow gradients

**The Pattern:**
```
Forward:  inputs → Function.forward() → output
Backward: grad_output → Function.apply() → grad_inputs

This pattern enables the chain rule to flow gradients through complex computations.
```
The code of this class is in **autograd.py**
"""

"""
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

"""

"""
##AddBackward - Gradient Rules for Addition

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
"""

"""
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
"""

"""
##SubBackward
These are gradient rules for subtraction

Subtraction is mathematically simple but important for operations like normalization

**Mathematical Principle:**
``
If z = a - b, then:
∂z/∂a = 1
∂z/∂b = -1

Gradient flow forward to the first operand, but **negated* to the second.
"""

"""
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


"""

"""
##MatmulBackward 
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
"""

""
##SumBackward
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
"""

"""
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

-->
