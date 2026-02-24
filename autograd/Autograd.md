"""
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
## Enhancing Tensor with Autograd Capabilities

Here we will inhance the existing Tensor class to use these gradient functions and build computation graphs automatically.

**Computation Graph Formation:**
```
Before Autograd:             After Autograd:
  x → operation → y           x → [Function] → y
                                     ↓
                               Stores operation
                               for backward pass
```

**The Enhancement Strategy:**
1. **Add backward() method** - Triggers gradient computation
2. **Enhanced operations** - Replace simple operations with gradient-tracking versions
3. **Track computation graphs** - each tensor remembers how it was created.
4. **Maintain compatibility** - All existing code continues to work

**Critical Desing Decision:**
We enhance the EXISTING Tensor class rather than create a new one.
This means that:
- All previous modules continuw working unchanged
- No imports changes needed
- Gradients are "opt-in" via required_grad=True
- No confusion between Tensor types

"""
### The enable_autograd() Function

This function is the key that brings gradients to life.
It enhances the existing Tensor class with autograd capabilities by :

1. **Monkey-patching operations** - Replaces `__add__`, `__mul__`, etc. with gradient-aware versions
2. **Adding backward() method** - Implements reverse-mode automatic differentiation
3. **Maintaining compatibility** - All existing code continues to work unchanged

**The Pattern:**
```
Original: x + y → simple addition
Enhanced: x + y → addition + gradient tracking (if requires_grad=True)
```

"""


