# Tensors: The Foundation of Deep Learning

## Recent Performance Update

- `Tensor.matmul()` now routes matrix multiplication through NumPy's `np.matmul` for all non-scalar cases.
- This removes the old Python nested-loop path for 2D matrix multiply and pushes the work into BLAS-backed vectorized kernels.
- Practical effect: linear layers, attention score products, and any batched tensor matmul now benefit from the faster backend path automatically.

A **Tensor** is the fundamental data structure used in machine learning and deep learning. At its core, it is a multi-dimensional array of numbers that serves as the building block for all data representation and transformation in neural networks.

---

## 1. What is a Tensor?

Tensors are a generalization of vectors and matrices to higher dimensions:

*   **0D Tensor (Scalar):** A single number (e.g., `5`).
*   **1D Tensor (Vector):** An array of numbers (e.g., `[1, 2, 3]`).
*   **2D Tensor (Matrix):** A table of numbers (e.g., `[[1, 2], [3, 4]]`).
*   **3D Tensor:** A cube of numbers (e.g., used for sequences or small images).
*   **ND Tensor:** Arrays with $N$ dimensions (e.g., 4D tensors for batches of color images: `[Batch, Channels, Height, Width]`).

In our implementation, the `Tensor` class wraps a **NumPy** array to provide a high-level interface for these multi-dimensional structures.

---

## 2. Why Tensors in Machine Learning?

Tensors are used in ML/DL for several critical reasons:

1.  **Uniform Data Representation:** Everything is a tensor. Images (pixels), text (embeddings), audio (spectrograms), and video (frames) are all converted into tensors before being processed by a model.
2.  **Vectorized Computation:** Modern hardware (CPUs and GPUs) is optimized for "Single Instruction, Multiple Data" (SIMD). Tensors allow us to perform operations on entire blocks of data simultaneously rather than looping through individual numbers, leading to massive speedups.
3.  **Mathematical Language:** Neural networks are essentially sequences of mathematical transformations (matrix multiplications, additions, etc.). Tensors provide the natural language to express these operations.

---

## 3. Functionality in `tensor.py`

The `Tensor` class in this project implements the core operations required for a deep learning framework:

### Core Properties
- **`data`**: The underlying NumPy array storing the values.
- **`shape`**: The dimensions of the tensor (e.g., `(3, 224, 224)`).
- **`size`**: The total number of elements in the tensor.
- **`dtype`**: The data type (fixed to `float32` for standard ML precision).

### Arithmetic Operations
Supports element-wise operations with **Broadcasting**:
- **Addition (`+`)**: `a + b`
- **Subtraction (`-`)**: `a - b`
- **Multiplication (`*`)**: `a * b` (Element-wise)
- **Division (`/`)**: `a / b`

### Linear Algebra
- **`matmul` / `@`**: Performs matrix multiplication between two tensors. This is the "workhorse" of neural networks, used in every linear layer.

### Shape Manipulation
- **`reshape`**: Changes the dimensions of a tensor without changing its data (e.g., flattening a 28x28 image into a 784 vector).
- **`transpose`**: Swaps dimensions, commonly used to flip axes in matrix math.

### Reduction Operations
- **`sum()`**: Calculates the sum of elements along specified axes.
- **`mean()`**: Calculates the average.
- **`max()`**: Finds the maximum value.

---

## 4. Usage Examples

### Creating Tensors
```python
from Tensor import Tensor
import numpy as np

# Scalar
s = Tensor(5.0)

# 2D Matrix
m = Tensor([[1, 2], [3, 4]])
print(m.shape) # (2, 2)
```

### Arithmetic and Broadcasting
```python
a = Tensor([1, 2, 3])
b = Tensor([10, 20, 30])

# Element-wise addition
c = a + b # Tensor([11, 22, 33])

# Broadcasting (adding a scalar to a vector)
d = a + 5 # Tensor([6, 7, 8])
```

### Matrix Multiplication
```python
x = Tensor([[1, 2], [3, 4]])
y = Tensor([[5, 6], [7, 8]])

# Using the @ operator
z = x @ y
# Result: [[19, 22], [43, 50]]
```

### Reshaping
```python
a = Tensor(np.arange(6)) # [0, 1, 2, 3, 4, 5]
b = a.reshape(2, 3)
# Result: [[0, 1, 2], [3, 4, 5]]
```
