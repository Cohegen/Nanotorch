# Introduction Acceleration Module

## Prerequisites
- This module assummes that you covered the module below:
  - Tensor Module
  - Autograd Module
  - Convolutions Module

## Recap on the Performance Challenge
- Neural networks often underutilize hardware due to:
    - Sequential operations (no parallelism)
    - Poor memory access patterns (cache misses)
    - Mssing SIMD (Single Instruction, Multiple Data) opportunities
    - Seperate operations (memory bandwidth waste)
- This module tries to fix the above issues with vectorization and kernel fusion,achieving 2-5x speedups.

## The Two Enemies of Performance
- Modern neural networks face two fundemental bottlenecks that limit their speed:

**1. Compute Bound Operations:**
```
CPU/GPU Cores: [====BUSY====] [====BUSY====] [====BUSY====]
Memory Bus:    [---idle---] [---idle---] [---idle---]

When: Matrix multiplication, convolutions
Solution: Vectorization, better algorithms
```

**2. Memory Bound Operations:**
```
CPU/GPU Cores: [--idle--] [--idle--] [--idle--]
Memory Bus:    [========SATURATED========]

When: Element-wise operations, small tensors
Solution: Kernel fusion, memory layout optimization
```

### The Roofline Model 
- Every processor has fundemental limits:

![Alt text](https://github.com/Cohegen/Nanotorch/blob/main/assets/roofline_performance.png)

**Key Insight**: Our goal is to understanding where our operations live on this graph so as to optimize effectively.

## Why This Module Matters
- Real-world performance wins due to acceleration are as follows:
     - **2-5× speedup** from vectorization
     - **2-3× throughput** from kernel fusion
     - **10× scaling improvement** for large models


## Vectorization
### The SIMD Revolution
- Modern processors can execute **Single Instruction, Multiple Data** operations:

```
Traditional Loop (Scalar):               SIMD Vectorized:
for i in range(4):        ┌─────┐      ┌─────┬─────┬─────┬─────┐
    c[i] = a[i] + b[i]    │ ALU │  →   │ALU 0│ALU 1│ALU 2│ALU 3│
                          └─────┘      └─────┴─────┴─────┴─────┘
                          1 element     4 elements per cycle
                          per cycle
```

### Memory Access Patterns
- This is the hidden performance killer.
- How is becomes a hidden killer is as shown below:

```
Sequential Access (FAST):
Memory: [A][B][C][D][E][F][G][H]
Access:  ↓  ↓  ↓  ↓  → Cache friendly

Strided Access (SLOWER):
Memory: [A][ ][B][ ][C][ ][D][ ]
Access:  ↓     ↓     ↓     ↓   → Cache misses

Random Access (SLOWEST):
Memory: [A][B][C][D][E][F][G][H]
Access:  ↓     ↑  ↓     ↑       → Cache chaos
```

### Matrix Multiplication
- Matrix multiplication is **perfectly** for vectorization:
```
Matrix A (M×K) × Matrix B (K×N) = Matrix C (M×N)

Computation Pattern:
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ a₁₁ a₁₂ a₁₃ a₁₄ │ × │ b₁₁ b₁₂ b₁₃ b₁₄ │ = │ c₁₁ c₁₂ c₁₃ c₁₄ │
│ a₂₁ a₂₂ a₂₃ a₂₄ │   │ b₂₁ b₂₂ b₂₃ b₂₄ │   │ c₂₁ c₂₂ c₂₃ c₂₄ │
│ a₃₁ a₃₂ a₃₃ a₃₄ │   │ b₃₁ b₃₂ b₃₃ b₃₄ │   │ c₃₁ c₃₂ c₃₃ c₃₄ │
│ a₄₁ a₄₂ a₄₃ a₄₄ │   │ b₄₁ b₄₂ b₄₃ b₄₄ │   │ c₄₁ c₄₂ c₄₃ c₄₄ │
└─────────────────┘   └─────────────────┘   └─────────────────┘

For c₁₁: Row₁ · Column₁ = a₁₁×b₁₁ + a₁₂×b₂₁ + a₁₃×b₃₁ + a₁₄×b₄₁
                                    ↑
                              VECTORIZABLE!
```

**Why vectorization wins:**
- **High arithmetic intensity**: 2N³ FLOPs for N³ data
- **Predictable memory access**: Sequential row/column reads
- **Parallelizable**: Independent dot products
- **Cache-friendly**: Data reuse in inner loops

## Kernel Fusion
- This eliminates memory bottlenecks.

### The Memory Bandwidth Crisis
- Consider this innocent-looking computation : `y = gelu(x * weight + bias)`

**Naive Implementation (Memory Intensive):**
```
Step 1: temp1 = x * weight     → Write 4GB to memory
Step 2: temp2 = temp1 + bias   → Read 4GB, Write 4GB
Step 3: y = gelu(temp2)        → Read 4GB, Write 4GB
                                 Total: 20GB memory traffic!
```

**Fused Implementation (Memory Efficient):**
```
Single Step: y = gelu(x * weight + bias)  → Read 8GB, Write 4GB
                                            Total: 12GB memory traffic!
                                            60% memory bandwidth reduction!
```

### Understanding GELU: The Smooth Activation.
- GELU (Gaussian Error Linear Unit) is used in transformers because it's **smooth** (differentiable everywhere).

## **ADD ACTIVATION FUNCTIONS COMPARED HERE**

**GELU Formula**: `GELU(x) = x * Φ(x)` where Φ is the standard normal CDF

**Fast Approximation** : `GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`

### Kernel Fusion Strategy

```
Unfused Operations:                    Fused Operation:
┌─────────────────┐                   ┌────────────────────┐
│ x³ computation  │ → temp1           │                    │
└─────────────────┘                   │                    │
┌─────────────────┐                   │                    │
│ polynomial part │ → temp2           │   All operations   │
└─────────────────┘                   │   combined in      │
┌─────────────────┐                   │   single kernel    │
│ tanh computation│ → temp3           │                    │
└─────────────────┘                   │                    │
┌─────────────────┐                   │                    │
│ final multiply  │ → result          │                    │
└─────────────────┘                   └────────────────────┘

5 memory round-trips                   1 memory round-trip
```

## Cache-Aware Matrix Multiplication
- For large matrices that don't fit in the cache, we neeed **tiling** (alos called blocking).
- This breaks the computation into cache-sized chunks for better performance.

### Why Cache Awareness Matters
- Modern processors have  a memory hierarchy:
```
L1 Cache:   32-64 KB   (fastest, 1-4 cycles)
L2 Cache:   256 KB-1MB (fast, 10-20 cycles)
L3 Cache:   8-32 MB    (moderate, 40-75 cycles)
Main RAM:   8-64 GB    (slow, 100-300 cycles)
```
- When matrices are larger than cache, we get **cache missses** that slow us down dramatically.
- Tiling keeps working set in cache for maximum reuse.

