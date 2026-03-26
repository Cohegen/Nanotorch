# Introduction the Compression Module.
## Prerequisites
- This module assumes that you gone through the following modules:
    - profiling
    - quantization

- In this module the following concepts will be covered:
    - Pruning (magnitude and structured)
    - Knowledge distillation
    - low-rank approximation

- After completing this module we will understand how compressed model maintain accuracy while using dramatically less storage and memory.

## Introduction to Model Compression Concepts
- Imagine a scenario where you have a massive library with milions of books, but you only reference 10 % of them regularly.
- Model compression is like creating a curated collection that keeps the essential knowledge while dramatically reducing  storage space.

- Model compreession reduces the size and computational requirements of neural networks while preserving their intelligence.
- It act as the bridge between powerful research models and practical deployment.

## Why Compression Matters in ML systems

**The Storage Challenge:**
- Modern language models: 100GB+ (GPT-3 scale)
- Mobile devices: <1GB available for models
- Edge devices: <100MB realistic limits
- Network bandwidth: Slow downloads kill user experience

**The Speed Challenge:**
- Research models: Designed for accuracy, not efficiency
- Production needs: Sub-second response times
- Battery life: Energy consumption matters for mobile
- Cost scaling: Inference costs grow with model size

### The Compression Landscape

```
┌───────────────────────────────────────────────────────────────────────┐
│                         COMPRESSION METHODS                           │
├───────────────────────────────────────────────────────────────────────┤
│  WEIGHT-BASED                       │  ARCHITECTURE-BASED             │
│  ┌────────────────────────────────┐ │  ┌────────────────────────────┐ │
│  │ Magnitude Pruning              │ │  │ Knowledge Distillation     │ │
│  │ • Remove small weights         │ │  │ • Teacher → Student        │ │
│  │ • 90% sparsity achievable      │ │  │ • 10x size reduction       │ │
│  │                                │ │  │                            │ │
│  │ Structured Pruning             │ │  │ Neural Architecture        │ │
│  │ • Remove entire channels       │ │  │ Search (NAS)               │ │
│  │ • Hardware-friendly            │ │  │ • Automated design         │ │
│  │                                │ │  │                            │ │
│  │ Low-Rank Approximation         │ │  │ Early Exit                 │ │
│  │ • Matrix factorization         │ │  │ • Adaptive compute         │ │
│  │ • SVD decomposition            │ │  │                            │ │
│  └────────────────────────────────┘ │  └────────────────────────────┘ │
└───────────────────────────────────────────────────────────────────────┘
```

- We can think of compression like optimizing a recipe, we want to keep the essential ingredients that create the flavor while removing anything that doesn't contribute to the final dish.

## Foundations the Mathematical Background

- Understanding the mathematics behind compression helps us choose the right technique for each situation and predict their effects on model performance.

### Magnitude-Based Pruning
- The core insigght: small weights contribute little to the final prediction.
- Magnitude pruning removes weights based on their absolute values.

```
Mathematical Foundation:
For weight w_ij in layer l:
    If |w_ij| < threshold_l → w_ij = 0

Threshold Selection:
- Global: One threshold for entire model
- Layer-wise: Different threshold per layer
- Percentile-based: Remove bottom k% of weights

Sparsity Calculation:
    Sparsity = (Zero weights / Total weights) × 100%
```

### Structured Pruning (Hardware friendly Compression)

- Unlike magnitude pruning which creates scattered zeros, structured pruning removes  entire computational units (neurons,channels,attention heads)

```
Channel Importance Metrics:

Method 1: L2 Norm
    Importance(channel_i) = ||W[:,i]||₂ = √(Σⱼ W²ⱼᵢ)

Method 2: Gradient-based
    Importance(channel_i) = |∂Loss/∂W[:,i]|

Method 3: Activation-based
    Importance(channel_i) = E[|activations_i|]

Pruning Decision:
    Remove bottom k% of channels based on importance ranking
```

### Knowledge Distillation (Learning From the Masters)

- Knowledge distillation transfers knowledge from a large **"teacher"** to a smaller **"student"** model.
- The student learns not just the correct answes, but the teacher's reasoning process.

```
Distillation Loss Function:
    L_total = α × L_soft + (1-α) × L_hard

Where:
    L_soft = KL_divergence(σ(z_s/T), σ(z_t/T))  # Soft targets
    L_hard = CrossEntropy(σ(z_s), y_true)        # Hard targets

    σ(z/T) = Softmax with temperature T
    z_s = Student logits, z_t = Teacher logits
    α = Balance parameter (typically 0.7)
    T = Temperature parameter (typically 3-5)

Temperature Effect:
    T=1: Standard softmax (sharp probabilities)
    T>1: Softer distributions (reveals teacher's uncertainty)
```

### Low-Rank Approximation (Matrix Compression)

- Large weight matrices often have redundancy that can be capture with lower-rank approximations using Singular Value Decompostion.

```
SVD Decomposition:
    W_{m×n} = U_{m×k} × Σ_{k×k} × V^T_{k×n}

Parameter Reduction:
    Original: m × n parameters
    Compressed: (m × k) + k + (k × n) = k(m + n + 1) parameters

    Compression achieved when: k < mn/(m+n+1)

Reconstruction Error:
    ||W - W_approx||_F = √(Σᵢ₌ₖ₊₁ʳ σᵢ²)

    Where σᵢ are singular values, r = rank(W)
```

## Sparsity Meaurement
- Before compressing models, we need to understand how dense they are.
- Sparsity tells us what percentage of weights are zero (or effectively zero).

### Understanding Sparsity
- Sparsity is like measuring how much of a packing lot is empty.
- A 90 % sparse model means 90% of its weights are zero, i.e only 10% of the **"parking spaces"** are occupied.

```
Dense Matrix (0% sparse):           Sparse Matrix (75% sparse):
┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐    ┌─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┐
│ 2.1 1.3 0.8 1.9 2.4 1.1 0.7 │    │ 2.1 0.0 0.0 1.9 0.0 0.0 0.0 │
│ 1.5 2.8 1.2 0.9 1.6 2.2 1.4 │    │ 0.0 2.8 0.0 0.0 0.0 2.2 0.0 │
│ 0.6 1.7 2.5 1.1 0.8 1.3 2.0 │    │ 0.0 0.0 2.5 0.0 0.0 0.0 2.0 │
│ 1.9 1.0 1.6 2.3 1.8 0.9 1.2 │    │ 1.9 0.0 0.0 2.3 0.0 0.0 0.0 │
└─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘    └─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─┘
All weights active                   Only 7/28 weights active
Storage: 28 values                   Storage: 7 values + indices
```
- The function intended to measure sparsity is present in **compression.py**.


## Magnitude-Based Pruning
- Magnitude pruning is the simplest and most intuitive compression technique.
- It's based on the observation that weights with small magnitude contribute little to the model's output.

### How Magnitude Pruning works
- Think of magniude like editing a document whereby we can remove words that don't significantly change the meaning.
- In neural networks, we remove weights that don't siginificantly affect predictions.

```
Magnitude Pruning Process:

Step 1: Collect All Weights (20 total across 2 layers)
┌──────────────────────────────────────────────────┐
│ Layer 1: [2.1, 0.08, -1.8, 0.04, 3.2,             │
│           -0.02, 1.5, -0.03, 2.8, 0.06]           │
│ Layer 2: [0.7, 2.4, -0.05, 1.9, 0.01,             │
│           -1.3, 0.03, 2.1, -0.07, 0.09]           │
└──────────────────────────────────────────────────┘
                    ↓
Step 2: Calculate Magnitudes
┌──────────────────────────────────────────────────┐
│ Sorted: [0.01, 0.02, 0.03, 0.03, 0.04, 0.05,     │
│          0.06, 0.07, 0.08, 0.09, 0.7, 1.3,       │
│          1.5, 1.8, 1.9, 2.1, 2.1, 2.4, 2.8, 3.2] │
└──────────────────────────────────────────────────┘
                    ↓
Step 3: Find Threshold (e.g., 50th percentile)
┌──────────────────────────────────────────────────┐
│ 20 values → 50th pctile between 10th and 11th    │ Threshold ≈ 0.4
│ Values ≤ 0.4: ten small weights get zeroed        │ (50% of weights removed)
└──────────────────────────────────────────────────┘
                    ↓
Step 4: Apply Pruning Mask
┌──────────────────────────────────────────────────┐
│ Layer 1: [2.1, 0.0, -1.8, 0.0, 3.2,              │
│           0.0, 1.5, 0.0, 2.8, 0.0]               │ 50% weights → 0
│ Layer 2: [0.7, 2.4, 0.0, 1.9, 0.0,               │ 50% preserved
│           -1.3, 0.0, 2.1, 0.0, 0.0]              │
└──────────────────────────────────────────────────┘

Memory Impact:
- Dense storage: 20 values × 4 bytes = 80 bytes
- Sparse storage: 10 values + 10 indices = 80 bytes (no savings!)
- At 90% sparsity: 2 values + 2 indices = 16 bytes (80% savings)
```

### Why Global Thresholding Works

Global thresholding treats the entire model as one big collection of weights, finding a single threshold that achieves the target sparsity across all layers.

**Advantages:**
- Simple to implement and understand
- Preserves overall model capacity
- Works well for uniform network architectures

**Disadvantages:**
- May over-prune some layers, under-prune others
- Doesn't account for layer-specific importance
- Can hurt performance if layers have very different weight distributions
"""
- Implementation of magnitude pruning is present in **compression.py**

## Structured Pruning
- While magnitude pruning creates scattered zeros throughout the network, structured pruning removes entire computational units (channels,neurons,heads).
- This creates sparsity patterns that modern hardware can actually accelerated.

### Why Structured Pruning matters

- Think of the difference between removing random words from a paragraph versus removing entire sentences.
- Structured pruning removes entire **"sentences"** (channels) rather than random **"words"** (individual weights).

```
Unstructured vs Structured Sparsity:

UNSTRUCTURED (Magnitude Pruning):
┌─────────────────────────────────────────────┐
│ Channel 0: [2.1, 0.0, 1.8, 0.0, 3.2]        │ ← Sparse weights
│ Channel 1: [0.0, 2.8, 0.0, 2.1, 0.0]        │ ← Sparse weights
│ Channel 2: [1.5, 0.0, 2.4, 0.0, 1.9]        │ ← Sparse weights
│ Channel 3: [0.0, 1.7, 0.0, 2.0, 0.0]        │ ← Sparse weights
└─────────────────────────────────────────────┘
Issues: Irregular memory access, no hardware speedup

STRUCTURED (Channel Pruning):
┌─────────────────────────────────────────────┐
│ Channel 0: [2.1, 1.3, 1.8, 0.9, 3.2]        │ ← Fully preserved
│ Channel 1: [0.0, 0.0, 0.0, 0.0, 0.0]        │ ← Fully removed
│ Channel 2: [1.5, 2.2, 2.4, 1.1, 1.9]        │ ← Fully preserved
│ Channel 3: [0.0, 0.0, 0.0, 0.0, 0.0]        │ ← Fully removed
└─────────────────────────────────────────────┘
Benefits: Regular patterns, hardware acceleration possible
```

## Channel Importance Ranking
- How do we decide which channels to remove?
- We rank them by importance using various metrics :

```
Channel Importance Metrics:

Method 1: L2 Norm (Most Common)
    For each output channel i:
    Importance_i = ||W[:, i]||_2 = √(Σⱼ w²ⱼᵢ)

    Intuition: Channels with larger weights have bigger impact

Method 2: Activation-Based
    Importance_i = E[|activation_i|] over dataset

    Intuition: Channels that activate more are more important

Method 3: Gradient-Based
    Importance_i = |∂Loss/∂W[:, i]|

    Intuition: Channels with larger gradients affect loss more

Ranking Process:
    1. Calculate importance for all channels
    2. Sort channels by importance (ascending)
    3. Remove bottom k% (least important)
    4. Zero out entire channels, not individual weights
```

### HardWare Benefits of Structured Sparsity

- Structured sparsity enables real hardware acceleration because:

1. **Memory Coalescing**: Accessing contiguous memory chunks is faster
2. **SIMD Operations**: Can process multiple remaining channels in parallel
3. **No Indexing Overhead**: Don't need to track locations of sparse weights
4. **Cache Efficiency**: Better spatial locality of memory access

## Low-Rank Approximation
- Low-rank approximation discovers that large weight matrices often contain redundant information that can be captures with smaller matrices through mathematical decompostion.

### The Intuition Behind Low-Rank Approximation
- Imagine you're storing a massive spreadsheet where many columns are highly correlated.
- Instead of storing all columns seperately, we can store a few "basis" columns and coefficients for how to combine them to recreate the original data.

```
Low-Rank Decomposition Visualization:

Original Matrix W (large):           Factorized Form (smaller):
┌─────────────────────────┐         ┌──────┐    ┌──────────────┐
│ 2.1  1.3  0.8  1.9  2.4 │         │ 1.1  │    │ 1.9  1.2  0.7│
│ 1.5  2.8  1.2  0.9  1.6 │    ≈    │ 2.4  │ @  │ 0.6  1.2  0.5│
│ 0.6  1.7  2.5  1.1  0.8 │         │ 0.8  │    │ 1.4  2.1  0.9│
│ 1.9  1.0  1.6  2.3  1.8 │         │ 1.6  │    │ 0.5  0.6  1.1│
└─────────────────────────┘         └──────┘    └──────────────┘
    W (4×5) = 20 params           U (4×2)=8  +  V (2×5)=10  = 18 params

Parameter Reduction:
- Original: 4 × 5 = 20 parameters
- Compressed: (4 × 2) + (2 × 5) = 18 parameters
- Compression ratio: 18/20 = 0.9 (10% savings)

For larger matrices, savings become dramatic:
- W (1000×1000): 1M parameters → U (1000×100) + V (100×1000): 200K parameters
- Compression ratio: 0.2 (80% savings)
```

### SVD :The Mathematical Foundation
- Singular Value Decomposition (SVD) finds the optimal low-rank approximation by identifying the most important  **"directions"** in the data:

```
SVD Decomposition:
    W = U × Σ × V^T

Where:
    U: Left singular vectors (input patterns)
    Σ: Singular values (importance weights)
    V^T: Right singular vectors (output patterns)

Truncated SVD (Rank-k approximation):
    W ≈ U[:,:k] × Σ[:k] × V^T[:k,:]

Quality vs Compression Trade-off:
    Higher k → Better approximation, less compression
    Lower k → More compression, worse approximation

Choosing Optimal Rank:
    Method 1: Fixed ratio (k = ratio × min(m,n))
    Method 2: Energy threshold (keep 90% of singular value energy)
    Method 3: Error threshold (reconstruction error < threshold)
```

### When Low-Ranks Works Best 
Low-rank approximation works well when:
- **Matrices are large**: Compression benefits scale with size
- **Data has structure**: Correlated patterns enable compression
- **Moderate accuracy loss acceptable**: Some precision traded for efficiency

It works poorly when:
- **Matrices are already small**: Overhead exceeds benefits
- **Data is random**: No patterns to exploit
- **High precision required**: SVD introduces approximation error

- The code that implements low-rank approximation is present in **compression.py**.

## Knowledge Distillation
- Knowledge distillation is like having an expert teacher simplify complex concepts for a student .
- The large **"teacher"** model shares its knowledge with a smaller **"student"** model, achieving similar performance with far fewer parameters.

### The Teacher-Student Learning Process
- Unlike traditional training where models learn from labels (cat/dog), knowledge distillation uses **"soft"** targets that contain richer information about the teacher's decision making process.

```
Knowledge Distillation Process:

                    TEACHER MODEL (Large)
                    ┌─────────────────────┐
Input Data ────────→│ 100M parameters     │
                    │ 95% accuracy        │
                    │ 500ms inference     │
                    └─────────────────────┘
                             │
                             ↓ Soft Targets
                    ┌─────────────────────┐
                    │  Logits: [2.1, 0.3, │
                    │           0.8, 4.2] │ ← Rich information
                    └─────────────────────┘
                             │
                             ↓ Distillation Loss
                    ┌─────────────────────┐
Input Data ────────→│ STUDENT MODEL       │
Hard Labels ───────→│ 10M parameters      │ ← 10x smaller
                    │ 93% accuracy        │ ← 2% loss
                    │ 50ms inference      │ ← 10x faster
                    └─────────────────────┘

Benefits:
• Size: 10x smaller models
• Speed: 10x faster inference
• Accuracy: Only 2-5% degradation
• Knowledge transfer: Student learns teacher's "reasoning"
```

### Temperature Scaling(Softening Decisions)

- Temperature scaling is a key innovation that makes knowledge distillation effective .
- It **"softens"** the teacher's confidence, revealing uncertainity that helps the student to learn.

```
Temperature Effect on Probability Distributions:

Without Temperature (T=1):           With Temperature (T=3):
Teacher Logits: [1.0, 2.0, 0.5]    Teacher Logits: [1.0, 2.0, 0.5]
                       ↓                               ↓ ÷ 3
Softmax: [0.09, 0.67, 0.24]         Logits/T: [0.33, 0.67, 0.17]
         ^      ^      ^                       ↓
      Low   High   Med              Softmax: [0.21, 0.42, 0.17]
                                             ^      ^      ^
Sharp decisions (hard to learn)           Soft   decisions (easier to learn)

Why Soft Targets Help:
1. Reveal teacher's uncertainty about similar classes
2. Provide richer gradients for student learning
3. Transfer knowledge about class relationships
4. Reduce overfitting to hard labels
```

### Loss Function Design
- The distillation loss balances learning from both the teacher's soft knowledge and the ground truth hard labels:

```
Combined Loss Function:

L_total = α × L_soft + (1-α) × L_hard

Where:
    L_soft = KL_divergence(Student_soft, Teacher_soft)
             │
             └─ Measures how well student mimics teacher

    L_hard = CrossEntropy(Student_predictions, True_labels)
             │
             └─ Ensures student still learns correct answers

Balance Parameter α:
• α = 0.7: Focus mainly on teacher (typical)
• α = 0.9: Almost pure distillation
• α = 0.3: Balance teacher and ground truth
• α = 0.0: Ignore teacher (regular training)

Temperature T:
• T = 1: No softening (standard softmax)
• T = 3-5: Good balance (typical range)
• T = 10+: Very soft (may lose information)
```

## The Compression Pipeline

### Compression Strategy Design
- Real-world compression often combines multiple techniques in sequences, each targeting different types of redundancy:

```
Multi-Stage Compression Pipeline:

Original Model (100MB, 100% accuracy)
         │
         ↓ Stage 1: Magnitude Pruning (remove 80% of small weights)
Sparse Model (20MB, 98% accuracy)
         │
         ↓ Stage 2: Structured Pruning (remove 30% of channels)
Compact Model (14MB, 96% accuracy)
         │
         ↓ Stage 3: Low-Rank Approximation (compress large layers)
Factorized Model (10MB, 95% accuracy)
         │
         ↓ Stage 4: Knowledge Distillation (train smaller architecture)
Student Model (5MB, 93% accuracy)

Final Result: 20x size reduction, 7% accuracy loss
```

### Compression Configuration
- Different deplolyment scenarios require different compression strategies:

```
Deployment Scenarios and Strategies:

MOBILE APP (Aggressive compression needed):
┌─────────────────────────────────────────┐
│ Target: <10MB, <100ms inference         │
│ Strategy:                               │
│ • Magnitude pruning: 95% sparsity       │
│ • Structured pruning: 50% channels      │
│ • Knowledge distillation: 10x reduction │
│ • Quantization: 8-bit weights           │
└─────────────────────────────────────────┘

EDGE DEVICE (Balanced compression):
┌─────────────────────────────────────────┐
│ Target: <50MB, <200ms inference         │
│ Strategy:                               │
│ • Magnitude pruning: 80% sparsity       │
│ • Structured pruning: 30% channels      │
│ • Low-rank: 50% rank reduction          │
│ • Quantization: 16-bit weights          │
└─────────────────────────────────────────┘

CLOUD SERVICE (Minimal compression):
┌─────────────────────────────────────────┐
│ Target: Maintain accuracy, reduce cost  │
│ Strategy:                               │
│ • Magnitude pruning: 50% sparsity       │
│ • Structured pruning: 10% channels      │
│ • Dynamic batching optimization         │
│ • Mixed precision inference             │
└─────────────────────────────────────────┘
```

