# Introduction to Attention Module
This modules gives some basic intuition about attention mechanism that allows models to focus on relevant parts of the input when processing sequences.

## Attention Mechanism Intuition
-Instead of processing words strictly left to right like how it was initially done in RNNs(Recurrent Neural Networks), attention let's every word look at every other word and decide what matters.

-We can imagine attention as a libarary research system whereby we have:
-  **Query(Q)**: "I need information about quantum mechanics"
-**Keys (K)**: Index cards describing each book's content
- **Values(V)**: the actual books on the shelves
- **Attention Process**:finding books whose description matches our query then we retrieve those books.

## Why Attention made a huge impact 
-Before attention, RNNs processed sequences step by step, creating an information bottleneck:

```
RNN Processing (Sequential):
Token 1 → Hidden → Token 2 → Hidden → ... → Final Hidden
         ↓              ↓                      ↓
    Limited Info   Compressed State    All Information Lost
```
-Attention allows direct connections between any two positions:

```
Attention Processing (Parallel):
Token 1 ←─────────→ Token 2 ←─────────→ Token 3 ←─────────→ Token 4
   ↑                   ↑                   ↑                   ↑
   └─────────────── Direct Connections ──────────────────────┘
```
- This enables:
- **1.Long-range dependencies**: since it connects words that are far apart
- **2.Parameter computation**: since there's no parallel dependencies
- **3.Interpretable focus patterns**: since we can see what the model attends to

## The Mathematical Foundation
-Attention computes a weighted sum of values, where weights are determined by the similarity between queries and keys

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

## Attention Mathematics

### The Three Components Visualization
Think of attention like a complex address book lookup

```
Query: "What information do I need?"
┌─────────────────────────────────────┐
│ Q: [0.1, 0.8, 0.3, 0.2]             │ ← Query vector (what we're looking for)
└─────────────────────────────────────┘

Keys: "What information is available at each position?"
┌─────────────────────────────────────┐
│ K₁: [0.2, 0.7, 0.1, 0.4]            │ ← Key 1 (description of position 1)
│ K₂: [0.1, 0.9, 0.2, 0.1]            │ ← Key 2 (description of position 2)
│ K₃: [0.3, 0.1, 0.8, 0.3]            │ ← Key 3 (description of position 3)
│ K₄: [0.4, 0.2, 0.1, 0.9]            │ ← Key 4 (description of position 4)
└─────────────────────────────────────┘

Values: "What actual content can I retrieve?"
┌─────────────────────────────────────┐
│ V₁: [content from position 1]       │ ← Value 1 (actual information)
│ V₂: [content from position 2]       │ ← Value 2 (actual information)
│ V₃: [content from position 3]       │ ← Value 3 (actual information)
│ V₄: [content from position 4]       │ ← Value 4 (actual information)
└─────────────────────────────────────┘
```

### The Attention Process 
```
Step 1: Compute Similarity Scores
Q · K₁ = 0.64    Q · K₂ = 0.81    Q · K₃ = 0.35    Q · K₄ = 0.42
  ↓               ↓               ↓               ↓
Raw similarity scores (higher = more relevant)

Step 2: Scale and Normalize
Scores / √d_k = [0.32, 0.41, 0.18, 0.21]  ← Scale for stability
     ↓
Softmax = [0.20, 0.45, 0.15, 0.20]        ← Convert to probabilities

Step 3: Weighted Combination
Output = 0.20×V₁ + 0.45×V₂ + 0.15×V₃ + 0.20×V₄
```

### Dimensions and Shapes
```
Input Shapes:
Q: (batch_size, seq_len, d_model)  ← Each position has a query
K: (batch_size, seq_len, d_model)  ← Each position has a key
V: (batch_size, seq_len, d_model)  ← Each position has a value

Intermediate Shapes:
QK^T: (batch_size, seq_len, seq_len)  ← Attention matrix (the O(n²) part!)
Weights: (batch_size, seq_len, seq_len)  ← After softmax
Output: (batch_size, seq_len, d_model)  ← Weighted combination of values
```

### Why Attention has a time complexity of O(n**2)
-For sequence length n and embedding dimension d, we compute:
1. **QK^T**: n queries × n keys, each a d-dimensional dot product = O(n² × d) operations
2. **Softmax**: n² weights to normalize = O(n²) operations
3. **Weights×V**: n² weights applied to d-dimensional values = O(n² × d) operations

-The total **time complexity** is **O(n² × d)** per attention head. 
-The **memory complexity** is **O(n²)** for storing the attention weight matrix. -This quadratic scaling in sequence length is attention's blessing (global connectivity) and curse (memory/compute limits).

### The Attention Matrix Visualization
```
Attention Matrix (after softmax):
        The   cat   sat  down
The   [0.30  0.20  0.15  0.35]  ← "The" attends mostly to "down"
cat   [0.10  0.60  0.25  0.05]  ← "cat" focuses on itself and "sat"
sat   [0.05  0.40  0.50  0.05]  ← "sat" attends to "cat" and itself
down  [0.25  0.15  0.10  0.50]  ← "down" focuses on itself and "The"

Each row sums to 1.0 (probability distribution)
```

### Causal Mask
-A **causal mask** is what makes models like GPT autoregressive (predicting future values based on a linear combination of its own previous observations).
- It ensures that when predicting the next token, the model cannot see the future tokens.
- In simple terms: each position can attend to itself and previous positions and never the future positions.
- We do this by setting future positions to -infinity before softmax, which makes their attention weights zero.

```
Causal Mask (4 tokens):       After masking:
+---+---+---+---+            +----+----+----+----+
| 1 | 0 | 0 | 0 |            | s1 |-inf|-inf|-inf|
| 1 | 1 | 0 | 0 |     ->     | s2 | s3 |-inf|-inf|
| 1 | 1 | 1 | 0 |            | s4 | s5 | s6 |-inf|
| 1 | 1 | 1 | 1 |            | s7 | s8 | s9 | s10|
+---+---+---+---+            +----+----+----+----+
```

### Scaled Dot-Product Attention
-Scaled dot-product attention is the core operation inside Transformers.
-It computes how much each token should pay attention to every other token.
- It follows the pipeline below:

```
Pipeline: Q,K -> scores -> scale -> mask -> softmax -> weights @ V -> output
```

## Multi-Head Attention
- Multi-head attention runs multiple attention "heads" in parallel, each learning to focus on different types of relationships.
- We can think of it as having multiple specialists: one for syntax,one for semantics , one for long-range dependencies.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ SINGLE-HEAD vs MULTI-HEAD ATTENTION ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│ SINGLE HEAD ATTENTION (Limited Representation):                         │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ Input (512) → [Linear] → Q,K,V (512) → [Attention] → Output (512)   │ │
│ │                  ↑           ↑            ↑            ↑            │ │
│ │            Single proj  Full dimensions  One head   Limited focus   │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ MULTI-HEAD ATTENTION (Rich Parallel Processing):                        │
│ ┌─────────────────────────────────────────────────────────────────────┐ │
│ │ Input (512)                                                         │ │
│ │      ↓                                                              │ │
│ │ [Q/K/V Projections] → 512 dimensions each                           │ │
│ │      ↓                                                              │ │
│ │ [Split into 8 heads] → 8 × 64 dimensions per head                   │ │
│ │      ↓                                                              │ │
│ │ Head₁: Q₁(64) ⊗ K₁(64) → Attention₁ → Output₁(64)  │ Syntax focus   │ │
│ │ Head₂: Q₂(64) ⊗ K₂(64) → Attention₂ → Output₂(64)  │ Semantic       │ │
│ │ Head₃: Q₃(64) ⊗ K₃(64) → Attention₃ → Output₃(64)  │ Position       │ │
│ │ Head₄: Q₄(64) ⊗ K₄(64) → Attention₄ → Output₄(64)  │ Long-range     │ │
│ │ Head₅: Q₅(64) ⊗ K₅(64) → Attention₅ → Output₅(64)  │ Local deps     │ │
│ │ Head₆: Q₆(64) ⊗ K₆(64) → Attention₆ → Output₆(64)  │ Coreference    │ │
│ │ Head₇: Q₇(64) ⊗ K₇(64) → Attention₇ → Output₇(64)  │ Composition    │ │
│ │ Head₈: Q₈(64) ⊗ K₈(64) → Attention₈ → Output₈(64)  │ Global view    │ │
│ │      ↓                                                              │ │
│ │ [Concatenate] → 8 × 64 = 512 dimensions                             │ │
│ │      ↓                                                              │ │
│ │ [Output Linear] → Final representation (512)                        │ │
│ └─────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│ Key Benefits of Multi-Head:                                             │
│ • Parallel specialization across different relationship types           │
│ • Same total parameters, distributed across multiple focused heads      │
│ • Each head can learn distinct attention patterns                       │
│ • Enables rich, multifaceted understanding of sequences                 │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The Multi-Head Process Detailed

```
Step 1: Project to Q, K, V
Input (512 dims) → Linear → Q, K, V (512 dims each)

Step 2: Split into Heads
Q (512) → Reshape → 8 heads × 64 dims per head
K (512) → Reshape → 8 heads × 64 dims per head
V (512) → Reshape → 8 heads × 64 dims per head

Step 3: Parallel Attention (for each of 8 heads)
Head 1: Q₁(64) attends to K₁(64) → weights₁ → output₁(64)
Head 2: Q₂(64) attends to K₂(64) → weights₂ → output₂(64)
...
Head 8: Q₈(64) attends to K₈(64) → weights₈ → output₈(64)

Step 4: Concatenate and Mix
[output₁ ∥ output₂ ∥ ... ∥ output₈] (512) → Linear → Final(512)
```

### Why Multiple Heads Are Powerful

Each head can specialize in different patterns:
- **Head 1**: Short-range syntax ("the cat" → subject-article relationship)
- **Head 2**: Long-range coreference ("John...he" → pronoun resolution)
- **Head 3**: Semantic similarity ("dog" ↔ "pet" connections)
- **Head 4**: Positional patterns (attending to specific distances)

This parallelization allows the model to attend to different representation subspaces simultaneously.

### Why Attention has complexity of O(n^2)
-After running the programs with prefix "analyze" in this directory, you will make a certain conclusion.
-That is that attention has a time complexity of O(n^2).
- If a sequence has:

```
n = number of tokens
```
the attention score matrix becomes:
```
A = QK^T
```
- Where;
```
Q =queries -> shape n x d_model
K = keys -> shape n x d_model
```
-Multiplying them gives
```
A 
```
-Meaning, every token compatres itself with every other token.
- So implying that total comparisions are:
```
nxn = n^2
```
So, computation is O(n^2 d) and memory O(n^2).

### Why Memory Explodes Faster than Expected
-Attention matrix stores float values.
-Most LLMs use float16 (2bytes).
-Memory needed is:
```
 n^2 x 2 bytes

```
-Example:
```
1024 tokens

1024^2 =1048576

```
- Memory consumed
```
1048576x 2 =2MB
```
- But transformers have multiple heads
Example GPT3
```
heads = 96
layers = 96
```
-So per layer:
```
2MB X 2 (softmax + intermediate) = 4MB
```
- We can estimate that:
```
4MB x96 layer = 0.375GB
```

### Why Time also EXplodes
-Computational Cost:
```
QK^T
```
-Multiplying:
```
(nxd).(dxn)

```
Complexity:

```
O(n^2 d)
```
- When the sequence length doubles:
```
n-> 2n
```
-Computation becomes:
```
(2n^2)^2 =4n^2
```
-So compute comes 4x larger
