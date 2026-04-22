# Introduction to Attention Module

## Recent Attention Update

- Added a FlashAttention-style tiled backend for lower-memory exact attention on CPU/NumPy.
- Added educational FlashAttention-2 and FlashAttention-3 style variants that extend the same exact-attention idea with more aggressive block scheduling.
- Added grouped-query attention (GQA), multi-query attention (MQA), and multi-latent attention (MLA) variants to show modern key/value sharing and compression strategies.
- Added sparse attention, linear attention, and paged attention to cover local sparsity, associative attention, and KV-cache paging ideas used in long-context systems.
- The classic `MultiHeadAttention` path still exists, and the new variants reuse the same projection and head-merging ideas with different query/key/value layouts.

This modules gives some basic intuition about attention mechanism that allows models to focus on relevant parts of the input when processing sequences.

## Attention Mechanism Intuition
- Instead of processing words strictly left to right like how it was initially done in RNNs(Recurrent Neural Networks), attention let's every word look at every other word and decide what matters.

-We can imagine attention as a libarary research system whereby we have:
-  **Query(Q)**: "I need information about quantum mechanics"
-  **Keys (K)**: Index cards describing each book's content
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
- We obtain the **Query**, **Key** and **Value** matrices by calculating the dot product of our input embedding matrrx **X** with **WQ(weight matrix for Query)**,**WK(weight matrix for Key)** and **WV (weight matrix for Value)** respectively.
![Alt text](https://github.com/Cohegen/Nanotorch/blob/main/assets/self-attention-matrix-calculation.png)
### The Attention Process 
- After obtaining the query,key and value matrices, we need to compute the attention scores , then scale them by dividing the attention scores matrix by square root of **d_model** and finally normalize them by softmax function.
- The entire process is highlighted below:
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
- The diagram below represents the calculation of attention scores, in this scenario, we are calculating the attention scores of the first row of the attention score matrix where we're multiplying first row of **Q(query matrix)** with columns of **K(key matrix)**.
- To find the full attention score matrix we just follow the rules of matrix-multiplication **matmul** as defined in Linear Algebra.
![Alt text](https://github.com/Cohegen/Nanotorch/blob/main/assets/transformer_self_attention_score.png)

- After this the attention scores in the attention scores matrix are normalized by **d_model(size of embedding vector of individual tokens)**.
- Then they are normalized by the Softmax function.
- After scaling and normalization the we obtain a new matrix called attention weights matrix.
- Then we calculate the dot product of the attention weights and Values Matrices to obtain context aware embedding matrix.
- The diagram below represents the above process.
![Alt text](https://github.com/Cohegen/Nanotorch/blob/main/assets/self-attention-matrix-calculation-2.png)
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

- The total **time complexity** is **O(n² × d)** per attention head. 
- The **memory complexity** is **O(n²)** for storing the attention weight matrix. -This quadratic scaling in sequence length is attention's blessing (global connectivity) and curse (memory/compute limits).
- The  complexity taking only comparisons in account is only **n²** .
- In the Attention matrix in the next section, each there's 16 token comparisons.
- The token **The** is compared 4 times to itself and other tokens, if we add up all token comparisons we obtain 16 token comparisons.
- Given an input sequence length 4 the total token comparison is 16 hence we make a conclusion, given an input of size **n** total comparisons are **n²**.
- By including the cost of the vector dot product, the complexity of attention becomes **O(n²d)**.
- This is because computing the attention matrix involves multiplying an **n × d** matrix with a **d × n** matrix.
- Each element of the resulting **n × n** matrix is computed using a dot product between a row vector and a column vector of length **d**.
- Since a dot product requires **d multiplications**, each of the **n² elements** costs **O(d)**.
- Therefore, the total complexity becomes:**O(n²) × O(d) = **O(n²d)**

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
![Alt text](https://github.com/Cohegen/Nanotorch/blob/main/assets/causal_mask.png)


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

![Alt text](https://github.com/Cohegen/Nanotorch/blob/main/assets/transformer_attention_heads_qkv.png)

![Alt text](https://github.com/Cohegen/Nanotorch/blob/main/assets/transformer_attention_heads_z.png)


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
![Alt text](https://github.com/Cohegen/Nanotorch/blob/main/assets/transformer_multi-headed_self-attention-recap.png)
### Why Multiple Heads Are Powerful

Each head can specialize in different patterns:
- **Head 1**: Short-range syntax ("the cat" → subject-article relationship)
- **Head 2**: Long-range coreference ("John...he" → pronoun resolution)
- **Head 3**: Semantic similarity ("dog" ↔ "pet" connections)
- **Head 4**: Positional patterns (attending to specific distances)

This parallelization allows the model to attend to different representation subspaces simultaneously.

## Modern Attention Variants

As models became larger, researchers noticed that the original multi-head attention formula was elegant but expensive.

Two main bottlenecks kept appearing:

- **Memory bottleneck**: the full attention score matrix has shape `(seq_len × seq_len)` for every head.
- **KV bottleneck**: every head keeps its own keys and values, which increases cache size during generation.

Modern attention variants keep the useful parts of multi-head attention while reducing one of those bottlenecks.

### FlashAttention

FlashAttention does **not** change the mathematics of attention.  
Instead, it changes **how the computation is scheduled** so we avoid storing the full attention matrix in memory.

The standard path conceptually looks like this:

```
Standard Attention:
QK^T → full score matrix → softmax → full weight matrix → weightsV
```

That means we often materialize a large matrix of shape:

```
(batch, heads, seq_len, seq_len)
```

FlashAttention uses a **tiled / blockwise** strategy:

```
FlashAttention:
Take a small block of K,V
→ compute local scores for that block
→ update running softmax statistics
→ accumulate partial output
→ move to next block
```

### Why this is powerful

- We still compute **exact attention**, not an approximation.
- We avoid keeping the entire score matrix in memory at once.
- This dramatically reduces peak memory pressure for long sequences.

### Intuition

Imagine reading a huge textbook to answer one question:

- **Standard attention**: photocopy the full book, highlight every sentence, then decide what matters.
- **FlashAttention**: read one chapter at a time, keep track of the best evidence so far, and build the final answer incrementally.

### Mathematical idea

Instead of computing:

```
softmax(QK^T / √d) V
```

all at once, FlashAttention computes it in blocks while maintaining:

- a **running maximum** for numerical stability
- a **running normalization sum**
- a **running partial output**

This is why the implementation in `attention.py` uses:

- `running_max`
- `running_sum`
- block-by-block accumulation over key/value chunks

### Shape picture

```
Input:
Q: (B, H, Tq, D)
K: (B, H, Tk, D)
V: (B, H, Tk, D)

Block processing:
K_block: (B, H, block_size, D)
V_block: (B, H, block_size, D)
```

### FlashAttention-2

FlashAttention-2 keeps the same **exact attention math** as FlashAttention, but improves the scheduling strategy even further.

The key idea is:

```
FlashAttention v1:
stream over K,V blocks

FlashAttention-2:
tile both Q blocks and K,V blocks
```

So instead of treating the whole query matrix as one giant object, we also split queries into chunks:

```
Q_block: (B, H, q_block, D)
K_block: (B, H, k_block, D)
V_block: (B, H, k_block, D)
```

### Why this matters

- Better work partitioning for long sequences
- More cache-friendly execution
- More balanced computation between query work and key/value streaming

### Intuition

Imagine a huge exam with many students and many reference books:

- **FlashAttention**: take all students together and process one set of books at a time.
- **FlashAttention-2**: divide students into smaller groups and books into smaller groups, then process group-by-group.

That gives better scheduling flexibility.

### FlashAttention-3

FlashAttention-3 continues the same philosophy:

- exact attention
- streaming computation
- even stronger emphasis on hardware-friendly scheduling

In this educational CPU implementation, FlashAttention-3 is represented as:

```
query blocking + page-aligned KV streaming
```

So we combine:

- query chunks
- page-sized key/value chunks

This mirrors the idea that modern high-performance kernels are not just about the formula, but also about **how data is moved through memory hierarchies**.

### A simple picture

```
FlashAttention-3:
Take a query block
→ stream page 1 of KV
→ update output
→ stream page 2 of KV
→ update output
→ ...
→ move to next query block
```

### Why this is useful

- It keeps the exact result of standard attention.
- It encourages page-aligned memory movement.
- It bridges the idea of FlashAttention and paged KV-cache systems.

### Grouped-Query Attention (GQA)

Grouped-query attention keeps **many query heads**, but uses **fewer key/value heads**.

Standard multi-head attention uses:

```
Q heads = H
K heads = H
V heads = H
```

GQA changes that to:

```
Q heads = H
K heads = G
V heads = G

where G < H
```

Then each key/value head is **shared** by a group of query heads.

### Intuition

Suppose we have 8 query heads but only 2 KV heads:

```
Query Heads:   Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8
KV Heads:      K1 K1 K1 K1 K2 K2 K2 K2
               V1 V1 V1 V1 V2 V2 V2 V2
```

That means several query specialists consult the **same memory bank**.

### Why GQA matters

- It keeps more expressive query diversity than MQA.
- It reduces KV-cache size compared with full multi-head attention.
- It is a strong middle ground between quality and efficiency.

### Shape example

If `embed_dim = 512`, `num_heads = 8`, and `num_kv_heads = 2`:

```
head_dim = 512 / 8 = 64

Queries: 8 heads × 64 dims
Keys:    2 heads × 64 dims
Values:  2 heads × 64 dims
```

In our implementation, the smaller K/V heads are projected first, then repeated so the attention computation can still run in a head-aligned format.

### Multi-Query Attention (MQA)

Multi-query attention is the most extreme form of KV sharing:

```
Q heads = H
K heads = 1
V heads = 1
```

So every query head has its own perspective, but all of them read from the **same key/value memory**.

### Intuition

Think of a team of analysts:

- Each analyst asks different questions about the data.
- But they all look at the same archive cabinet.

That is MQA:

- diverse query heads
- one shared K head
- one shared V head

### Why MQA is useful

- Very small KV cache during autoregressive generation
- Faster decoding in large language models
- Simpler memory layout than full multi-head attention

### Tradeoff

The cost savings are excellent, but sharing a single K/V head can reduce representational richness compared with full MHA or GQA.

### Multi-Latent Attention (MLA)

Multi-latent attention adds a **compression step** before building keys and values.

The idea is:

```
Input X
→ project into a smaller latent space
→ build K and V from that latent representation
→ keep Q in the original head space
```

So instead of computing K and V directly from the full embedding, we first create a smaller bottleneck:

```
X → latent → K,V
```

### Why this helps

- The model can compress memory information before attention.
- K/V projections become cheaper than projecting directly from the full hidden size.
- It opens a path to lower-memory and lower-compute attention while still keeping multiple query heads.

### Intuition

Imagine summarizing a long document before putting it into a retrieval system:

- **Standard attention**: store every detail directly.
- **MLA**: first compress the document into a more compact latent summary, then attend over that.

### Shape example

If:

```
embed_dim = 512
latent_dim = 128
num_heads = 8
num_kv_heads = 2
```

then:

```
Input X:      (B, T, 512)
Latent:       (B, T, 128)
Queries:      from 512-d input
Keys/Values:  from 128-d latent
```

This is exactly why the implementation introduces a `kv_down_proj` layer before the K/V projections.

### Sparse Attention

Sparse attention changes the **connectivity pattern** of attention itself.

Instead of allowing every token to compare with every other token:

```
Full attention:
token i attends to all tokens 1...n
```

sparse attention restricts each token to a smaller neighborhood:

```
Sparse attention:
token i attends only to a local window
plus optional global tokens
```

### Sliding-window intuition

For a window size of 2:

```
Token 5 can attend to:
Token 3, Token 4, Token 5, Token 6, Token 7
```

instead of all tokens in the sequence.

### Why sparse attention matters

- Lower effective attention workload on long sequences
- Strong inductive bias for local structure
- Useful for documents, speech, vision patches, and long-context models

### Global tokens

Some tokens are especially important, such as:

- `[CLS]` in classification models
- summary tokens
- memory tokens
- special sink tokens

Sparse attention often allows those tokens to be visible to everyone:

```
Local window + global tokens
```

That is exactly why our sparse attention function supports:

- `window_size`
- `global_indices`

### Linear Attention

Linear attention attacks a different bottleneck from sparse attention.

Instead of changing *who attends to whom*, it changes *how the attention computation is algebraically arranged*.

Standard attention does:

```
softmax(QK^T) V
```

which creates the explicit `n × n` score matrix.

Linear attention uses a positive feature map:

```
phi(Q), phi(K)
```

and rewrites the computation so it can use associativity:

```
phi(Q) [phi(K)^T V]
```

instead of first building:

```
phi(Q) phi(K)^T
```

### Why this is powerful

- We avoid constructing the full attention matrix explicitly.
- Time and memory can scale closer to **O(n d²)** or **O(n d)** depending on the formulation rather than **O(n² d)**.
- This makes it attractive for very long contexts.

### Intuition

Standard attention asks:

- "How much should every token attend to every other token?"

Linear attention asks:

- "Can we summarize the keys and values first, then let each query read from that summary?"

That is why it feels more like **summary retrieval** than full pairwise comparison.

### Feature map in this implementation

Our implementation uses a positive map inspired by ELU:

```
phi(x) = ELU(x) + 1
```

The `+1` is important because it keeps the feature map positive, which helps the associative normalization work properly.

### Paged Attention

Paged attention is closely tied to **KV-cache management during generation**.

When a large language model generates tokens autoregressively, it stores keys and values from previous steps in a cache.
If the sequence becomes long, storing that cache as one big contiguous block can become awkward and memory-fragmented.

Paged attention solves this by storing KV memory in **fixed-size pages**:

```
KV Cache:
Page 1 | Page 2 | Page 3 | Page 4 | ...
```

Then attention reads those pages one by one.

### Intuition

Think of a huge memory archive split into filing cabinets:

- **Contiguous KV cache**: one giant cabinet
- **Paged KV cache**: many small labeled drawers

The model can retrieve only the drawers it needs while keeping memory management simpler.

### Why paged attention matters

- Better memory reuse
- Better support for long-running generation servers
- Cleaner interaction with dynamic batching and variable sequence lengths

### In this implementation

Our educational version models paged attention as:

```
exact attention
+ streamed KV access
+ fixed-size pages
```

So the score math is still exact, but KV is processed page by page rather than as one monolithic block.

## How the Variants Compare

```
Variant         Query Heads      KV Heads        Main Goal
MHA             many             many            maximum flexibility
FlashAttention  many             many            lower memory scheduling
FlashAttention2 many             many            query+KV tiled scheduling
FlashAttention3 many             many            page-aligned flash scheduling
GQA             many             fewer           smaller KV cache
MQA             many             one             minimal KV cache
MLA             many             fewer/latent    compressed KV construction
Sparse          many             many            local connectivity
Linear          many             many            associative computation
Paged           many             many            paged KV-cache access
```

### A systems view

- **FlashAttention** optimizes the **attention computation path**.
- **FlashAttention-2 / 3** push the same idea further with better tiling and page-aware scheduling.
- **GQA / MQA** optimize the **key-value storage path**.
- **MLA** optimizes the **representation path** by compressing before K/V construction.
- **Sparse attention** optimizes the **connectivity pattern**.
- **Linear attention** optimizes the **algebraic computation pattern**.
- **Paged attention** optimizes the **memory layout of KV cache access**.

So these variants are solving related but different problems:

1. `FlashAttention` says: "How do we compute attention without memory blow-up?"
2. `FlashAttention-2 / 3` say: "How do we schedule that exact computation even better?"
3. `GQA` says: "How do we keep multiple query heads without storing full KV heads?"
4. `MQA` says: "How far can we push KV sharing?"
5. `MLA` says: "Can we build keys and values from a smaller latent bottleneck?"
6. `Sparse attention` says: "Can we restrict who talks to whom?"
7. `Linear attention` says: "Can we avoid the full pairwise score matrix algebraically?"
8. `Paged attention` says: "How do we keep long KV caches manageable in memory?"

### Why Attention has complexity of O(n^2)
- After running the programs with prefix "analyze" in this directory, you will make a certain conclusion.
- That is that attention has a time complexity of O(n^2).
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
- Multiplying them gives
```
A 
```
- Meaning, every token compatres itself with every other token.
- So implying that total comparisions are:
```
nxn = n^2
```
- So, computation is O(n^2 d) and memory O(n^2).

### Why Memory Explodes Faster than Expected
-Attention matrix stores float values.
-Most LLMs use float16 (2bytes).
-Memory needed is:
```
 n^2 x 2 bytes

```
- Example:
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
- So per layer:
```
2MB X 2 (softmax + intermediate) = 4MB
```
- We can estimate that:
```
4MB x96 layer = 0.375GB
```

### Why Time also EXplodes
- Computational Cost:
```
QK^T
```
- Multiplying:
```
(nxd).(dxn)

```
- Complexity:

```
O(n^2 d)
```
- When the sequence length doubles:
```
n-> 2n
```
- Computation becomes:
```
(2n^2)^2 =4n^2
```
- So compute comes 4x larger















