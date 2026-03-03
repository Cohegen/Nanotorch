# Introduction to Embedding Module

## What is the essence of Embeddings
- Neural networks operate on dense vectors, but languange of discrete tokens.
- Embeddings are crucial bridge that converts discrete tokens into continous, learnable vector representations that capture semantic meaning.

### The Token-to-Vector Challenge
- Consider the tokens from our tokenizer: [1,42,7].
- How do we turn these discrete indices into meaningful vectors that capture semantic relationships?

```
┌─────────────────────────────────────────────────────────────────┐
│  EMBEDDING PIPELINE: Discrete Tokens → Dense Vectors            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (Token IDs):     [1, 42, 7]                              │
│           │                                                     │
│           ├─ Step 1: Lookup in embedding table                  │
│           │         Each ID → vector of learned features        │
│           │                                                     │
│           ├─ Step 2: Add positional information                 │
│           │         Same word at different positions → different│
│           │                                                     │
│           ├─ Step 3: Create position-aware representations      │
│           │         Ready for attention mechanisms              │
│           │                                                     │
│           └─ Step 4: Enable semantic understanding              │
│                     Similar words → similar vectors             │
│                                                                 │
│  Output (Dense Vectors): [[0.1, 0.4, ...], [0.7, -0.2, ...]]    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The Four-Layer Embedding System
- Modern embedding systems combine multiple components:
**1. Token embeddings**- Learn semantic representation for each vocabulary token
**2. Positional encoding** - Add information about position in sequence
**3. Optional scaling** - Normalize embedding magnitudes
**4 Integration** - Combining everything into position-aware representations

### Why does Embeddings matter?
The choice of embedding strategy dramatically affects:
-**Semantic understanding**- how well the model captures word meaning
-**Memory requirements** - embedding tables can be gigabytes in size
-**Extrapolation** - how well the model handles longer sequences than training.

## Embedding Strategies
- Different embedding approaches make different trade-offs between memory,semantic understanding and computational efficiency.

### Token Embedding Lookup Process
**Approach**: each token ID maps to a learned dense vector

```
┌──────────────────────────────────────────────────────────────┐
│ TOKEN EMBEDDING LOOKUP PROCESS                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Build Embedding Table (vocab_size × embed_dim)      │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Token ID  │  Embedding Vector (learned features)       │  │
│  ├────────────────────────────────────────────────────────┤  │
│  │    0      │  [0.2, -0.1,  0.3, 0.8, ...]  (<UNK>)      │  │
│  │    1      │  [0.1,  0.4, -0.2, 0.6, ...]  ("the")      │  │
│  │   42      │  [0.7, -0.2,  0.1, 0.4, ...]  ("cat")      │  │
│  │    7      │  [-0.3, 0.1,  0.5, 0.2, ...]  ("sat")      │  │
│  │   ...     │             ...                            │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Step 2: Lookup Process (O(1) per token)                     │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Input: Token IDs [1, 42, 7]                           │  │
│  │                                                        │  │
│  │   ID 1  → embedding[1]  → [0.1,  0.4, -0.2, ...]       │  │
│  │   ID 42 → embedding[42] → [0.7, -0.2,  0.1, ...]       │  │
│  │   ID 7  → embedding[7]  → [-0.3, 0.1,  0.5, ...]       │  │
│  │                                                        │  │
│  │  Output: Matrix (3 × embed_dim)                        │  │
│  │  [[0.1,  0.4, -0.2, ...],                              │  │
│  │   [0.7, -0.2,  0.1, ...],                              │  │
│  │   [-0.3, 0.1,  0.5, ...]]                              │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Step 3: Training Updates Embeddings                         │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Gradients flow back to embedding table                │  │
│  │                                                        │  │
│  │  Similar words learn similar vectors:                  │  │
│  │  "cat" and "dog" → closer in embedding space           │  │
│  │  "the" and "a"   → closer in embedding space           │  │
│  │  "sat" and "run" → farther in embedding space          │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Advantages**
- Dense representation i.e every dimension is meaningful
- Learnable i.e captures semantic relationships through training.
- Efficient lookup since it has O(1) time complexity
- Scales to large vocabulary

**Disadvantages**
- It memory extensive since (vocab_sizex embed_dim parameters) grow as inputs increase
- Requires training to develop semantic relationships
- Has fixed vocabulary which in turn make new tokens to need special handling like,representing words which are not in tokens with the <UNK> token.
Showing that it's not present

### Positional Encoding Strategies
- Since ordinary embeddings lack information about the position of the token in the sentence, we need positional information about the positions of individual tokens with a sentence.
- This is where the need of postional embeddings come into place.
- These positional embeddings are concatenate to the token embeddings to produce position-aware embeddings.

```
Position-Aware Embeddings = Token Embeddings + Positional Encoding

Learned Approach:     Fixed Mathematical Approach:
Position 0 → [learned]     Position 0 → [sin/cos pattern]
Position 1 → [learned]     Position 1 → [sin/cos pattern]
Position 2 → [learned]     Position 2 → [sin/cos pattern]
...                        ...
```

**Learned Positional Encoding**
- Trainable position embeddings
- Can learn task-specific patterns
- Limited to maximum training sequence length

**Sinusoidal Positional Encoding**
- Mathematical sine/cosine patterns
- No additional parameters 
- Can extrapolate to longer sequences

### Strategy Comparison

```
Text: "cat sat on mat" → Token IDs: [42, 7, 15, 99]

Token Embeddings:    [vec_42, vec_7, vec_15, vec_99]  # Same vectors anywhere
Position-Aware:      [vec_42+pos_0, vec_7+pos_1, vec_15+pos_2, vec_99+pos_3]
                      ↑ Now "cat" at position 0 ≠ "cat" at position 1
```
- The combination enables transformers to understand both meaning and order



## Gradient Computatition for Embedding Lookup

### Mathematical definition of Embedding Lookup
- Suppose we have:
   *1.Vocabulary size*:V
   *2.Embedding dimension*:d
   *3.Embedding matrix*: E (whose dimensions is V by d)

- Each row is a vecotr for one token.
- If our input token index is: i.
- Then the embedding lookup will be: embedding = E[i]

### Lookup = One-Hot x Embedding matrix
- This is the trick that explains gradients.
- We know that lookup E[i] is mathematically equivalent to x_oneHot * E
- Where, x_oneHot is a vector that has, 1 position at position i and 0 everywhere else
- Example (V=3,token=2)
      x = [0,0,1,0,0]
      xE = E[2]

### Forward pass

- Given that input token i = 3 and embedding matrix E.
- Forward pass:
     * h = E[i]

This vector h goes into the rest of the network.
Loss,L = loss(h).

### Backpropagation.
During backpropagation,we compute:
     dL/dE i.e gradient of loss wrt embedding matrix E

However, only the row that was looked up gets a gradient.

This is because h = E[i] does not depend on any other row of E.

So: dL/dE[j] = 0 for j != i
And: dL/dE[i] = dL/dh

### Concise Example
Suppose our embedding E = [e0,e1,e2,e3,e4]
Input token =2 
Forward: h = e2
backpropagation gives : dL/dh = g

This means the gradient update becomes:
- Row2 gets gradient of g
- All other rows get zero.
So the resultant gradient vector will look like this:
dL/dE = [
    0
    0
    g
    0
    0
]

### Learnable Positional Encoding
- Trainable position embeddings that can learn position-specific patterns.
- This approach treats each position as a learnable parameter, similar to token embeddings.

```
Learned Position Embedding Process:

Step 1: Initialize Position Embedding Table
┌───────────────────────────────────────────────────────────────┐
│ Position  │  Learnable Vector (trainable parameters)          │
├───────────────────────────────────────────────────────────────┤
│    0      │ [0.1, -0.2,  0.4, ...]  ← learns "start" patterns │
│    1      │ [0.3,  0.1, -0.1, ...]  ← learns "second" patterns│
│    2      │ [-0.1, 0.5,  0.2, ...]  ← learns "third" patterns │
│   ...     │        ...                                        │
│  511      │ [0.4, -0.3,  0.1, ...]  ← learns "late" patterns  │
└───────────────────────────────────────────────────────────────┘

Step 2: Add to Token Embeddings
Input: ["The", "cat", "sat"] → Token IDs: [1, 42, 7]

Token embeddings:     Position embeddings:     Combined:
[1]  → [0.1, 0.4, ...] + [0.1, -0.2, ...] = [0.2, 0.2, ...]
[42] → [0.7, -0.2, ...] + [0.3, 0.1, ...] = [1.0, -0.1, ...]
[7]  → [-0.3, 0.1, ...] + [-0.1, 0.5, ...] = [-0.4, 0.6, ...]

Result: Position-aware embeddings that can learn task-specific patterns!
```
### Sinusodial Positional Encoding

Mathematical postion encoding that creates unique signatures for each position using trigonometric functions.
This approach requires no additional parameters and can extrapolate to sequences longer than seen during training.

```
┌───────────────────────────────────────────────────────────────────────┐
│ SINUSOIDAL POSITION ENCODING: Mathematical Position Signatures        │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ MATHEMATICAL FORMULA:                                                 │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ PE(pos, 2i)   = sin(pos / 10000^(2i/embed_dim))  # Even dims      │ │
│ │ PE(pos, 2i+1) = cos(pos / 10000^(2i/embed_dim))  # Odd dims       │ │
│ │                                                                   │ │
│ │ Where:                                                            │ │
│ │   pos = position in sequence (0, 1, 2, ...)                       │ │
│ │   i = dimension pair index (0, 1, 2, ...)                         │ │
│ │   10000 = base frequency (creates different wavelengths)          │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ FREQUENCY PATTERN ACROSS DIMENSIONS:                                  │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Dimension:  0     1     2     3     4     5     6     7           │ │
│ │ Frequency:  High  High  Med   Med   Low   Low   VLow  VLow        │ │
│ │ Function:   sin   cos   sin   cos   sin   cos   sin   cos         │ │
│ │                                                                   │ │
│ │ pos=0:    [0.00, 1.00, 0.00, 1.00, 0.00, 1.00, 0.00, 1.00]        │ │
│ │ pos=1:    [0.84, 0.54, 0.01, 1.00, 0.00, 1.00, 0.00, 1.00]        │ │
│ │ pos=2:    [0.91,-0.42, 0.02, 1.00, 0.00, 1.00, 0.00, 1.00]        │ │
│ │ pos=3:    [0.14,-0.99, 0.03, 1.00, 0.00, 1.00, 0.00, 1.00]        │ │
│ │                                                                   │ │
│ │ Each position gets a unique mathematical "fingerprint"!           │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ WHY THIS WORKS:                                                       │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Wave Pattern Visualization:                                       │ │
│ │                                                                   │ │
│ │ Dim 0: ∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿∿  (rapid oscillation)                  │ │
│ │ Dim 2: ∿---∿---∿---∿---∿---∿  (medium frequency)                  │ │
│ │ Dim 4: ∿-----∿-----∿-----∿--  (low frequency)                     │ │
│ │ Dim 6: ∿----------∿----------  (very slow changes)                │ │
│ │                                                                   │ │
│ │ • High frequency dims change rapidly between positions            │ │
│ │ • Low frequency dims change slowly                                │ │
│ │ • Combination creates unique signature for each position          │ │
│ │ • Similar positions have similar (but distinct) encodings         │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ KEY ADVANTAGES:                                                       │
│ • Zero parameters (no memory overhead)                                │
│ • Infinite sequence length (can extrapolate)                          │
│ • Smooth transitions (nearby positions are similar)                   │
│ • Mathematical elegance (interpretable patterns)                      │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```
- This method of coming up with positional embeddings is crucial because it creates unique positional signatures and enables smooth interpolation to longer sequences.
- Attention mechanisms leverage these properties to distinguish token positions.


### Computing the Sinusoidal Table

The core of sinusoidal positional encoding is building a table of sin/cos values
where each dimension oscillates at a different frequency. This helper computes
the raw numpy array that both `create_sinusoidal_embeddings` and other components
can reuse.

```
Sinusoidal Table Construction:

Step 1: Position column vector     Step 2: Frequency row vector
  [0]                                [high_freq, ..., low_freq]
  [1]     (max_len, 1)               (embed_dim//2,)
  [2]
  [...]

Step 3: Outer product → angles     Step 4: Interleave sin/cos
  positions * frequencies            pe[:, 0::2] = sin(angles)
  = (max_len, embed_dim//2)          pe[:, 1::2] = cos(angles)
                                     = (max_len, embed_dim)
```
