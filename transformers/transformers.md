# Introduction to Transformers Module
- Transformers are the revolutionary architecture that powers modern AI languange models like DeepSeek, ChatGPT,Gemini,Claude etc.
- The key breakthrough is **self-attention**, which allows every token in sequence to interact with every other token, creating rich contextual understanding.
- The **self-attention** concepts are covered in depth in the **attention** module.

### The Transformer Revolution
- The **Transformer Revolution** refers to the dramatic shift in artificial intelligence that began in 2017 when researchers at Google Brain who later marged wit Deepmind to form Google Deepmind, published the papper **Attention Is All You Need**.
- This paper introduced the **Transformer architecture**, which replaced the traditional sequence models like RNNs(Recurrent Neural Networks) and LSTM(Long Short Term Memory Networks) with a mechanism based entirely on attention.
- This breakthrough trigerred the **modern AI boom** whereby large language models,generative AI and many other breakthroughs emerged in the field of Machine Learning.

#### 1. What Came Before Transfomers
- Before transfomers, sequence data like text were handled mainly :
    - Recurrent Neural Networks(RNNS)
    - Long Short-Term Memory networks (LSTMS)
    - Gated Recurrent Networks (GRU)

- These models processed data token by token sequentially.
- Example sentence:
```
"I like deep learning it's interesting"
```
- Processing looked like this:

```
I -> like -> deep -> learning -> it's -> interesting
```

##### Problems
###### 1.No Parallelization
- You had to wit for the previous token before computing the next.
- This made training very slow on GPUs.

###### 2. Long-Range Dependencies
- RNNs struggled to remember information far back in the sequence.
- Example:
```
"FlashAttention makes GPUs go brrr, that is making GPUs go faster"
```
- The model struggles to connect **FlashAttention** to **faster** because they are far apart.

###### 3. Vanishing Gradients
- Gradients shrunk as sequences grow.

#### 2. The BreakThrough Idea: Attention
- Transformers introduced self-attention.
- Instead of processing tokens sequencially, every token looks at every other token.
- Example
```
The quick brown fox ran after the rabbit
```
- The word **ran** attends to:
  - fox
  - the 
  - brown
  - rabbit
  - after

- The model decided which words matter more than other words.
- Mathemtically:
```
Attention(Q,K,V) = softmax(QK^T/sqrt(d_model))*V
```
- Where:
   - **Q = Query**
   - **K = key**
   - **V = value**

#### 3. Why Transformers made an outstanding change

##### Parallel Computation
- All tokens are processed **simultaneously**.
- Instead of:
```
O(n) sequential steps
```
- we compute:
```
attention matrix -> O(n**2)

```

- Which runs massively parrel on GPUs.

##### Better Long-Range Understanding
- Attention connects any two tokens directly.
- Distance in the sequence no longer matters.
- Attention links tokens which far away from each other instantly.

##### Scales Extremely Well
- Transformers scale beautifully with:
   - data
   - parameters
   - compute
- This caling behavior led to foundation models.


```
┌─────────────────────────────────────────────────────────────────┐
│  COMPLETE GPT ARCHITECTURE: From Text to Generation             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT: "Hello world"  →  Token IDs: [15496, 1917]              │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                EMBEDDING LAYER                            │  │
│  │                                                           │  │
│  │  ┌─────────────┐       ┌─────────────────────────────┐    │  │
│  │  │Token Embed  │   +   │ Positional Embedding        │    │  │
│  │  │15496→[0.1,  │       │ pos_0→[0.05, -0.02, ...]    │    │  │
│  │  │     0.3,..]│       │ pos_1→[0.12,  0.08, ...]     │    │  │
│  │  │1917→[0.2,   │       │                             │    │  │
│  │  │    -0.1,..]│       │                              │    │  │
│  │  └─────────────┘       └─────────────────────────────┘    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              TRANSFORMER BLOCK 1                          │  │
│  │                                                           │  │
│  │  x → LayerNorm → MultiHeadAttention → + x → result        │  │
│  │  │                                      ↑                 │  │
│  │  │              residual connection     │                 │  │
│  │  └──────────────────────────────────────┘                 │  │
│  │  │                                                        │  │
│  │  result → LayerNorm → MLP (Feed Forward) → + result       │  │
│  │  │                                           ↑            │  │
│  │  │                residual connection        │            │  │
│  │  └───────────────────────────────────────────┘            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│              TRANSFORMER BLOCK 2 (same pattern)                 │
│                                ↓                                │
│                      ... (more blocks) ...                      │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   OUTPUT HEAD                             │  │
│  │                                                           │  │
│  │  final_hidden → LayerNorm → Linear(embed_dim, vocab_size) │  │
│  │                              ↓                            │  │
│  │               Vocabulary Logits: [0.1, 0.05, 0.8, ...]    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│  OUTPUT: Next Token Probabilities                               │
│  "Hello" → 10%,  "world" → 5%,  "!" → 80%,  ...                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Add picture of Transformer here

### Building Blocks of Transformer

1. **LayerNorm** :stabilizes training by normalizing activations
2. **Multi-layer Perceptron** : provides non-linear transformation
3. **TransformerBlock** : combine attention + MLP with residuals
4. **Token embeddings** : text are converted into vectors
5. **Positional Encoding** : transformers process tokens in parallel, so they need a way to know token order.
6. **Self-Attention** : each token looks at every other token and decides how important it is.
7. **Multi-Head Attention** : instead of computing attention once,  the model does it multiple times in parallel
8. **FeedForward Network (FNN)** : after attention, each token is passed through a fully connected neural network.
9. **Residual Connections** : each sublayer uses a skip connection. This prevents vanishing gradient and stablizes deep models.

## Transformer Mathematics

### Layer Normalization
- Layer Normalization is crucial for training deep transformer networks.
- Unlike batch normalization which normalizes across the batch, the layer norm normalizes across the feature dimension for each individual sample.
- Neural netwoeks become unstable if values grow too large or too small during training.

- Layer normalization solves this by:
   1. **Computing the mean**
   2. **Computing the variance**
   3. **Rescaling the values to have zero mean and unit (1) variance**

```
Mathematical Formula:
output = (x - μ) / σ * γ + β

where:
  μ = mean(x, axis=features)     # Mean across feature dimension
  σ = sqrt(var(x) + ε)          # Standard deviation + small epsilon
  γ = learnable scale parameter  # Initialized to 1.0
  β = learnable shift parameter  # Initialized to 0.0
```
#### Layer Norm Visualization

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER NORMALIZATION: Stabilizing Deep Networks                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT TENSOR: (batch=2, seq=3, features=4)                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Sample 1: [[1.0,  2.0,  3.0,  4.0],     ← Position 0      │  │
│  │            [5.0,  6.0,  7.0,  8.0],     ← Position 1      │  │
│  │            [9.0, 10.0, 11.0, 12.0]]     ← Position 2      │  │
│  │                                                           │  │
│  │ Sample 2: [[13., 14., 15., 16.],         ← Position 0     │  │
│  │            [17., 18., 19., 20.],         ← Position 1     │  │
│  │            [21., 22., 23., 24.]]         ← Position 2     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│           NORMALIZE ACROSS FEATURES (per position)              │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ AFTER NORMALIZATION: Each position → mean=0, std=1        │  │
│  │                                                           │  │
│  │ Sample 1: [[-1.34, -0.45,  0.45,  1.34],                  │  │
│  │            [-1.34, -0.45,  0.45,  1.34],                  │  │
│  │            [-1.34, -0.45,  0.45,  1.34]]                  │  │
│  │                                                           │  │
│  │ Sample 2: [[-1.34, -0.45,  0.45,  1.34],                  │  │
│  │            [-1.34, -0.45,  0.45,  1.34],                  │  │
│  │            [-1.34, -0.45,  0.45,  1.34]]                  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│            APPLY LEARNABLE PARAMETERS: γ * norm + β             │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ FINAL OUTPUT: Model can learn any desired distribution    │  │
│  │ γ (scale) and β (shift) are learned during training       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  KEY INSIGHT: Unlike batch norm, each sample normalized         │
│  independently - perfect for variable-length sequences!         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```
#### Why LayerNorm is Crucial for Transformers
- Without LayerNorm:
    - gradient explode or vanish
    - training becomes unstable
    - deep stacks fail

- With LayerNorm:
  - stable gradients
  - faster training
  - deeper architectures

#### What Exactly Gets Normalized in Transformers

- If a token embedding has dimension:
```
d_model = 768
```
- LayerNorm computes the mean and variance across the 768 features of that token.
- So **normalization** is done per token, not across **tokens**.

### Residual Connections

- Residual connections are the secret to training deep networks.
- They act as "gradient highways" that allow information to flow directly through the network.
- The core idea behind residual connections is that instead of a layer learning a **complete transformation**, it learns only the residual (difference) from the input.

- A normal layer learns like this:
```
y = F(x)
```
- Residual connection:
```
y = x + F(x)
```
where:
```
x = input
F(x) = transformation (attention or either feed-forward network)
```
- The output is the original input plus learned change

#### Simple Example
- Say a layer receives:
```
x = [1,2,3]

```
- The layer computes:
```
F(x) = [0.1,-0.2,0.3]
```
- Residual output:
```
y = x + F(x)
y = [1.1,1.8,3.3]
```
- So the layer slightly adjusts the **representation** instead of replacing it.
- This makes learning much easier

#### Significance of Residual Connections 
- Deep neural networks suffer from several problems:

1. **Vanishing gradients** : gradients shrink as they move backward through many layers.

2. **Information loss** - earlier information gets distorted through many transformations.

3. **Training instabilty** - deep models may fail to converge

- Residual connection fix this by creating a direct information path.

```
┌─────────────────────────────────────────────────────────────────┐
│  RESIDUAL CONNECTIONS: The Gradient Highway System              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  PRE-NORM ARCHITECTURE (Modern Standard):                       │
│                                                                 │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                ATTENTION SUB-LAYER                        │  │
│  │                                                           │  │
│  │  Input (x) ────┬─→ LayerNorm ─→ MultiHeadAttention ─┐     │  │
│  │                │                                    │     │  │
│  │                │         ┌──────────────────────────┘     │  │
│  │                │         ▼                                │  │
│  │                └────→ ADD ─→ Output to next sub-layer     │  │
│  │                      (x + attention_output)               │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                ↓                                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                   MLP SUB-LAYER                           │  │
│  │                                                           │  │
│  │  Input (x) ────┬─→ LayerNorm ─→ MLP (Feed Forward)  ─┐    │  │
│  │                │                                     │    │  │
│  │                │         ┌───────────────────────────┘    │  │
│  │                │         ▼                                │  │
│  │                └────→ ADD ─→ Final Output                 │  │
│  │                      (x + mlp_output)                     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  KEY INSIGHT: Each sub-layer ADDS to the residual stream        │
│  rather than replacing it, preserving information flow!         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**ADD RESIDUAL CONNECTIONS IMAGE HERE!!!**

**Gradient Flow Visualization:**
```
Backward Pass Without Residuals:    With Residuals:
Loss                                Loss
 │ gradients get smaller             │ gradients stay strong
 ↓ at each layer                    ↓ via residual paths
Layer N  ← tiny gradients          Layer N  ← strong gradients
 │                                  │     ↗ (direct path)
 ↓                                  ↓   ↗
Layer 2  ← vanishing                Layer 2  ← strong gradients
 │                                  │     ↗
 ↓                                  ↓   ↗
Layer 1  ← gone!                   Layer 1  ← strong gradients
```

### Feed-Forward Network (MLP)
The MLP is where the "thinking" happens in each transformer block.
It's a simple feed-forward network that provides non-linear transformation capacity.

#### The Role of MLP in Transformers

While attention handles relationships between tokens, the MLP processes each position independetly, adding computational depth and non-linearity.

#### Where the Feedforward Network Appears in a Transformer
- A transformer block contains two main sublayers:
```
Input
  ↓
Multi-Head Attention
  ↓
Add + LayerNorm
  ↓
Feed Forward Network
  ↓
Add + LayerNorm
```
- So after tokens exchange information through attention, the FFN processes the **representation of each token individually**.

#### Structure of the Feed Forward Network
- The FFN is simply a **two-layer fully connected neural network**
- Mathematically:
```
FFN(x) = max(0,x*W1 +b1)W2 + b2
```
- Where:
```
W1 expands the dimension
ReLU(or any other activation function ) introduces non-linearity
W2 project back to original dimension
```
##### Dimension Expanding
- If the transformer hidden size is:
```
d_model
```
then the FFN usually expands to:
```
dff = 4 * d_model
```
Example in models:
```
Model     Hidden Size    FFN size
GPT-2      768             3072
BERT        768             3072
LLAMA        4096           11008
```
- So the transformation follows this procedure:
```
d_model -> dff -> d_model
```
- Example
```
4096 -> 11008 -> 4096

```

- This expansion allows the model to learn complex transformations.

##### MLP Architecture and Information Flow
```
Information Flow Through MLP:

Input: (batch, seq_len, embed_dim=512)
         ↓
┌─────────────────────────────────────────────┐
│ Linear Layer 1: Expansion                   │
│ Weight: (512, 2048)  Bias: (2048,)          │
│ Output: (batch, seq_len, 2048)              │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│ GELU Activation                             │
│ Smooth, differentiable activation           │
│ Better than ReLU for language modeling      │
└─────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────┐
│ Linear Layer 2: Contraction                 │
│ Weight: (2048, 512)  Bias: (512,)           │
│ Output: (batch, seq_len, 512)               │
└─────────────────────────────────────────────┘
         ↓
Output: (batch, seq_len, embed_dim=512)
```

### Transformer Block
- The TransformerBlock is the core building unit of GPT and other transformer models.
- It combines self-attention with feed-forward processing using a carefully designed residual architecture.

#### Pre-Norm vs Post-Norm Architecture
- Modern transformers use "pre-norm" architecture where LayerNorm comes BEFORE the sub-layers, not after.
- This provides better training stability.

```
Pre-Norm Architecture (What We Implement):
┌────────────────────────────────────────────────────────┐
│                     INPUT (x)                          │
│                       │                                │
│       ┌───────────────┴───────────────┐                │
│       │                               │                │
│       ▼                               │                │
│  LayerNorm                            │                │
│       │                               │                │
│       ▼                               │                │
│ MultiHeadAttention                    │                │
│       │                               │                │
│       └───────────────┬───────────────┘                │
│                       │          (residual connection) │
│                       ▼                                │
│                  x + attention                         │
│                       │                                │
│       ┌───────────────┴───────────────┐                │
│       │                               │                │
│       ▼                               │                │
│  LayerNorm                            │                │
│       │                               │                │
│       ▼                               │                │
│      MLP                              │                │
│       │                               │                │
│       └───────────────┬───────────────┘                │
│                       │          (residual connection) │
│                       ▼                                │
│                   x + mlp                              │
│                       │                                │
│                       ▼                                │
│                    OUTPUT                              │
└────────────────────────────────────────────────────────┘
```
#### Why Pre-Norm Is Better for Deep Models
- The main reason is because it has better **gradient flow**.

- In Post-Norm:
```
x ->F(x) -> LayerNorm
```
- Gradients must pass through **LayerNorm** and **nonlinear** layers, which can destabilize training in deep networks.

- In Pre-Norm:
```
x + F(LayerNorm(x))
```
- The residual path becomes almost an **identity function**, allowing gradients to flow easily.

#### Gradient Flow Insight
- With Pre-Norm:
```
output = x + F(LayerNorm(x))
```
- During backpropagation:
```
dL/dx = 1 + dF/dx
```
- The "1" term ensures gradients can propagate even if the sublayer becomes unstable.

#### Information Processing in Transformation Block
```
Step-by-Step Data Transformation:

1. Input Processing:
   x₀: (batch, seq_len, embed_dim) # Original input

2. Attention Sub-layer:
   x₁ = LayerNorm(x₀)               # Normalize input
   attn_out = MultiHeadAttn(x₁)     # Self-attention
   x₂ = x₀ + attn_out               # Residual connection

3. MLP Sub-layer:
   x₃ = LayerNorm(x₂)               # Normalize again
   mlp_out = MLP(x₃)                # Feed-forward
   x₄ = x₂ + mlp_out                # Final residual

4. Output:
   return x₄                        # Ready for next block
```

#### Residual Stream Concept

Think of the residual connections as a "stream" that carries information through the network:

```
Residual Stream Flow:

Layer 1: [original embeddings] ─┐
                                 ├─→ + attention info ─┐
Attention adds information ──────┘                      │
                                                        ├─→ + MLP info ─┐
MLP adds information ───────────────────────────────────┘               │
                                                                        │
Layer 2: carries accumulated information ───────────────────────────────┘
```

Each layer adds information to this stream rather than replacing it, creating a rich representation.

### GPT(Generative Pre-Trained Transformer)
- GPT is the comlete language model that combines all the transformer components into a text generation system.
- It's designed for **autoregressive** generation i.e predicting the next token based on all previous tokens.
- It is **decoder-only**, meaning it uses the **transformer blocks** that focus on the **autoregressive generation**,predicting one token at a time.
#### Core Architecture
- GPT uses a stack of transformer blocks, each containing:
```
1. Pre-Norm Layer Intialization
2.Masked MultiHead Self-Attention - ensures tokens cannot see future tokens.
3. Residual connections/Residual stream - allows smooth gradient flow
4. Feed Forward Network(FFN) - adds non-linear computation per token
```

#### GPT's Autoregressive Nature
- GPT generatrs text one token at a time, using all previously generated tokens as context:
```
Autoregressive Generation Process:

Step 1: "The cat" → model predicts → "sat"
Step 2: "The cat sat" → model predicts → "on"
Step 3: "The cat sat on" → model predicts → "the"
Step 4: "The cat sat on the" → model predicts → "mat"

Result: "The cat sat on the mat"
```

#### Complete GPT Architecture
```
+-------------------------------------------------------------+
|                      GPT ARCHITECTURE                       |
|                                                             |
|  Input: Token IDs [15496, 1917, ...]                        |
|                       |                                     |
|                       v                                     |
|  +--------------------+--------------------+                |
|  |           EMBEDDING LAYER               |                |
|  |  +--------------+  +-----------------+  |                |
|  |  | Token Embed  |  | Position Embed  |  |                |
|  |  | vocab->vector|  | sequence->vector|  |                |
|  |  +--------------+  +-----------------+  |                |
|  |              \          /               |                |
|  |               +--------+                |                |
|  +--------------------+--------------------+                |
|                       |                                     |
|                       v                                     |
|  +--------------------+--------------------+                |
|  |        TRANSFORMER BLOCK 1              |                |
|  |  +---------+    +---------+    +-----+  |                |
|  |  |LayerNorm| -> |Attention| -> | +x  |  |                |
|  |  +---------+    +---------+    +--+--+  |                |
|  |                                   |     |                |
|  |  +---------+    +---------+    +--v--+  |                |
|  |  |LayerNorm| -> |   MLP   | -> | +x  |  |                |
|  |  +---------+    +---------+    +-----+  |                |
|  +--------------------+--------------------+                |
|                       |                                     |
|          ... (more transformer blocks) ...                  |
|                       |                                     |
|                       v                                     |
|  +--------------------+--------------------+                |
|  |            OUTPUT HEAD                  |                |
|  |  +---------+    +--------------------+  |                |
|  |  |LayerNorm| -> |Linear(embed->vocab)|  |                |
|  |  +---------+    +--------------------+  |                |
|  +--------------------+--------------------+                |
|                       |                                     |
|                       v                                     |
|  Output: Vocabulary Logits [0.1, 0.05, 0.8, ...]            |
+-------------------------------------------------------------+
```
**ADD PICTURE OF GPT ARCHITECTURE**

#### Causal Masking for Autoregressive Training
- During training, GPT sees the entire sequnece but must "cheat" by looking at the future token.

```
┌─────────────────────────────────────────────────────────────────┐
│  CAUSAL MASKING: Preventing Future Information Leakage          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  SEQUENCE: ["The", "cat", "sat", "on"]                          │
│  POSITIONS:   0      1      2     3                             │
│                                                                 │
│  ATTENTION MATRIX (what each position can see):                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │      Pos:  0   1   2   3                                 │   │
│  │  Pos 0:  [ ✓   ✗   ✗   ✗ ]  ← "The" only sees itself     │   │
│  │  Pos 1:  [ ✓   ✓   ✗   ✗ ]  ← "cat" sees "The" + self    │   │
│  │  Pos 2:  [ ✓   ✓   ✓   ✗ ]  ← "sat" sees all previous    │   │
│  │  Pos 3:  [ ✓   ✓   ✓   ✓ ]  ← "on" sees everything       │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  IMPLEMENTATION: Upper triangular matrix with -∞                │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ [[  0, -∞, -∞, -∞],                                      │   │
│  │  [  0,   0, -∞, -∞],                                     │   │
│  │  [  0,   0,   0, -∞],                                    │   │
│  │  [  0,   0,   0,   0]]                                   │   │
│  │                                                          │   │
│  │ After softmax: -∞ becomes 0 probability                  │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
│  WHY THIS WORKS: During training, model sees entire sequence    │
│  but mask ensures position i only attends to positions ≤ i      │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

```

#### Generation Temparature Control
- Temperature controls the randomness of generation.
- Temperature **T** scales the logits:
```
T = 1 ->default, use model probabilites as-is
T < 1, makes the disbribution sharper, more deterministic
T > 1, makes the distribution flatter, more random
```
- Low temperature - the model "plays it safe", choosing high-probability tokens more often.
- High temperature -> model explores less likely tokens,producing creative or diverse output.

```
Temperature Effects:

Original logits: [1.0, 2.0, 3.0]

Temperature = 0.1 (Conservative):
Scaled: [10.0, 20.0, 30.0] → Sharp distribution
Probs: [0.00, 0.00, 1.00] → Always picks highest

Temperature = 1.0 (Balanced):
Scaled: [1.0, 2.0, 3.0] → Moderate distribution
Probs: [0.09, 0.24, 0.67] → Weighted sampling

Temperature = 2.0 (Creative):
Scaled: [0.5, 1.0, 1.5] → Flatter distribution
Probs: [0.18, 0.33, 0.49] → More random
```

#### Model Scaling and Parameters
```
GPT Model Size Scaling:

Nano GPT (our implementation):
- embed_dim: 64, layers: 2, heads: 4
- Parameters: ~50K
- Use case: Learning and experimentation

GPT-2 Small:
- embed_dim: 768, layers: 12, heads: 12
- Parameters: 117M
- Use case: Basic text generation

GPT-3:
- embed_dim: 12,288, layers: 96, heads: 96
- Parameters: 175B
- Use case: Advanced language understanding

GPT-4 (estimated):
- embed_dim: ~16,384, layers: ~120, heads: ~128
- Parameters: ~1.7T
- Use case: Reasoning and multimodal tasks
```

## Systems Analysis (Parameter Scaling and Memory)
- Transformer models scale drastically with size, leading to both opportunities and challenges.

### The Scaling Laws Revolution
- The scaling laws revolution is one of the most important discoveries in modern AI.
- It showed that simply scaling three things can improve AI performance:
```
1.Model size(parameters)
2. Dataset size
3. Compute (training FLOPs)
```
- The **Scaling Laws** revealed that:
```
""If you scale models,dta and compute in the right proportions,performance improves predictably"""
```
#### The core discovery: Power law
- Model performance follows a power-law relationship.
- Loss roughly behaves like:
```
Loss ≈ A * N^(-α)
```

- Where:
```
N = model size or compute 
α = scaling exponent
A = constant
```

- Meaning:
```
- doubling compute always improves performance
- Improvement becomes gradually smaller
```
But importantly: 
**improvement never
stops**

#### Three Scaling Dimensions
- In a paper published by OpenAI called **Scaling laws for Neural Language Models**, found that performance imporves when scaling three variables.

##### 1.Model Size
- Increasing parameters improves capability.

| Model     | Parameters  |
| :---      | :---        |
| GPT-2    | 1.5B       |
| GPT-3     | 175B        |
| modern models     | trillons       |

##### 2.Dataset Size 
- Training data must grow with model size.
| Model | Dataset size |
| :---      | :---        |
| GPT-2    | ~40GB        |
| GPT-3     | ~570GB        |
| modern LLMs     | multiple TB        |

- If the dataset is too small,the model overfits .

##### 3.Compute 
- Training compute is measured in FLOPs.
- Compute roughly scales like:
```
Computes = parameters x tokens x steps
```
- More compute means:
```
longer training
larger models
bigger datasets
```

### Memory Scaling Analysis
- Memory requirements grow in different ways for different componets:

```
Memory Scaling by Component:

1. Parameter Memory (Linear with model size):
   - Embeddings: vocab_size × embed_dim
   - Transformer blocks: ~4 × embed_dim²
   - Total: O(embed_dim²)

2. Attention Memory (Quadratic with sequence length):
   - Attention matrices: batch × heads × seq_len²
   - This is why long context is expensive!
   - Total: O(seq_len²)

3. Activation Memory (Linear with batch size):
   - Forward pass activations for backprop
   - Scales with: batch × seq_len × embed_dim
   - Total: O(batch_size)
```
### The Attention Memory Wall
```
┌─────────────────────────────────────────────────────────────────┐
│  ATTENTION MEMORY WALL: Why Long Context is Expensive           │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  MEMORY USAGE BY SEQUENCE LENGTH (Quadratic Growth):            │
│                                                                 │
│  1K tokens:   [▓] 16 MB                ← Manageable             │
│  2K tokens:   [▓▓▓▓] 64 MB             ← 4× memory (quadratic)  │
│  4K tokens:   [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] 256 MB   ← 16× memory          │
│  8K tokens:   [████████████████████████████████] 1 GB           │
│  16K tokens:  [████████████████████████████████████████] 4 GB   │
│  32K tokens:  [████████████████████████████████████████████] →  │
│               ← extends to 16 GB (off the chart!)               │
│                                                                 │
│  REAL-WORLD CONTEXT LIMITS:                                     │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ GPT-3:     2K tokens  (limited by memory)                 │  │
│  │ GPT-4:     8K tokens  (32K with optimizations)            │  │
│  │ Claude-3:  200K tokens (special techniques required!)     │  │
│  │ GPT-4o:    128K tokens (efficient attention)              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                 │
│  MATHEMATICAL SCALING:                                          │
│  Memory = batch_size × num_heads × seq_len² × 4 bytes           │
│                                   ↑                             │
│                          This is the killer!                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

```
#### Diving deep into Attention Memory Wall
- The **attention memory wall** is a major limitation of transformer models.
- It refers to the point where **memory requirements of the attention mechanism grow too large for hardware to handle efficiently**.
- To put it in simpler terms:
```
As the input sequence becomes longer, the memory needed for attention growns quadratically,quickly overwhelming GPUs.
```
##### Why Attention Causes the Memory Wall
- The attention mechanism compares **every token with every other token**.

- If a sequence has **n** tokens, the attention matrix is:
```
n x n
```
- So the number od interactions is:
```
n**2
```

example:
| Tokens | Attention Matrix Size |
| :---      | :---        |
| 512    | 262k     |
| 2048     | 4.2M        |
| 10,000     | 100M        |
| 100,000     | 10B        |

- This matrix must be stored in GPU memory during training.

##### What Actually Uses the Memory
- During attention computation,several large tensors are stored:
```
1.Query Matrix (Q)
2.Key matrix(K)
3. Value matrix(V)
4. Attention scores
5.Softmax results
6.Gradient during backpropagation
```
- The attention scores matrix is the biggest problem.
-It size :
```
n x n x number_of_heads
```
- This quickly exceeds GPU memory.

##### Example of the Memory Explosion
- Suppose:
```
Sequence length = 16,000
Heads = 32
Float size = 4 bytes
```
- Attention matrix size:
```
16000 x 16000 x 32 x 4 bytes = 32 GB just for attention scores
```
- This is more than the memory of most GPUs.

##### Whay it is called a **Memory Wall**
- In computing, a memory wall occures when:
```
memory bandwidth or capacity limits system performance
```
- For transformers, compute is fast and GPUs are powerful.
- But : **memory cannot keep up**.
- The GPU spends time waiting for memory transfers.
- This becomes the main bottleneck.
