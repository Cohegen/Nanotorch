# Introduction to Memoization Module

## Prerequisites 
- This module assumes that you are familiar with:
  - Tensor module.
  - Autograd module.
  - Transformers module.
  - Profiling module.
  - Acceleration module.


## Why Memoization Matters for Transformers
- In transformers there's an O(n**2) growth in latency as we generate text.
- In machine learning systems, memoization is a fundemental optimization pattern: cache expensive computations so they don't need to be repeated.
- For transformers, this means caching the key-value pairs that attention computes, since they never change for already-processed tokens.

```
Memoization Pattern:
┌─────────────────────────────────────────────────────────────┐
│  Without Memoization (Naive):                               │
│  f(x) called 100 times → 100 computations                  │
│                                                             │
│  With Memoization (Cached):                                │
│  f(x) called 100 times → 1 computation + 99 cache lookups  │
└─────────────────────────────────────────────────────────────┘
```

## Understanding the Autoregressive Generation Problem

### The Core Inefficiency
- When generating text token by token, transformers face a fundemental computational bottleneck.

```
Token Generation Process (Without Caching):

Step 1: Generate "Hello"
Input: [START]
Attention: Q₁ × [K₁] × [V₁]               ← 1 computation

Step 2: Generate "world"
Input: [START, Hello]
Attention: Q₂ × [K₁, K₂] × [V₁, V₂]       ← 2 computations (K₁,V₁ RECOMPUTED!)

Step 3: Generate "!"
Input: [START, Hello, world]
Attention: Q₃ × [K₁, K₂, K₃] × [V₁, V₂, V₃] ← 3 computations (K₁,V₁,K₂,V₂ RECOMPUTED!)
```

**The Problem**: For each new token, we recompute ALL previous key-value pairs even though they never change.

### Computational Complexity Analysis

```
Naive Generation Complexity:
Step 1: 1 K,V computation
Step 2: 2 K,V computations
Step 3: 3 K,V computations
...
Step n: n K,V computations

Total: 1 + 2 + 3 + ... + n = n(n+1)/2 = O(n²) complexity!
```
- For a 100-token sequence, this means **5,050 redundant computations**.

### Real-World Impact
- This inefficiency makes production LLM serving economically impossible without optimization:

- **ChatGPT/GPT-4**: Would be too slow for real-time chat without caching
- **Code completion**: IDEs couldn't provide instant suggestions
- **Mobile deployment**: On-device generation would drain batteries instantly
- **API serving**: Server costs would be 10x+ higher

**The Solution**: Cache key-value pairs after computing them once, transforming O(n²) into O(n).

## The Key-Value Caching Insight

### Mathematical Foundation
- The core insight comes from understanding what changes during autoregressive generation:

```
Attention Computation Breakdown:

Q = new_token @ W_q        ← Only new token (changes each step)
K = all_tokens @ W_k       ← Includes old tokens (mostly redundant!)
V = all_tokens @ W_v       ← Includes old tokens (mostly redundant!)

attention_output = softmax(Q @ K.T / √d_k) @ V
```
- K and Vmatrices for previous tokens NEVER change

```
Token Dependencies:
K₁ = token₁ @ W_k  ← Computed once, never changes
K₂ = token₂ @ W_k  ← Computed once, never changes
K₃ = token₃ @ W_k  ← Computed once, never changes

Same for V₁, V₂, V₃...
```

### Cache-Optimized Generation

```
Optimized Generation Process (With Caching):

Step 1: Generate "Hello"
Compute: K₁, V₁ → Store in cache
Attention: Q₁ × cached[K₁] × cached[V₁]

Step 2: Generate "world"
Compute: K₂, V₂ → Append to cache
Attention: Q₂ × cached[K₁, K₂] × cached[V₁, V₂]

Step 3: Generate "!"
Compute: K₃, V₃ → Append to cache
Attention: Q₃ × cached[K₁, K₂, K₃] × cached[V₁, V₂, V₃]
```

**Result**:Each step computes only ONE new K,V pair instead of recomputing ALL.

### Memory vs Compute Trade-Off

```
Traditional Approach:
Memory: O(1)          (no storage needed)
Compute: O(n²)        (recompute everything)

Cached Approach:
Memory: O(n × d_k)    (store all K,V pairs)
Compute: O(n)         (only compute new pairs)

For n=100, d_k=64:
Memory cost: 6.4 KB per layer
Compute savings: 50x reduction in K,V computations
```

**Trade-off Winner**: Memory is cheap, compute is expensive.  Uses O(n) to save O(n**2) compute.


## KVCACHE Class
- Our KVCache needs to efficiently handles:

1. **Multi-layer storage**: Each transformer layer needs its own K,V cache
2. **Multi-head attention**: Each attention head has separate K,V pairs
3. **Batch processing**: Support multiple sequences simultaneously (batch inference)
4. **Dynamic updates**: Efficiently append new tokens without copying data
5. **Memory management**: Pre-allocate space to avoid dynamic resizing overhead

### Cache Architecture Visualization

```
KVCache Memory Layout:
┌────────────────────────────────────────┐
│                KVCache Object          │
├────────────────────────────────────────┤
│ Layer 0: ┌─────────────┬─────────────┐ │
│          │ Key Cache   │ Value Cache │ │
│          │ (B,H,S,D)   │ (B,H,S,D)   │ │
│          └─────────────┴─────────────┘ │
├────────────────────────────────────────┤
│ Layer 1: ┌─────────────┬─────────────┐ │
│          │ Key Cache   │ Value Cache │ │
│          │ (B,H,S,D)   │ (B,H,S,D)   │ │
│          └─────────────┴─────────────┘ │
├────────────────────────────────────────┤
│   ...    ┌─────────────┬─────────────┐ │
│ Layer N: │ Key Cache   │ Value Cache │ │
│          │ (B,H,S,D)   │ (B,H,S,D)   │ │
│          └─────────────┴─────────────┘ │
└────────────────────────────────────────┘

Where:
B = batch_size    (number of sequences)
H = num_heads     (attention heads per layer)
S = max_seq_len   (maximum sequence length)
D = head_dim      (dimension per attention head)
```

### Update Operation Flow

```
Cache Update Process:
                      seq_pos = 2
                         ↓
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ K₁  │ K₂  │ ??? │ ??? │ ??? │ ??? │ ← Key Cache
├─────┼─────┼─────┼─────┼─────┼─────┤
│ V₁  │ V₂  │ ??? │ ??? │ ??? │ ??? │ ← Value Cache
└─────┴─────┴─────┴─────┴─────┴─────┘

New token arrives: K₃, V₃

                      seq_pos = 2
                         ↓
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ K₁  │ K₂  │ K₃  │ ??? │ ??? │ ??? │ ← Write K₃ here
├─────┼─────┼─────┼─────┼─────┼─────┤
│ V₁  │ V₂  │ V₃  │ ??? │ ??? │ ??? │ ← Write V₃ here
└─────┴─────┴─────┴─────┴─────┴─────┘

Then: seq_pos += 1 (advance to position 3)
```

## Cache-Aware Generation

### Integration Strategy
- We need a clean way to enable KV caching in our existing transformer models without breaking the existing code.
- A **enable_kv_cache()** function has been created in **memoization.py** which:
   1. Creates a KVCache instance sized for the model
   2. Patches the model's attention layers to use caching
   3. Returns the cache for manual control if needed

- The actual integration with attention happens through monkey-patching where we:
   1. Check if cache is enabled
   2. Only compute K,V for new token (not all tokens)
   3. Update cache with new K,V
   4. Use cached K,V for attention computation

### Generation Flow Comparison

```
Without Cache (Current):
for each new token:
    input_seq = [all tokens so far]        # Length grows: 1, 2, 3, ...
    logits = model.forward(input_seq)       # Recomputes everything!
    next_token = sample(logits[-1])
    append next_token

With Cache (New):
cache = enable_kv_cache(model)
for each new token:
    input_token = [just new token]          # Length always 1
    logits = model.forward(input_token)     # Uses cache automatically!
    next_token = sample(logits[-1])
    append next_token
```

**Key Difference** :Input changes from growing sequence to single token, with cache providing history.

## Integration (non-Invasive Model Enhacement)

### The Challenge
- We have built KV caching in this module, but our transformer in the transformer module doesn't know about it.
- There are two possible ways to fix this but one is bad while the other is perfect:

**BAD Solution**: Go back and modify the transformer module specifically (MultiHeadAttention)
- Breaks "forward-only" learning, where we revisit old modules.
- Makes the transformer module depend on this module (wrong dependency direction!)
- Violates clean module boundaries

**GOOD Solution**: this module ADDS caching to existing models without modification!
- Use composition + monkey-patching (like `enable_autograd()`)
- this module wraps/enhances the transformer module, not modifies it
- Here we are trying to  learn systems engineering i.e : "Add capabilities, don't break old code"

### Using KVCache in Practice
- To use KV caching in our transformer generation:

**Before Generation:**
1. Enable caching with `enable_kv_cache(model)`
2. Cache is automatically sized for your model architecture
3. Verify memory usage is acceptable

**During Generation:**
1. For the first token (prompt), process normally and populate cache
2. For subsequent tokens:
   - Only process the NEW token (not entire sequence)
   - Cache is automatically updated with new K,V pairs
   - Cached values are automatically used in attention
   - Cache position advances after all layers

**After Generation:**
1. Reset cache if generating another sequence: `model._kv_cache.reset()`
2. Disable caching if needed: `disable_kv_cache(model)`
3. Monitor memory usage for production deployment

### Performance

```

Expected Speedup by Sequence Length:
┌───────────┬──────────┬───────────┬──────────┐
│ Seq Len   │ No Cache │ With Cache│ Speedup  │
├───────────┼──────────┼───────────┼──────────┤
│  10 tokens│ ~80 tok/s│ ~600 tok/s│   7.5x   │
│  25 tokens│ ~40 tok/s│ ~500 tok/s│  12.5x   │
│  50 tokens│ ~25 tok/s│ ~400 tok/s│  16.0x   │
│ 100 tokens│ ~12 tok/s│ ~200 tok/s│  16.7x   │
└───────────┴──────────┴───────────┴──────────┘

Key Insight: Speedup increases with sequence length!
Why? Longer sequences = more redundant computation without cache.
```

### Production Considerations

**Memory Management:**
- Cache memory = `batch_size × num_layers × num_heads × max_seq_len × head_dim × 4 bytes`
- For GPT-2 (12 layers, 12 heads, seq_len=1024, head_dim=64): ~37 MB per sequence
- For GPT-3 (96 layers, 96 heads, seq_len=2048, head_dim=128): ~4.7 GB per sequence

**Trade-off Analysis:**
- **10x+ speedup** for typical generation lengths (50-200 tokens)
- **Modest memory cost** compared to model parameters (often <1% of model size)
- **Enables real-time interaction** that's impossible without caching

**Best Practices:**
1. Always use caching for production serving
2. Tune `max_seq_len` to expected generation length (don't over-allocate)
3. Consider batch inference to amortize model loading costs
4. Monitor cache memory usage in production







