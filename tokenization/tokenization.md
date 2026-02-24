# Introduction to Tokenization Module

In this module we will build tokenization systems, i.e., character and Byte-Pair Encoding (BPE) based.
Following this, we will be able to process text for language models and NLP tasks.
This module aims to help you understand vocabulary management and decoding/encoding operations.

---

## Introduction to Tokenization

Neural networks operate on numbers, but humans communicate with text.
Tokenization is the crucial bridge that converts text into numerical sequences that the model can process.

### The Text-to-Number Challenge

Consider the sentence: `"Hello, World!"`
How do we turn this into numbers a neural network can process?

A good approach is to use the pipeline below:

```text
┌─────────────────────────────────────────────────────────────────┐
│  TOKENIZATION PIPELINE: Text → Numbers                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (Human Text):     "Hello, world!"                        │
│           │                                                     │
│           ├─ Step 1: Split into tokens                          │
│           │         ['H','e','l','l','o',',', ...]             │
│           │                                                     │
│           ├─ Step 2: Map to vocabulary IDs                      │
│           │         [72, 101, 108, 108, 111, ...]               │
│           │                                                     │
│           ├─ Step 3: Handle unknowns                            │
│           │         Unknown chars → special <UNK> token         │
│           │                                                     │
│           └─ Step 4: Enable decoding                            │
│                     IDs → original text                         │
│                                                                 │
│  Output (Token IDs):  [72, 101, 108, 108, 111, 44, 32, ...]     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### The Four-Step Process of Tokenization

How do we represent text for a neural network to understand?
We need a systematic pipeline that follows this order:

1. **Split text into tokens** – Breaking text into meaningful units (words, subwords, or even characters).
2. **Map tokens into integers** – Create a vocabulary that assigns each token a unique ID.
3. **Handle unknown text** – Deals gracefully with tokens not seen during training.
4. **Enable reconstruction** – Converts the numbers back to readable text for interpretation.

### Essence of This Strategy

The choice of tokenization strategy dramatically affects:

* **Model performance** – How well the model understands text.
* **Vocabulary size** – Memory requirements for embedding tables.
* **Computational efficiency** – Sequence length affects processing time.
* **Robustness** – How well the model handles new/rare words.

---

## Foundation: Tokenization Strategies

Different tokenization approaches make different trade-offs between vocabulary size, sequence length, and semantic understanding.

### Character-Level Tokenization

**Approach:** Each character gets its own token.

```text
┌──────────────────────────────────────────────────────────────┐
│ CHARACTER TOKENIZATION PROCESS                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│ Step 1: Build Vocabulary from Unique Characters             │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ Corpus: ["hello", "world"]                             │  │
│ │ Unique chars: ['h', 'e', 'l', 'o', 'w', 'r', 'd']      │  │
│ │ Vocabulary: ['<UNK>','h','e','l','o','w','r','d']      │  │
│ │ IDs: 0 1 2 3 4 5 6 7                                     │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ Step 2: Encode Text Character by Character                  │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ Text: "hello"                                         │  │
│ │ 'h' → 1    (lookup in vocabulary)                    │  │
│ │ 'e' → 2                                              │  │
│ │ 'l' → 3                                              │  │
│ │ 'l' → 3                                              │  │
│ │ 'o' → 4                                              │  │
│ │ Result: [1, 2, 3, 3, 4]                               │  │
│ └────────────────────────────────────────────────────────┘  │
│                                                              │
│ Step 3: Decode by Reversing ID Lookup                       │
│ ┌────────────────────────────────────────────────────────┐  │
│ │ IDs: [1, 2, 3, 3, 4]                                  │  │
│ │ 1 → 'h'                                               │  │
│ │ 2 → 'e'                                               │  │
│ │ 3 → 'l'                                               │  │
│ │ 3 → 'l'                                               │  │
│ │ 4 → 'o'                                               │  │
│ │ Result: "hello"                                       │  │
│ └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

**Pros:**

* Small vocabulary (~100 characters)
* Handles any text perfectly
* No unknown tokens (every character can be mapped)
* Simple to implement

**Cons:**

* Long sequences (1 character = 1 token)
* Limited semantic understanding (no word boundaries)
* More compute-intensive (longer sequences to process)

---

### Word-Level Tokenization

**Approach:** Each word gets its own token.

```text
Text: "Hello world"
Tokens: ['Hello', 'world']
IDs: [5847, 1254]
```

**Pros:** Semantic meaning preserved, shorter sequences
**Cons:** Huge vocabularies (100K+), many unknown tokens

---

### Subword Tokenization (BPE)

**Approach:** Learn frequent character pairs, build subword pairs.

```text
Text: "tokenization"
Initial: ['t', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n']
Merged: ['to', 'ken', 'ization']
IDs: [142, 1847, 2341]
```

**Pros:** Balance between vocabulary size and sequence length
**Cons:** More complex training process

---

### Strategy Comparison

```text
Text: "tokenization" (12 characters)

Character: ['t','o','k','e','n','i','z','a','t','i','o','n'] → 12 tokens, vocab ~100  
Word: ['tokenization'] → 1 token, vocab 100K+  
BPE: ['token','ization'] → 2 tokens, vocab 10-50K
```

> The sweet spot for most applications is BPE with 10K–50K vocabulary size.

---

## Byte Pair Encoding (BPE) Tokenizer

BPE is the secret ingredient behind modern language models like GPT and BERT.
It learns to merge frequent character pairs, creating subword units that balance vocabulary size with sequence length.

```text
BPE TRAINING ALGORITHM: Learning Subword Units
STEP 1: Initialize with Character Vocabulary
Training Data: ["hello", "hello", "help"]
Initial Tokens: ['h','e','l','l','o</w>'], ['h','e','l','l','o</w>'], ['h','e','l','p</w>']
Starting Vocab: ['h','e','l','o','p','</w>']
STEP 2: Count All Adjacent Pairs
Pair Frequency Analysis: ('h','e'): 3, ('e','l'): 3, ('l','l'): 2, ('l','o'): 2, ('o','<'): 2, ('l','p'):1, ('p','<'):1
STEP 3: Merge Most Frequent Pair ('h','e') → 'he'
Updated Vocab: ['h','e','l','o','p','</w>', 'he']
STEP 4: Repeat Until Target Vocab Size Reached
Merge ('l','l') → 'll'
Updated Vocab: ['h','e','l','o','p','</w>','he','ll']
FINAL RESULTS: "hello" → ['he', 'll', 'o</w>'], "help" → ['he','l','p</w>']
```

### Counting Byte Pairs

* The first step in each BPE iteration is counting how often each adjacent token pair appears across all words, weighted by frequency.

```text
Count Pairs Across All Words (weighted by frequency):

"hello" → ['h','e','l','l','o</w>'] freq=3
"help" → ['h','e','l','p</w>'] freq=1

Pair counting (freq-weighted):
('h','e'): 4
('e','l'): 4
('l','l'): 3
('l','o</w>'): 3
('l','p</w>'): 1
```

### Merging a Byte Pair

* Once we identify the most frequent pair, we merge it everywhere it appears.
* This scans through every word's token list and replaces adjacent occurrences of the pair with a single concatenated token.

```text
Merge Operation: ('h','e') → 'he'

BEFORE merging: "hello" → ['h','e','l','l','o</w>'], "help" → ['h','e','l','p</w>']
AFTER merging: "hello" → ['he','l','l','o</w>'], "help" → ['he','l','p</w>']

Algorithm (linear scan per word):
i=0: tokens[0]='h', tokens[1]='e' → match! append 'he', skip 2
i=2: tokens[2]='l' → no match, append 'l', advance 1
...continue until end of tokens
```

---


# End of Module
