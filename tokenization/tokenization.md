# Introduction to Tokenization Module
-In this module we will build tokenization systems i.e character and Byte-Pair Encoding (BPE) based.
-Following this we will be able to process text for language models and NLP tasks.
-This modules has the hope of helping you understand vocabu;ary mangement and decoding/encoding operations.

## Introduction to Tokenization.
-Neural networks operates on number, but humans communicate with text.
-Tokenization is the crucial bridge that converts text into numerical sequences that the model can process.

### The Text-to-Number Challenge

-Consider the sentence: "Hello,World!".
-How do we turn this into numbers a neural network can process?
-A good approach is to use the pipeline below:
```
┌─────────────────────────────────────────────────────────────────┐
│  TOKENIZATION PIPELINE: Text → Numbers                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input (Human Text):     "Hello, world!"                        │
│           │                                                     │
│           ├─ Step 1: Split into tokens                          │
│           │         ['H','e','l','l','o',',', ...']             │
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
### The Four Step Process of Tokenization
- How do we represent text for a neural network to understand?
- To be able to do this we would need a systematic pipeline, which follows the following order.

**1.Split text into tokens** - Breaking text into meaningful units (words,subwords or even charcters)
**2.Map tokens into integers** - creating a vocabulary that assigns each token a unique ID.
**3.Handle unknown text**-deals gracefully with tokens not seen during training.
**4.Enable reconstruction**- Converts the numbers back to readalbe text for interpretation.

### Essence of this strategy
The choice of tokenization strategy dramatically affects:
- **Model performance** - How well the model understands text.
- **Vocabulary size** - Memory requirements for embedding tables.
-**Computational Efficiency** - Sequence length affects processing time.
- **Robustness** - how well the model handles new/rare words .


## Foundation (Tokenization Strategies)
-Different tokenization approaches make different trade-offs between vocabulary size, sequence length and semantic understanding.

### Character-Level Tokenization
**Approach**: Each character gets its own token

```
┌──────────────────────────────────────────────────────────────┐
│ CHARACTER TOKENIZATION PROCESS                               │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: Build Vocabulary from Unique Characters             │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Corpus: ["hello", "world"]                             │  │
│  │                ↓                                       │  │
│  │ Unique chars: ['h', 'e', 'l', 'o', 'w', 'r', 'd']      │  │
│  │                ↓                                       │  │
│  │ Vocabulary:  ['<UNK>','h','e','l','o','w','r','d']     │  │
│  │ IDs:            0      1   2   3   4   5   6   7       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Step 2: Encode Text Character by Character                  │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Text: "hello"                                         │  │
│  │                                                        │  │
│  │   'h' → 1    (lookup in vocabulary)                    │  │
│  │   'e' → 2                                              │  │
│  │   'l' → 3                                              │  │
│  │   'l' → 3                                              │  │
│  │   'o' → 4                                              │  │
│  │                                                        │  │
│  │  Result: [1, 2, 3, 3, 4]                               │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Step 3: Decode by Reversing ID Lookup                       │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  IDs: [1, 2, 3, 3, 4]                                  │  │
│  │                                                        │  │
│  │   1 → 'h'    (reverse lookup)                          │  │
│  │   2 → 'e'                                              │  │
│  │   3 → 'l'                                              │  │
│  │   3 → 'l'                                              │  │
│  │   4 → 'o'                                              │  |
│  │                                                        │  │
│  │  Result: "hello"                                       │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Pros**:
-Small vocabulary (~100 characters)
-Handles any text perfectly
- No unknown tokens i.e every character can be mapped
- Simple to implement.

**Cons**
-Long sequences (1 character = 1 token)
- Limited semantic understanding (no word boundaries)
- More compute i.e it takes longer sequences to process.

### World-Level Tokenization
**Approach**:each word gets it own token

```
Text: "Hello world"
       ↓
Tokens: ['Hello', 'world']
       ↓
IDs:    [5847, 1254]
```

**Pros**: Semantic meaning preserved, shorter sequences
**Cons**: Huge vocabularies (100K+), many unknown tokens

### Subword Tokenization (BPE)
**Approach**:Learn frequent character pairs, build subword pairs:

``
Text: "tokenization"
       ↓ Character level
Initial: ['t', 'o', 'k', 'e', 'n', 'i', 'z', 'a', 't', 'i', 'o', 'n']
       ↓ Learn frequent pairs
Merged: ['to', 'ken', 'ization']
       ↓
IDs:    [142, 1847, 2341]
```

**Pros**: Balance between vocabulary size and sequence length
**Cons**: More complex training process

### Strategy Comparison

```
Text: "tokenization" (12 characters)

Character: ['t','o','k','e','n','i','z','a','t','i','o','n'] → 12 tokens, vocab ~100
Word:      ['tokenization']                                   → 1 token, vocab 100K+
BPE:       ['token','ization']                               → 2 tokens, vocab 10-50K
```

The sweet spot for most applications is BPE with 10K-50K vocabulary size.

## Byte Pair Encoding (BPE) Tokenizer
- BPE is the secret ingredient behind modern language models like GPT and BERT.
- It learns to merge frequent character pairs, creating subword units that balance vocabulary size with sequence length.

```
┌───────────────────────────────────────────────────────────────────────┐
│ BPE TRAINING ALGORITHM: Learning Subword Units                        │
├───────────────────────────────────────────────────────────────────────┤
│                                                                       │
│ STEP 1: Initialize with Character Vocabulary                          │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Training Data: ["hello", "hello", "help"]                         │ │
│ │                                                                   │ │
│ │ Initial Tokens (with end-of-word markers):                        │ │
│ │   ['h','e','l','l','o</w>']    (hello)                            │ │
│ │   ['h','e','l','l','o</w>']    (hello)                            │ │
│ │   ['h','e','l','p</w>']        (help)                             │ │
│ │                                                                   │ │
│ │ Starting Vocab: ['h', 'e', 'l', 'o', 'p', '</w>']                 │ │
│ │                   ↑ All unique characters                         │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ STEP 2: Count All Adjacent Pairs                                      │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Pair Frequency Analysis:                                          │ │
│ │                                                                   │ │
│ │   ('h', 'e'): ██████  3 occurrences  ← MOST FREQUENT!             │ │
│ │   ('e', 'l'): ██████  3 occurrences                               │ │
│ │   ('l', 'l'): ████    2 occurrences                               │ │
│ │   ('l', 'o'): ████    2 occurrences                               │ │
│ │   ('o', '<'): ████    2 occurrences                               │ │
│ │   ('l', 'p'): ██      1 occurrence                                │ │
│ │   ('p', '<'): ██      1 occurrence                                │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ STEP 3: Merge Most Frequent Pair                                      │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Merge Operation: ('h', 'e') → 'he'                                │ │
│ │                                                                   │ │
│ │ BEFORE:                          AFTER:                           │ │
│ │   ['h','e','l','l','o</w>']  →  ['he','l','l','o</w>']            │ │
│ │   ['h','e','l','l','o</w>']  →  ['he','l','l','o</w>']            │ │
│ │   ['h','e','l','p</w>']      →  ['he','l','p</w>']                │ │
│ │                                                                   │ │
│ │ Updated Vocab: ['h','e','l','o','p','</w>', 'he']                 │ │
│ │                                              ↑ NEW TOKEN!         │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ STEP 4: Repeat Until Target Vocab Size Reached                        │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Iteration 2: Next most frequent is ('l', 'l')                     │ │
│ │ Merge ('l','l') → 'll'                                            │ │
│ │                                                                   │ │
│ │   ['he','l','l','o</w>']     →  ['he','ll','o</w>']               │ │
│ │   ['he','l','l','o</w>']     →  ['he','ll','o</w>']               │ │
│ │   ['he','l','p</w>']         →  ['he','l','p</w>']                │ │
│ │                                                                   │ │
│ │ Updated Vocab: ['h','e','l','o','p','</w>','he','ll']             │ │
│ │                                                  ↑ NEW!           │ │
│ │                                                                   │ │
│ │ Continue merging until vocab_size target...                       │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│ FINAL RESULTS:                                                        │
│ ┌───────────────────────────────────────────────────────────────────┐ │
│ │ Trained BPE can now encode efficiently:                           │ │
│ │                                                                   │ │
│ │ "hello" → ['he', 'll', 'o</w>']  = 3 tokens (vs 5 chars)          │ │
│ │ "help"  → ['he', 'l', 'p</w>']   = 3 tokens (vs 4 chars)          │ │
│ │                                                                   │ │
│ │  Key Insights: BPE automatically discovers:                       │ │
│ │    - Common prefixes ('he')                                       │ │
│ │    - Morphological patterns ('ll')                                │ │
│ │    - Natural word boundaries (</w>)                               │ │
│ └───────────────────────────────────────────────────────────────────┘ │
│                                                                       │
└───────────────────────────────────────────────────────────────────────┘
```

### Counting Byte Pairs

-The first step in each BPE iteration is counting how often each adjacent token pair appears across all words, weighted by the frequency.
-Doing this, gives us a glimpse of which pair to merge next.

```
Count Pairs Across All Words (weighted by frequency):

  word_tokens:                     word_freq:
  "hello" → ['h','e','l','l','o</w>']    freq=3
  "help"  → ['h','e','l','p</w>']        freq=1

  Pair counting (freq-weighted):
    ('h','e'):  3+1 = 4   ← appears in both words
    ('e','l'):  3+1 = 4   ← appears in both words
    ('l','l'):  3   = 3   ← only in "hello"
    ('l','o</w>'): 3 = 3  ← only in "hello"
    ('l','p</w>'): 1 = 1  ← only in "help"
```

### Merging a Byte Piar

- Once after we have identified the most frequent pair, we need to merge it everywhere it appears.
- This scans through every word's token list and replaces adjacent occurences of the pair with a single concatenated token.

```
Merge Operation: ('h', 'e') → 'he'

BEFORE merging:                    AFTER merging:
  "hello" → ['h','e','l','l','o</w>']  →  ['he','l','l','o</w>']
  "help"  → ['h','e','l','p</w>']      →  ['he','l','p</w>']

Algorithm (linear scan per word):
  i=0: tokens[0]='h', tokens[1]='e' → match! append 'he', skip 2
  i=2: tokens[2]='l' → no match, append 'l', advance 1
  ...continue until end of tokens
```