
## Understanding the Data Pipeline

## The Data Pipeline Journey
Imagine you have 50,000 images of cats and dogs, you want to train a neural network to classify them:

```
Raw Data Storage          Dataset Interface         DataLoader Batching         Training Loop
┌─────────────────┐      ┌──────────────────┐      ┌────────────────────┐      ┌─────────────┐
│ cat_001.jpg     │      │ dataset[0]       │      │ Batch 1:           │      │ model(batch)│
│ dog_023.jpg     │ ───> │ dataset[1]       │ ───> │ [cat, dog, cat]    │ ───> │ optimizer   │
│ cat_045.jpg     │      │ dataset[2]       │      │ Batch 2:           │      │ loss        │
│ ...             │      │ ...              │      │ [dog, cat, dog]    │      │ backward    │
│ (50,000 files)  │      │ dataset[49999]   │      │ ...                │      │ step        │
└─────────────────┘      └──────────────────┘      └────────────────────┘      └─────────────┘
```

### Why this pipeline matters
**Individual Access(Dataset)**: Neural networks can't process 50,000 files at once. We neeed a way to access one sample at a time i.e "Give me image #1247"

**Batching Processing(DataLoader)**: GPUs are parrallel machines ,they much faster in processing 32 images simultanously than 1 image 32 times.

**Memory Efficiency** : loading all 50,000 images into memory would require ~150GB. Instead, we load only the current batch (~150MB).

**Training Variety**: Shuffling ensures the model sees different combinations each epochs, preventing memorization.


## The Dataset Abstraction

The Dataset class provides a uniform interface for accessing data regardless of whether it's stored as files, in memory in database or generated on-the-fly:


```
Dataset Interface

┌─────────────────────────────────────┐
│ __len__()     → "How many samples?" │
│ __getitem__(i) → "Give me sample i" │
└─────────────────────────────────────┘
          ↑                ↑
     Enables for     Enables indexing
    loops/iteration   dataset[index]
```

**Connection to systems**: This abstraction is crucial because it seperates *how data is stored* from *how it's accessed*, enabling optimization like caching, prefetching and parallel loading.

"""

"""
##TensorDataset- this is when data lives in Memory


##Understanding TensorDataset Structure
TensorDataset takes multiple tensors and aligns them by their first dimension i.e  the sample dimension.

```
Input Tensors (aligned by first dimension):
  Features Tensor        Labels Tensor         Metadata Tensor
  ┌─────────────────┐   ┌───────────────┐     ┌─────────────────┐
  │ [1.2, 3.4, 5.6] │   │ 0 (cat)       │     │ "image_001.jpg" │ ← Sample 0
  │ [2.1, 4.3, 6.5] │   │ 1 (dog)       │     │ "image_002.jpg" │ ← Sample 1
  │ [3.0, 5.2, 7.4] │   │ 0 (cat)       │     │ "image_003.jpg" │ ← Sample 2
  │ ...             │   │ ...           │     │ ...             │
  └─────────────────┘   └───────────────┘     └─────────────────┘
        (N, 3)               (N,)                   (N,)

Dataset Access:
  dataset[1] → (Tensor([2.1, 4.3, 6.5]), Tensor(1), "image_002.jpg")
```

### Why TensorDataset is poerful
**Memory Allocation**: All data is pre-loaded ans stored contigously in memory, enabling fast access patterns.

**Vectorized Operations**:since everything is already tensors, no conversion overhead during training.

**Supervised Learning Perfect**: Naturally handles (features,labels) pairs, plus any additional metadata.

**Batch-Friendly**: When DataLoader needs a batch, it can slice multiple samples efficiently.

##Real world Usage pattern.

```
# Computer Vision
images = Tensor(shape=(50000, 32, 32, 3))  # CIFAR-10 images
labels = Tensor(shape=(50000,))            # Class labels 0-9
dataset = TensorDataset(images, labels)

# Natural Language Processing
token_ids = Tensor(shape=(10000, 512))     # Tokenized sentences
labels = Tensor(shape=(10000,))            # Sentiment labels
dataset = TensorDataset(token_ids, labels)

# Time Series
sequences = Tensor(shape=(1000, 100, 5))   # 100 timesteps, 5 features
targets = Tensor(shape=(1000, 10))         # 10-step ahead prediction
dataset = TensorDataset(sequences, targets)
```

So the TensorDataset transforms "array of data" into "dataset that serves samples."

"""

""""
## DataLoader

The dataloader is the component which transforms individula dataset samples into batches that  neural networks crave.
This where data loading becomes a systems challenge.

### Understanding Batching: From Samples to Tensors
Dataloader performs a crucial transformation as it collects individual samples and stacks them into batch tensors:

```
Step 1: Individual Samples from Dataset
  dataset[0] → (features: [1, 2, 3], label: 0)
  dataset[1] → (features: [4, 5, 6], label: 1)
  dataset[2] → (features: [7, 8, 9], label: 0)
  dataset[3] → (features: [2, 3, 4], label: 1)

Step 2: DataLoader Groups into Batch (batch_size=2)
  Batch 1:
    features: [[1, 2, 3],    ← Stacked into shape (2, 3)
               [4, 5, 6]]
    labels:   [0, 1]         ← Stacked into shape (2,)

  Batch 2:
    features: [[7, 8, 9],    ← Stacked into shape (2, 3)
               [2, 3, 4]]
    labels:   [0, 1]         ← Stacked into shape (2,)
```

### The Shuffling Process

The shuffling process randomizes which samples appear in which batches which is crucial for good training.

```
Without Shuffling (epoch 1):          With Shuffling (epoch 1):
  Batch 1: [sample 0, sample 1]         Batch 1: [sample 2, sample 0]
  Batch 2: [sample 2, sample 3]         Batch 2: [sample 3, sample 1]
  Batch 3: [sample 4, sample 5]         Batch 3: [sample 5, sample 4]

Without Shuffling (epoch 2):          With Shuffling (epoch 2):
  Batch 1: [sample 0, sample 1]  ✗      Batch 1: [sample 1, sample 4]  ✓
  Batch 2: [sample 2, sample 3]  ✗      Batch 2: [sample 0, sample 5]  ✓
  Batch 3: [sample 4, sample 5]  ✗      Batch 3: [sample 2, sample 3]  ✓

  (Same every epoch = overfitting!)     (Different combinations = better learning!)
```

### Dataloader as a System Component

**Memory management**: Dataloaders  only holds one batch in memory at a time, not the entire dataset.
**Iteration Interface**: Provides Python iterator protocol so training loops can use `for batch in dataloader:`.
**Collation Strategy**: Automatically stacks tensors from individual samples into batch tensors
**Performance Critical**: This is often the bottleneck in the training pipelines -loading and preparing data can be slower than the forward pass.

###Dataloader Algorithm
```
1.Create indices list: [0,1,2,...,dataset_length-1]
2.If shuffle=True: randomly shuffle the indices
3.Group indices into chunks of batch_size
4.For each chunk:
   a. Retrieve samples: [dataset[i] for i in chunk]
   b. Collate samples: stack individual tensors into batch tensors
   c. Yield the batch tensor tuple
```
This transfroms the dataset from "acces one sample" to "iterate through batches" .
"""

"""
### Data Augmentation
Data augmentation is crucial as it prevents overfitting through variety.

Data Augmentation is one of the most effective techniques for improving model generalization.
We do it by applying random transformations during training, artificially expand the dataset and force the model to learn robust, invariant features.

```
Without Augmentation:                With Augmentation:
Model sees exact same images         Model sees varied versions
every epoch                          every epoch

Cat photo #247                       Cat #247 (original)
Cat photo #247                       Cat #247 (flipped)
Cat photo #247                       Cat #247 (cropped left)
Cat photo #247                       Cat #247 (cropped right)
     ↓                                    ↓
Model memorizes position             Model learns "cat-ness"
Overfits to training set             Generalizes to new cats
```

```
RandomHorizontalFlip (50% probability):
┌──────────┐     ┌──────────┐
│  🐱 →    │  →  │    ← 🐱  │
│          │     │          │
└──────────┘     └──────────┘
Cars, cats, dogs look similar when flipped!

RandomCrop with Padding:
┌──────────┐     ┌────────────┐     ┌──────────┐
│   🐱     │  →  │░░░░░░░░░░░░│  →  │  🐱      │
│          │     │░░  🐱     ░│     │          │
└──────────┘     │░░░░░░░░░░░░│     └──────────┘
  Original        Pad edges        Random crop
                  (with zeros)     (back to 32×32)
```

## Training vs Evaluation

**Critical**- augmentation applies ONLY during training!


```
Training:                              Evaluation:
┌─────────────────┐                   ┌─────────────────┐
│ Original Image  │                   │ Original Image  │
│      ↓          │                   │      ↓          │
│ Random Flip     │                   │ (no transforms) │
│      ↓          │                   │      ↓          │
│ Random Crop     │                   │ Direct to Model │
│      ↓          │                   └─────────────────┘
│ To Model        │
└─────────────────┘
```
"""

"""
### Understanding Image Data

Images are just 2D arrays of numbers (pixels). Here actual 8x8 handwritten digits:

```
Digit "5" (8×8):        Digit "3" (8×8):        Digit "8" (8×8):
 0  0 12 13  5  0  0  0   0  0 11 12  0  0  0  0   0  0 10 14  8  1  0  0
 0  0 13 15 10  0  0  0   0  2 16 16 16  7  0  0   0  0 16 15 15  9  0  0
 0  3 15 13 16  7  0  0   0  0  8 16  8  0  0  0   0  0 15  5  5 13  0  0
 0  8 13  6 15  4  0  0   0  0  0 12 13  0  0  0   0  1 16  5  5 13  0  0
 0  0  0  6 16  5  0  0   0  0  1 16 15  9  0  0   0  6 16 16 16 16  1  0
 0  0  5 15 16  9  0  0   0  0 14 16 16 16  7  0   1 16  3  1  1 15  1  0
 0  0  9 16  9  0  0  0   0  5 16  8  8 16  0  0   0  9 16 16 16 15  0  0
 0  0  0  0  0  0  0  0   0  3 16 16 16 12  0  0   0  0  0  0  0  0  0  0

Visual representation:
░█████░          ░█████░          ░█████░
░█░░░█░          ░░░░░█░          █░░░░█░
░░░░█░░          ░░███░░          ░█████░
░░░█░░░          ░░░░█░░          █░░░░█░
░░█░░░░          ░█████░          ░█████░
```

**Shape Transformations in Dataloader:**

```
Individual Sample (from Dataset):
  image: (8, 8)      ← Single 8×8 image
  label: scalar      ← Single digit (0-9)

After DataLoader batching (batch_size=32):
  images: (32, 8, 8)  ← Stack of 32 images
  labels: (32,)       ← Array of 32 labels

This is what the model sees during training!
```
We have buit the **data loading infrastructure** that powers all modern ML:
-Dataset abstraction 
- TensorDataset - wraps one or more tensors into a single dataset object.
- Dataloader - resposible for batching, shuffling, iteration.
- Data Augmentation - expands the size and diversity of a training dataset without collecting new samples.
**Real-world connections:** So we have implemented the same patterns as:
- Pytorch's `torch.utils.data.DataLoader`
- Pytorch's `torchvision.transforms`
- TensorFlow's `tf.data.Dataset`


"""

"""
## System Analysis 
Now let's understand where time and memory go since it's crucial for building ML systems.
In a typical training step, time is split between data loading and computation:

```
Training Step Breakdown:
┌─────────────────────────────────────────────────────────────┐
│ Data Loading        │ Forward Pass     │ Backward Pass      │
│ ████████████        │ ███████          │ ████████           │
│ 40ms                │ 25ms             │ 35ms               │
└─────────────────────────────────────────────────────────────┘
              100ms total per step

Bottleneck Analysis:
- If data loading > forward+backward: "Data starved" (CPU bottleneck) i.e when the CPU cannot process data fast enough for the GPU to process.
- If forward+backward > data loading: "Compute bound" (GPU bottleneck) i.e when the GPU is the limiting factor because it is fuly saturated with the computational load itself.
- Ideal: Data loading ≈ computation time (balanced pipeline)
```

### Memory Scaling i.e THe Batch size Trade-off

Batch size creates a fundemental trade-off in memory vs efficiency:

```
Batch Size Impact:

Small Batches (batch_size=8):
┌─────────────────────────────────────────┐
│ Memory: 8 × 28 × 28 × 4 bytes = 25KB    │ ← Low memory
│ Overhead: High (many small batches)     │ ← High overhead
│ GPU Util: Poor (underutilized)          │ ← Poor efficiency
└─────────────────────────────────────────┘

Large Batches (batch_size=512):
┌─────────────────────────────────────────┐
│ Memory: 512 × 28 × 28 × 4 bytes = 1.6MB │ ← Higher memory
│ Overhead: Low (fewer large batches)     │ ← Lower overhead
│ GPU Util: Good (well utilized)          │ ← Better efficiency
└─────────────────────────────────────────┘
```

### Shuffling Overheard Analysis

Shuffling seems simple but it comes at a cost let's measure it:

```
Shuffle Operation Breakdown:

1. Index Generation:    O(n) - create [0, 1, 2, ..., n-1]
2. Shuffle Operation:   O(n) - randomize the indices
3. Sample Access:       O(1) per sample - dataset[shuffled_idx]

Memory Impact:
- No Shuffle: 0 extra memory (sequential access)
- With Shuffle: 8 bytes × dataset_size (store indices)

For 50,000 samples: 8 × 50,000 = 400KB extra memory
```

The key insight is shuffling overheard is typically negligible compared to the actual data loading and tensor operations.

### Pipeline Bottleneck Identification

We measure three critical metrics i.e:

1.**Throughput**: Samples processed per second
2. **Memory Usage** : Peak memory during batch loading
3. **Overheard**: Time spent on data vs computation

These measurements will reveal whether our pipeline is CPU-bound(slow data loading) or compute-bound (slow model).
The analyis is in the **analyze_dataloader_performance.py**
"""
"""
## Understanding the Data Pipeline

## The Data Pipeline Journey
Imagine you have 50,000 images of cats and dogs, you want to train a neural network to classify them:

```
Raw Data Storage          Dataset Interface         DataLoader Batching         Training Loop
┌─────────────────┐      ┌──────────────────┐      ┌────────────────────┐      ┌─────────────┐
│ cat_001.jpg     │      │ dataset[0]       │      │ Batch 1:           │      │ model(batch)│
│ dog_023.jpg     │ ───> │ dataset[1]       │ ───> │ [cat, dog, cat]    │ ───> │ optimizer   │
│ cat_045.jpg     │      │ dataset[2]       │      │ Batch 2:           │      │ loss        │
│ ...             │      │ ...              │      │ [dog, cat, dog]    │      │ backward    │
│ (50,000 files)  │      │ dataset[49999]   │      │ ...                │      │ step        │
└─────────────────┘      └──────────────────┘      └────────────────────┘      └─────────────┘
```

### Why this pipeline matters
**Individual Access(Dataset)**: Neural networks can't process 50,000 files at once. We neeed a way to access one sample at a time i.e "Give me image #1247"

**Batching Processing(DataLoader)**: GPUs are parrallel machines ,they much faster in processing 32 images simultanously than 1 image 32 times.

**Memory Efficiency** : loading all 50,000 images into memory would require ~150GB. Instead, we load only the current batch (~150MB).

**Training Variety**: Shuffling ensures the model sees different combinations each epochs, preventing memorization.


## The Dataset Abstraction

The Dataset class provides a uniform interface for accessing data regardless of whether it's stored as files, in memory in database or generated on-the-fly:


```
Dataset Interface

┌─────────────────────────────────────┐
│ __len__()     → "How many samples?" │
│ __getitem__(i) → "Give me sample i" │
└─────────────────────────────────────┘
          ↑                ↑
     Enables for     Enables indexing
    loops/iteration   dataset[index]
```

**Connection to systems**: This abstraction is crucial because it seperates *how data is stored* from *how it's accessed*, enabling optimization like caching, prefetching and parallel loading.

"""

"""
##TensorDataset- this is when data lives in Memory


##Understanding TensorDataset Structure
TensorDataset takes multiple tensors and aligns them by their first dimension i.e  the sample dimension.

```
Input Tensors (aligned by first dimension):
  Features Tensor        Labels Tensor         Metadata Tensor
  ┌─────────────────┐   ┌───────────────┐     ┌─────────────────┐
  │ [1.2, 3.4, 5.6] │   │ 0 (cat)       │     │ "image_001.jpg" │ ← Sample 0
  │ [2.1, 4.3, 6.5] │   │ 1 (dog)       │     │ "image_002.jpg" │ ← Sample 1
  │ [3.0, 5.2, 7.4] │   │ 0 (cat)       │     │ "image_003.jpg" │ ← Sample 2
  │ ...             │   │ ...           │     │ ...             │
  └─────────────────┘   └───────────────┘     └─────────────────┘
        (N, 3)               (N,)                   (N,)

Dataset Access:
  dataset[1] → (Tensor([2.1, 4.3, 6.5]), Tensor(1), "image_002.jpg")
```

### Why TensorDataset is poerful
**Memory Allocation**: All data is pre-loaded ans stored contigously in memory, enabling fast access patterns.

**Vectorized Operations**:since everything is already tensors, no conversion overhead during training.

**Supervised Learning Perfect**: Naturally handles (features,labels) pairs, plus any additional metadata.

**Batch-Friendly**: When DataLoader needs a batch, it can slice multiple samples efficiently.

##Real world Usage pattern.

```
# Computer Vision
images = Tensor(shape=(50000, 32, 32, 3))  # CIFAR-10 images
labels = Tensor(shape=(50000,))            # Class labels 0-9
dataset = TensorDataset(images, labels)

# Natural Language Processing
token_ids = Tensor(shape=(10000, 512))     # Tokenized sentences
labels = Tensor(shape=(10000,))            # Sentiment labels
dataset = TensorDataset(token_ids, labels)

# Time Series
sequences = Tensor(shape=(1000, 100, 5))   # 100 timesteps, 5 features
targets = Tensor(shape=(1000, 10))         # 10-step ahead prediction
dataset = TensorDataset(sequences, targets)
```

So the TensorDataset transforms "array of data" into "dataset that serves samples."

"""

""""
## DataLoader

The dataloader is the component which transforms individula dataset samples into batches that  neural networks crave.
This where data loading becomes a systems challenge.

### Understanding Batching: From Samples to Tensors
Dataloader performs a crucial transformation as it collects individual samples and stacks them into batch tensors:

```
Step 1: Individual Samples from Dataset
  dataset[0] → (features: [1, 2, 3], label: 0)
  dataset[1] → (features: [4, 5, 6], label: 1)
  dataset[2] → (features: [7, 8, 9], label: 0)
  dataset[3] → (features: [2, 3, 4], label: 1)

Step 2: DataLoader Groups into Batch (batch_size=2)
  Batch 1:
    features: [[1, 2, 3],    ← Stacked into shape (2, 3)
               [4, 5, 6]]
    labels:   [0, 1]         ← Stacked into shape (2,)

  Batch 2:
    features: [[7, 8, 9],    ← Stacked into shape (2, 3)
               [2, 3, 4]]
    labels:   [0, 1]         ← Stacked into shape (2,)
```

### The Shuffling Process

The shuffling process randomizes which samples appear in which batches which is crucial for good training.

```
Without Shuffling (epoch 1):          With Shuffling (epoch 1):
  Batch 1: [sample 0, sample 1]         Batch 1: [sample 2, sample 0]
  Batch 2: [sample 2, sample 3]         Batch 2: [sample 3, sample 1]
  Batch 3: [sample 4, sample 5]         Batch 3: [sample 5, sample 4]

Without Shuffling (epoch 2):          With Shuffling (epoch 2):
  Batch 1: [sample 0, sample 1]  ✗      Batch 1: [sample 1, sample 4]  ✓
  Batch 2: [sample 2, sample 3]  ✗      Batch 2: [sample 0, sample 5]  ✓
  Batch 3: [sample 4, sample 5]  ✗      Batch 3: [sample 2, sample 3]  ✓

  (Same every epoch = overfitting!)     (Different combinations = better learning!)
```

### Dataloader as a System Component

**Memory management**: Dataloaders  only holds one batch in memory at a time, not the entire dataset.
**Iteration Interface**: Provides Python iterator protocol so training loops can use `for batch in dataloader:`.
**Collation Strategy**: Automatically stacks tensors from individual samples into batch tensors
**Performance Critical**: This is often the bottleneck in the training pipelines -loading and preparing data can be slower than the forward pass.

###Dataloader Algorithm
```
1.Create indices list: [0,1,2,...,dataset_length-1]
2.If shuffle=True: randomly shuffle the indices
3.Group indices into chunks of batch_size
4.For each chunk:
   a. Retrieve samples: [dataset[i] for i in chunk]
   b. Collate samples: stack individual tensors into batch tensors
   c. Yield the batch tensor tuple
```
This transfroms the dataset from "acces one sample" to "iterate through batches" .
""""

"""
### Data Augmentation
Data augmentation is crucial as it prevents overfitting through variety.

Data Augmentation is one of the most effective techniques for improving model generalization.
We do it by applying random transformations during training, artificially expand the dataset and force the model to learn robust, invariant features.

```
Without Augmentation:                With Augmentation:
Model sees exact same images         Model sees varied versions
every epoch                          every epoch

Cat photo #247                       Cat #247 (original)
Cat photo #247                       Cat #247 (flipped)
Cat photo #247                       Cat #247 (cropped left)
Cat photo #247                       Cat #247 (cropped right)
     ↓                                    ↓
Model memorizes position             Model learns "cat-ness"
Overfits to training set             Generalizes to new cats
```

```
RandomHorizontalFlip (50% probability):
┌──────────┐     ┌──────────┐
│  🐱 →    │  →  │    ← 🐱  │
│          │     │          │
└──────────┘     └──────────┘
Cars, cats, dogs look similar when flipped!

RandomCrop with Padding:
┌──────────┐     ┌────────────┐     ┌──────────┐
│   🐱     │  →  │░░░░░░░░░░░░│  →  │  🐱      │
│          │     │░░  🐱     ░│     │          │
└──────────┘     │░░░░░░░░░░░░│     └──────────┘
  Original        Pad edges        Random crop
                  (with zeros)     (back to 32×32)
```

## Training vs Evaluation

**Critical**- augmentation applies ONLY during training!


```
Training:                              Evaluation:
┌─────────────────┐                   ┌─────────────────┐
│ Original Image  │                   │ Original Image  │
│      ↓          │                   │      ↓          │
│ Random Flip     │                   │ (no transforms) │
│      ↓          │                   │      ↓          │
│ Random Crop     │                   │ Direct to Model │
│      ↓          │                   └─────────────────┘
│ To Model        │
└─────────────────┘
```
"""

"""
### Understanding Image Data

Images are just 2D arrays of numbers (pixels). Here actual 8x8 handwritten digits:

```
Digit "5" (8×8):        Digit "3" (8×8):        Digit "8" (8×8):
 0  0 12 13  5  0  0  0   0  0 11 12  0  0  0  0   0  0 10 14  8  1  0  0
 0  0 13 15 10  0  0  0   0  2 16 16 16  7  0  0   0  0 16 15 15  9  0  0
 0  3 15 13 16  7  0  0   0  0  8 16  8  0  0  0   0  0 15  5  5 13  0  0
 0  8 13  6 15  4  0  0   0  0  0 12 13  0  0  0   0  1 16  5  5 13  0  0
 0  0  0  6 16  5  0  0   0  0  1 16 15  9  0  0   0  6 16 16 16 16  1  0
 0  0  5 15 16  9  0  0   0  0 14 16 16 16  7  0   1 16  3  1  1 15  1  0
 0  0  9 16  9  0  0  0   0  5 16  8  8 16  0  0   0  9 16 16 16 15  0  0
 0  0  0  0  0  0  0  0   0  3 16 16 16 12  0  0   0  0  0  0  0  0  0  0

Visual representation:
░█████░          ░█████░          ░█████░
░█░░░█░          ░░░░░█░          █░░░░█░
░░░░█░░          ░░███░░          ░█████░
░░░█░░░          ░░░░█░░          █░░░░█░
░░█░░░░          ░█████░          ░█████░
```

**Shape Transformations in Dataloader:**

```
Individual Sample (from Dataset):
  image: (8, 8)      ← Single 8×8 image
  label: scalar      ← Single digit (0-9)

After DataLoader batching (batch_size=32):
  images: (32, 8, 8)  ← Stack of 32 images
  labels: (32,)       ← Array of 32 labels

This is what the model sees during training!
```
We have buit the **data loading infrastructure** that powers all modern ML:
-Dataset abstraction 
- TensorDataset - wraps one or more tensors into a single dataset object.
- Dataloader - resposible for batching, shuffling, iteration.
- Data Augmentation - expands the size and diversity of a training dataset without collecting new samples.
**Real-world connections:** So we have implemented the same patterns as:
- Pytorch's `torch.utils.data.DataLoader`
- Pytorch's `torchvision.transforms`
- TensorFlow's `tf.data.Dataset`


"""

"""
## System Analysis 
Now let's understand where time and memory go since it's crucial for building ML systems.
In a typical training step, time is split between data loading and computation:

```
Training Step Breakdown:
┌─────────────────────────────────────────────────────────────┐
│ Data Loading        │ Forward Pass     │ Backward Pass      │
│ ████████████        │ ███████          │ ████████           │
│ 40ms                │ 25ms             │ 35ms               │
└─────────────────────────────────────────────────────────────┘
              100ms total per step

Bottleneck Analysis:
- If data loading > forward+backward: "Data starved" (CPU bottleneck) i.e when the CPU cannot process data fast enough for the GPU to process.
- If forward+backward > data loading: "Compute bound" (GPU bottleneck) i.e when the GPU is the limiting factor because it is fuly saturated with the computational load itself.
- Ideal: Data loading ≈ computation time (balanced pipeline)
```

### Memory Scaling i.e THe Batch size Trade-off

Batch size creates a fundemental trade-off in memory vs efficiency:

```
Batch Size Impact:

Small Batches (batch_size=8):
┌─────────────────────────────────────────┐
│ Memory: 8 × 28 × 28 × 4 bytes = 25KB    │ ← Low memory
│ Overhead: High (many small batches)     │ ← High overhead
│ GPU Util: Poor (underutilized)          │ ← Poor efficiency
└─────────────────────────────────────────┘

Large Batches (batch_size=512):
┌─────────────────────────────────────────┐
│ Memory: 512 × 28 × 28 × 4 bytes = 1.6MB │ ← Higher memory
│ Overhead: Low (fewer large batches)     │ ← Lower overhead
│ GPU Util: Good (well utilized)          │ ← Better efficiency
└─────────────────────────────────────────┘
```

### Shuffling Overheard Analysis

Shuffling seems simple but it comes at a cost let's measure it:

```
Shuffle Operation Breakdown:

1. Index Generation:    O(n) - create [0, 1, 2, ..., n-1]
2. Shuffle Operation:   O(n) - randomize the indices
3. Sample Access:       O(1) per sample - dataset[shuffled_idx]

Memory Impact:
- No Shuffle: 0 extra memory (sequential access)
- With Shuffle: 8 bytes × dataset_size (store indices)

For 50,000 samples: 8 × 50,000 = 400KB extra memory
```

The key insight is shuffling overheard is typically negligible compared to the actual data loading and tensor operations.

## Practical Examples

Let's look at how to use these components in code.

### 1. Creating a Custom Dataset
If your data isn't already in memory as tensors (e.g., it's in CSV files or images on disk), you can create a custom `Dataset` class.

```python
from dataloader import Dataset
from Tensor import Tensor
import numpy as np

class MyCustomDataset(Dataset):
    def __init__(self, data_size=100):
        # In a real app, you might load file paths here
        self.features = np.random.randn(data_size, 10)
        self.labels = np.random.randint(0, 2, size=(data_size,))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Return a single sample as (feature_tensor, label_tensor)
        x = Tensor(self.features[idx])
        y = Tensor(self.labels[idx])
        return x, y

dataset = MyCustomDataset(data_size=1000)
print(f"Dataset size: {len(dataset)}")
```

### 2. Using TensorDataset
For data that is already loaded into memory as `Tensor` objects, `TensorDataset` is the easiest way to wrap them.

```python
from dataloader import TensorDataset
from Tensor import Tensor
import numpy as np

# Create some dummy data
X = Tensor(np.random.randn(5000, 32, 32, 3)) # 5000 RGB images
Y = Tensor(np.random.randint(0, 10, size=(5000,))) # 10 classes

# Wrap them in a dataset
dataset = TensorDataset(X, Y)

# Access a single sample
x_0, y_0 = dataset[0]
print(f"First image shape: {x_0.data.shape}")
```

### 3. Using the DataLoader
The `DataLoader` handles batching and shuffling for you. This is what you actually iterate over during training.

```python
from dataloader import Dataloader

# Create a loader
train_loader = Dataloader(dataset, batch_size=32, shuffle=True)

print(f"Number of batches: {len(train_loader)}")

# Iterate through the data
for batch_idx, (images, labels) in enumerate(train_loader):
    # images.data.shape will be (32, 32, 32, 3)
    # labels.data.shape will be (32,)
    
    # training_step(images, labels)
    if batch_idx == 0:
        print(f"Batch 0 images shape: {images.data.shape}")
        break
```

### 4. Implementing Data Augmentation
Use `Compose` to chain multiple transformations together.

```python
from dataloader import RandomHorizontalFlip, RandomCrop, Compose

# Define a pipeline
transform = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomCrop(size=32, padding=4)
])

# Apply to an image (Tensor or NumPy array)
img = Tensor(np.random.randn(32, 32, 3))
augmented_img = transform(img)

# In a real custom dataset, you would apply this in __getitem__:
class AugmentedDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            img = self.transform(img)
            
        return img, label
```

### 5. Full Pipeline Example
Putting it all together for a training loop structure.

```python
# 1. Prepare data
X_train = Tensor(np.random.randn(1000, 3, 32, 32))
Y_train = Tensor(np.random.randint(0, 10, size=(1000,)))

# 2. Define transforms
train_transform = Compose([
    RandomHorizontalFlip(p=0.5),
    RandomCrop(32, padding=4)
])

# 3. Create Dataset and DataLoader
train_ds = AugmentedDataset(X_train, Y_train, transform=train_transform)
train_loader = Dataloader(train_ds, batch_size=64, shuffle=True)

# 4. Training Loop
epochs = 5
for epoch in range(epochs):
    total_loss = 0
    for images, labels in train_loader:
        # 1. Forward pass
        # outputs = model(images)
        # loss = criterion(outputs, labels)
        
        # 2. Backward pass
        # loss.backward()
        
        # 3. Update weights
        # optimizer.step()
        # optimizer.zero_grad()
        pass
    
    print(f"Epoch {epoch+1} complete")
```

