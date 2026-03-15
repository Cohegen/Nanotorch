
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

### Pipeline Bottleneck Identification

We measure three critical metrics i.e:

1.**Throughput**: Samples processed per second
2. **Memory Usage** : Peak memory during batch loading
3. **Overheard**: Time spent on data vs computation

These measurements will reveal whether our pipeline is CPU-bound(slow data loading) or compute-bound (slow model).
The analysis is in the **analyze_dataloader_performance.py**
