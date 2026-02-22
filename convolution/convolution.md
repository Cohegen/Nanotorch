## Introduction to Convolutions

###1. Introduction to Spatial Operations
Spatial operations are operations that transform machine learning from working with simple vector to understanding images and spatial patterns.
As humans when we look at a photo, our brains naturally processes spatial relationships,including edges, textures and objects.
Spatial operations give neural networks this similar capability.
However a formal definition of spatial operations is that they are  image processing operations where the output pixel value depends on the values of neighboring pixels which allows the model to capture spatial relationships and structural information in the image.

#### The Core Spatial Operations
**Convolution**: Detects local patterns by sliding filters (kernel) across the input
**Pooling**:Reduces spatial dimensions while preserving important features.

### Convolution in action
![Alt text](https://github.com/Cohegen/Nanotorch/blob/main/assets/convolution_kernel.png)

```
Input Image (5×5):        Kernel (3×3):        Output (3×3):
┌─────────────────┐      ┌───────────┐       ┌─────────┐
│ 1  2  3  4  5   │      │  1  0  -1 │       │ ?  ?  ? │
│ 6  7  8  9  0   │  *   │  1  0  -1 │   =   │ ?  ?  ? │
│ 1  2  3  4  5   │      │  1  0  -1 │       │ ?  ?  ? │
│ 6  7  8  9  0   │      └───────────┘       └─────────┘
│ 1  2  3  4  5   │
└─────────────────┘

Sliding Window Process:
Position (0,0): [1,2,3]   Position (0,1): [2,3,4]   Position (0,2): [3,4,5]
               [6,7,8] *               [7,8,9] *               [8,9,0] *
               [1,2,3]                 [2,3,4]                 [3,4,5]
               = Output[0,0]           = Output[0,1]           = Output[0,2]
```
Each output pixel summarizes a local neighborhood,allowing the network to detect patterns like edges,corners and textures.

```
Without Convolution:                    With Convolution:
32×32×3 image = 3,072 inputs          32×32×3 → Conv → 32×32×16
↓                                      ↓                     ↓
Dense(3072 → 1000) = 3M parameters    Shared 3×3 kernel = 432 parameters
↓                                      ↓                     ↓
Memory explosion + no spatial awareness Efficient + preserves spatial structure
```
Convolution achieves dramatic parameter reduction (1000x fewer!) while preserving the spatial relationships that matter for visual understanding.

## Mathematical Foundations for Convolution
Convolution is just "sliding window multiplication and summation".
```
Step 1: Position the kernel over input
Input:          Kernel:
┌─────────┐     ┌─────┐
│ 1 2 3 4 │     │ 1 0 │  ← Place kernel at position (0,0)
│ 5 6 7 8 │  ×  │ 0 1 │
│ 9 0 1 2 │     └─────┘
└─────────┘

Step 2: Multiply corresponding elements
Overlap:        Computation:
┌─────┐         1×1 + 2×0 + 5×0 + 6×1 = 1 + 0 + 0 + 6 = 7
│ 1 2 │
│ 5 6 │
└─────┘

Step 3: Slide kernel and repeat
Position (0,1):  Position (1,0):  Position (1,1):
┌─────┐         ┌─────┐          ┌─────┐
│ 2 3 │         │ 5 6 │          │ 6 7 │
│ 6 7 │         │ 9 0 │          │ 0 1 │
└─────┘         └─────┘          └─────┘
Result: 9       Result: 5        Result: 7

Final Output:  ┌─────┐
               │ 7 9 │
               │ 5 7 │
               └─────┘
```
###  The Mathematical Formula
For 2D convolution, we slide the kernel K across input I:
```
O[i,j] = Σ Σ I[i+m, j+n] × K[m,n]
         m n
```
This formula captures the "multiply and sum" operation for each kernel position.

### Pooling: Spatial Summarization

```
Max Pooling Example (2x2 window):
Input:             Output:
┌───────────────┐  ┌───────┐
│ 1  3  2  4    │  │ 6   8 │  ← max([1,3,5,6])=6, max([2,4,7,8])=8
│ 5  6  7  8    │  │ 9   9 │  ← max([2,9,0,1])=9, max([1,3,9,3])=9
│ 2  9  1  3    │  └───────┘
│ 0  1  9  3    │
└───────────────┘

Average Pooling (same window):
┌─────────────┐
│ 3.75   5.25 │  ← avg([1,3,5,6])=3.75, avg([2,4,7,8])=5.25
│ 3.0    4.0  │  ← avg([2,9,0,1])=3.0, avg([1,3,9,3])=4.0
└─────────────┘
```
### Essense of this Complexity
For convolution with input (1,3,224,224) and kernel (64,3,3,3):
-**Operations**: 1x64x3x3x3x224x224 = 86.7 million multiply-adds operations
-**Memory**: Input(600KB) + Weight (6.9KB)+ Output(12.8MB) =~ 13.4MB

This is why kernel size matters enormously i.e a 7x7 kernel would require 5.4x more computation.

### Key Properties that Enable Deep Learning
**Translation Equivariance**:move the cat , then detection moves the same way
**Parameter sharing**:same edge detector works everywhere in the image
**Local Connectivity**: Each output only looks at nearby inputs
**Hierarchial Features**:Early layers detect edges then later layers detect objects

## How to build spatial operations

### Conv2d(Detecting Patterns with sliding windows)
Conv2D is a spatial operation that applies a filter(kernel) over a 2D input (image or feature map) to extract features like textures,edges and patterns.
![Alt text](assets/conv2d.png)
Here the kernel slides across the image and computes weighted sums.

```
Convolution Visualization:
Input (4×4):              Kernel (3×3):           Output (2×2):
┌─────────────┐          ┌─────────┐             ┌─────────┐
│ a b c d │            │ k1 k2 k3│             │ o1  o2 │
│ e f g h │     ×      │ k4 k5 k6│      =      │ o3  o4 │
│ i j k l │            │ k7 k8 k9│             └─────────┘
│ m n o p │            └─────────┘
└─────────────┘

Computation Details:
o1 = a×k1 + b×k2 + c×k3 + e×k4 + f×k5 + g×k6 + i×k7 + j×k8 + k×k9
o2 = b×k1 + c×k2 + d×k3 + f×k4 + g×k5 + h×k6 + j×k7 + k×k8 + l×k9
o3 = e×k1 + f×k2 + g×k3 + i×k4 + j×k5 + k×k6 + m×k7 + n×k8 + o×k9
o4 = f×k1 + g×k2 + h×k3 + j×k4 + k×k5 + l×k6 + n×k7 + o×k8 + p×k9
```

### The Six Nested Loops of Convolution
Our implementation in convolutions.py will use explicit loops to ashow exactly where the computational cost came from:
```
for batch in range(B):          # Loop 1: Process each sample
    for out_ch in range(C_out):     # Loop 2: Generate each output channel
        for out_h in range(H_out):      # Loop 3: Each output row
            for out_w in range(W_out):      # Loop 4: Each output column
                for k_h in range(K_h):          # Loop 5: Each kernel row
                    for k_w in range(K_w):          # Loop 6: Each kernel column
                        for in_ch in range(C_in):       # Loop 7: Each input channel
                            # The actual multiply-accumulate operation
                            result += input[...] * kernel[...]
```
Input shape = (B,C_in,H_in,W_in)
Kernel shape = (C_out,C_in,K_h,K_w)
Output shape = (B,C_out,H_out,W_out)
Where:
B = batch size
C_in = input channels
C_out = output channels
H_out,W_out = output spatial size
K_h,K_w = kernel size


Total operations: B × C_out × H_out × W_out × K_h × K_w × C_in

For typical values (B=32, C_out=64, H_out=224, W_out=224, K_h=3, K_w=3, C_in=3):
That's 32 × 64 × 224 × 224 × 3 × 3 × 3 = **2.8 billion operations** per forward pass!

### How Conv2d transforms Machine Learning

```
Before Conv2d (Dense Only):         After Conv2d (Spatial Aware):
Input: 32×32×3 = 3,072 values      Input: 32×32×3 structured as image
         ↓                                   ↓
Dense(3072→1000) = 3M params       Conv2d(3→16, 3×3) = 448 params
         ↓                                   ↓
No spatial awareness               Preserves spatial relationships
Massive parameter count            Parameter sharing across space
```
### Weight Intialization : He intialization for ReLu networks

Our Conv2duses He initialization, specifically designed for ReLU activations:
-The problem is that wrong initialization leads to vanishing or exploding gradients
-The solution is std = sqrt(2/ fan_in) where fan_in = channels x kernel_height x kernel_width
Why solution works because it maintains variance through ReLU nonlinearity

### The 6-Loop Implementation Strategy
In convolutions.py we will implement convolution with explicit loops to show the true computational cost:
```
Nested Loop Structure:
for batch:           ← Process each sample in parallel (in practice)
  for out_channel:   ← Generate each output feature map
    for out_h:       ← Each row of output
      for out_w:     ← Each column of output
        for k_h:     ← Each row of kernel
          for k_w:   ← Each column of kernel
            for in_ch: ← Accumulate across input channels
              result += input[...] * weight[...]
```
This reveals why convolution is expensive by O(n**6) i.e O(B×C_out×H×W×K_h×K_w×C_in) operations!

## Pooling Operations
Pooling operations compress spatial information while keeping the most important features.

### MaxPool2d
Max pooling finds the strongest activation in each window, preserving sharp features like edges and corners.

```
MaxPool2d Example (2×2 kernel, stride=2):
Input (4×4):              Windows:               Output (2×2):
┌─────────────┐          ┌─────┬─────┐          ┌───────┐
│ 1  3 │ 2  8 │          │ 1 3 │ 2 8 │          │ 6   8 │
│ 5  6 │ 7  4 │    →     │ 5 6 │ 7 4 │    →     │ 9   7 │
├──────┼──────┤          ├─────┼─────┤          └───────┘
│ 2  9 │ 1  7 │          │ 2 9 │ 1 7 │
│ 0  1 │ 3  6 │          │ 0 1 │ 3 6 │
└─────────────┘          └─────┴─────┘

Window Computations:
Top-left: max(1,3,5,6) = 6     Top-right: max(2,8,7,4) = 8
Bottom-left: max(2,9,0,1) = 9  Bottom-right: max(1,7,3,6) = 7
```

### AvgPool2d(Smoothing Local Features)
Average pooling computes the mean of each window,creating smoother and more general features.

```
AvgPool2d Example (same 2×2 kernel, stride=2):
Input (4×4):              Output (2×2):
┌─────────────┐          ┌─────────────┐
│ 1  3 │ 2  8 │          │ 3.75   5.25 │
│ 5  6 │ 7  4 │    →     │ 3.0    4.25 │
├──────┼──────┤          └─────────────┘
│ 2  9 │ 1  7 │
│ 0  1 │ 3  6 │
└─────────────┘

Window Computations:
Top-left: (1+3+5+6)/4 = 3.75    Top-right: (2+8+7+4)/4 = 5.25
Bottom-left: (2+9+0+1)/4 = 3.0  Bottom-right: (1+7+3+6)/4 = 4.25
```
#### When to use Average Pooling
-Global Vergae Pooling for classification
-When we want smoother,less noisy features
-When exact feature location doesn't matter
-In shallower networks where sharp features aren't critical.

```
Typical Pattern:
Feature Maps → Global Average Pool → Dense → Classification
(256×7×7)   →        (256×1×1)      → FC   →    (10)
              Replaces flatten+dense with parameter reduction
```
### Importance of Pooling 
1.It reduces spatial size for example:
```
Memory Impact:
Input: 224×224×64 = 3.2M values    After 2×2 pooling: 112×112×64 = 0.8M values
Memory reduction: 4× less!         Computation reduction: 4× less!

```
2.It filters noise: max pooling removes weak activations and keeps strong signals whereas average pooling smoothens noisy activations.

3.It reduces Overfitting since pooling reduces paramters in the later layer so with fewer parameters there's a lower risk of memorizing training data.

### Sliding Window Pattern
Both pooling operations follow the same sliding window pattern:

```
Sliding 2×2 window with stride=2:
Step 1:     Step 2:     Step 3:     Step 4:
┌──┐        ┌──┐
│▓▓│        │▓▓│
└──┘        └──┘                    ┌──┐        ┌──┐
                                    │▓▓│        │▓▓│
                                    └──┘        └──┘

Non-overlapping windows → Each input pixel used exactly once
Stride=2 → Output dimensions halved in each direction
```
![Alt text](assets/sliding_window.gif)

## Batch Normalization
Batch Normalization is one of the most important techniques for training deep networks.
It normalizes activations across  the batch dimension thus dramatically imporving training stability and speed.

### Why BatchNorm Matters

```
Without BatchNorm:                  With BatchNorm:
Layer outputs can have              Layer outputs are normalized
wildly varying scales:              to consistent scale:

Layer 1: mean=0.5, std=0.3         Layer 1: mean≈0, std≈1
Layer 5: mean=12.7, std=8.4   →    Layer 5: mean≈0, std≈1
Layer 10: mean=0.001, std=0.0003   Layer 10: mean≈0, std≈1

Result: Unstable gradients         Result: Stable training
        Slow convergence                   Fast convergence
        Careful learning rate              Robust to hyperparameters
```

### BatchNorm Explained
BatchNorm normalizes the output of each neuron across a mini-batch so that they have roughly **zero mean** and **unit variance**.
This prevents **internal covariate shift**, where the distribution of layer input changes during training.

During training:
- Earlier layers keep updating their weights
- Inputs to deeper layers shift continuously
- This makes training **slow and unstable**

BatchNorm ensures that **each neuron receives well-scaled inputs**, making gradient descent smoother and faster.

### Mathematical implementation of batchNorm

Given a mini-batch of size \(m\):

1. **Compute the batch mean for each neuron**:
\[
\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i
\]

2. **Compute the batch variance**:
\[
\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m} (x_i - \mu_B)^2
\]

3. **Normalize the activations**:
\[
\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}
\]

> Here, \(x_i\) is the output of a single neuron for sample \(i\), and \(\epsilon\) is a small constant to avoid division by zero.

4. **Scale and shift (learnable parameters)**:
\[
y_i = \gamma \hat{x}_i + \beta
\]

- \(\gamma\) → scales the normalized output  
- \(\beta\) → shifts the normalized output  

This allows the network to **restore the original distribution** if needed.


### The BatchNorm Computation in our Convolution use case

For each channel c, BatchNorm computes:

```
1. Batch Statistics (during training):
   μ_c = mean(x[:, c, :, :])     # Mean over batch and spatial dims
   σ²_c = var(x[:, c, :, :])     # Variance over batch and spatial dims

2. Normalize:
   x̂_c = (x[:, c, :, :] - μ_c) / sqrt(σ²_c + ε)

3. Scale and Shift (learnable parameters):
   y_c = γ_c * x̂_c + β_c       # γ (gamma) and β (beta) are learned
```

### Train vs Eval Mode
```
Training Mode:                      Eval Mode:
┌────────────────────┐             ┌────────────────────┐
│ Use batch stats    │             │ Use running stats  │
│ Update running     │             │ (accumulated from  │
│ mean/variance      │             │  training)         │
└────────────────────┘             └────────────────────┘
   ↓                                  ↓
Computes μ, σ² from                Uses frozen μ, σ² for
current batch                      consistent inference
```

## Building a Complete CNN

### CNN architecture (from Pixels to Predictions)
 A CNN processes images through alternating convolution and pooling layers, gradually extracting higher-level features:

```
Complete CNN Pipeline:

Input Image (32×32×3)     Raw RGB pixels
       ↓
Conv2d(3→16, 3×3)        Detect edges, textures
       ↓
ReLU Activation          Remove negative values
       ↓
MaxPool(2×2)             Reduce to (16×16×16)
       ↓
Conv2d(16→32, 3×3)       Detect shapes, patterns
       ↓
ReLU Activation          Remove negative values
       ↓
MaxPool(2×2)             Reduce to (8×8×32)
       ↓
Flatten                  Reshape to vector (2048,)
       ↓
Linear(2048→10)          Final classification
       ↓
Softmax                  Probability distribution
```
### The Parameter Efficiency Comparision
```
CNN vs Dense Network Comparison:

CNN Approach:                     Dense Approach:
┌─────────────────┐               ┌─────────────────┐
│ Conv1: 3→16     │               │ Input: 32×32×3  │
│ Params: 448     │               │ = 3,072 values  │
├─────────────────┤               ├─────────────────┤
│ Conv2: 16→32    │               │ Hidden: 1,000   │
│ Params: 4,640   │               │ Params: 3M+     │
├─────────────────┤               ├─────────────────┤
│ Linear: 2048→10 │               │ Output: 10      │
│ Params: 20,490  │               │ Params: 10K     │
└─────────────────┘               └─────────────────┘
Total: ~25K params                Total: ~3M params

```
### Spatial Hierarchy i.e Why this architecture works
```
Layer-by-Layer Feature Evolution:

Layer 1 (Conv 3→16):              Layer 2 (Conv 16→32):
┌─────┐ ┌─────┐ ┌─────┐           ┌─────┐ ┌──────┐ ┌───────┐
│Edge │ │Edge │ │Edge │           │Shape│ │Corner│ │Texture│
│ \\ /│ │  |  │ │ / \\│           │ ◇   │ │  L   │ │ ≈≈≈≈≈ │
└─────┘ └─────┘ └─────┘           └─────┘ └──────┘ └───────┘
Simple features                   Complex combinations

Why pooling between layers:
-Reduces computation for next layer
-Increases receptive field (each conv sees larger input area)
-Provides translation invariance (cat moved 1 pixel still detected)
```



