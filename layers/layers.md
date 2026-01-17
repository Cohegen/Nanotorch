## Introduction of Neural Network Layers

Neural network layers are fundemental building blocks that transform data as flows through a network. Each layer performs a specific computation: 

- **Linear Layers** apply learned transformations: `y = xW + b`
- **Dropout Layers** randomly zero elements for regularization.

-Each layer learns its own piece of the puzzle. Linear layers learn which features matter, while dropout prevents overfitting by forcing robustness.

"""
## Mathematical Background
A linear layer implements: **y = xW + b**

````
Input x (batch_size, in_features) @ Weight W (in_features, out_features) + bias(out_features)
= Output y (batch_size, out_features)
````
### Weight Initialization
Random intialization is crucial for breaking symmmetry:
- **Xavier/Glorot**: Scale by sqrt(1/fan_in) for stable gradients
- **He** - scale by srt(2/fan_in) for ReLU activation.
- **To small**: Gradients vanish, learning is slow
- **To large** : Gradients explode, training is unstable

## Parameter Counting
```
Linear(784,256): 784x256 + 256 = 200,960 parameters

Manual Compostion:
    layer1 = Linear(784,256) #200,960 params
    activation =ReLU() # 0 params
    layer2 = Linear(256,10) #2,570 params
                            #Total: 203,530 params
```

Memort usage: 4 bytes/param x 203,530 =~ 814kb for weights alone
"""

"""
##Linear Layer- The Foundation of Neural Networks

Linear layers also called Dense or Fully Connected layers are fundemental building blocks of neural networks.They implement the mathematical operation:
 **y = xW + b**

Where:
- **x**: input features (what we know)
- **W**: weight matrix (what we learn)
- **b**: bias vector (adjusts the output)
- **y**: Output features (what we predict)

### Why Linear Layers Matter
Linear layers learn **feature combinations**. Each output neuron asks: "What combination of input features is most useful for my task?" The netwoek discovers these combinations through training.

###Data Flow visual
```
Input Features    Weight Matrix          Bias Vector   Output Features
[batch, in_feat] @ [in_feat,out_feat]     [out_feat] = [batch,out_feat]

Example: MNIST Digit Recognition
[32, 784]       @  [784, 10]          + [10]        =  [32, 10]
  ↑                   ↑                    ↑             ↑
32 images         784 pixels          10 classes    10 probabilities
                  to 10 classes       adjustments   per image
```
### Memory Layout
```
Linear(784, 256) Parameters:
┌─────────────────────────────┐
│ Weight Matrix W             │  784 × 256 = 200,704 params
│ [784, 256] float32          │  × 4 bytes = 802.8 KB
├─────────────────────────────┤
│ Bias Vector b               │  256 params
│ [256] float32               │  × 4 bytes = 1.0 KB
└─────────────────────────────┘
                Total: 803.8 KB for one layer
```
-The code for Linear layer is in the python script **layers.py**
"""

"""
### Dropoout layer - prevents overfitting
Dropout is regularization technique that randomly "turns off" neurons during training. This forces the network to not rely heavily on any single neuron, making it more robust and generalizable.

### Why Dropout Matters
**The Problem** : Neural networks can memorize training data instead of learning generalizable patterns. This leads to poor performance on new, unseen data.

**The Solution**: Dropout randomly zeros out neurons, forcing the network to learn multiple indepedent ways to solve the problem.

### Dropout in Action
```
Training Mode (p=0.5 dropout):
Input:  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
         ↓ Random mask with 50% survival rate
Mask:   [1,   0,   1,   0,   1,   1,   0,   1  ]
         ↓ Apply mask and scale by 1/(1-p) = 2.0
Output: [2.0, 0.0, 6.0, 0.0, 10.0, 12.0, 0.0, 16.0]

Inference Mode (no dropout):
Input:  [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
         ↓ Pass through unchanged
Output: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
```

### Training vs Inference
```
                Training Mode              Inference Mode
               ┌─────────────────┐        ┌─────────────────┐
Input Features │ [×] [ ] [×] [×] │        │ [×] [×] [×] [×] │
               │ Active Dropped  │   →    │   All Active    │
               │ Active Active   │        │                 │
               └─────────────────┘        └─────────────────┘
                      ↓                           ↓
                "Learn robustly"            "Use all knowledge"
```
### Memory and Performance
```
Dropout Memory Usage:
┌─────────────────────────────┐
│ Input Tensor: X MB          │
├─────────────────────────────┤
│ Random Mask: X/4 MB         │  (boolean mask, 1 byte/element)
├─────────────────────────────┤
│ Output Tensor: X MB         │
└─────────────────────────────┘
        Total: ~2.25X MB peak memory

Computational Overhead: Minimal (element-wise operations)
```

"""

"""
## Sequential - layer container for composition
`Sequential` chains layer together, calling forward() on each in order.


"""

"""
### Network Architecture Visualization
```
MNIST Classification Network (3-Layer MLP):

    Input Layer          Hidden Layer 1        Hidden Layer 2        Output Layer
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│     784         │    │      256        │    │      128        │    │       10        │
│   Pixels        │───▶│   Features      │───▶│   Features      │───▶│    Classes      │
│  (28×28 image)  │    │   + ReLU        │    │   + ReLU        │    │  (0-9 digits)   │
│                 │    │   + Dropout     │    │   + Dropout     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        ↓                       ↓                       ↓                       ↓
   "Raw pixels"          "Edge detectors"        "Shape detectors"        "Digit classifier"

Data Flow:
[32, 784] → Linear(784,256) → ReLU → Dropout(0.5) → Linear(256,128) → ReLU → Dropout(0.3) → Linear(128,10) → [32, 10]
```

### Parameter Count Analysis
```
Parameter Breakdown (Manual Layer Composition):
┌─────────────────────────────────────────────────────────────┐
│ layer1 = Linear(784 → 256)                                  │
│   Weights: 784 × 256 = 200,704 params                       │
│   Bias:    256 params                                       │
│   Subtotal: 200,960 params                                  │
├─────────────────────────────────────────────────────────────┤
│ activation1 = ReLU(), dropout1 = Dropout(0.5)               │
│   Parameters: 0 (no learnable weights)                      │
├─────────────────────────────────────────────────────────────┤
│ layer2 = Linear(256 → 128)                                  │
│   Weights: 256 × 128 = 32,768 params                        │
│   Bias:    128 params                                       │
│   Subtotal: 32,896 params                                   │
├─────────────────────────────────────────────────────────────┤
│ activation2 = ReLU(), dropout2 = Dropout(0.3)               │
│   Parameters: 0 (no learnable weights)                      │
├─────────────────────────────────────────────────────────────┤
│ layer3 = Linear(128 → 10)                                   │
│   Weights: 128 × 10 = 1,280 params                          │
│   Bias:    10 params                                        │
│   Subtotal: 1,290 params                                    │
└─────────────────────────────────────────────────────────────┘
                    TOTAL: 235,146 parameters
                    Memory: ~940 KB (float32)
```

"""

"""
### Memory Analysis Overview
```
Layer Memory Components:
┌─────────────────────────────────────────────────────────────┐
│                    PARAMETER MEMORY                         │
├─────────────────────────────────────────────────────────────┤
│ • Weights: Persistent, shared across batches                │
│ • Biases: Small but necessary for output shifting           │
│ • Total: Grows with network width and depth                 │
├─────────────────────────────────────────────────────────────┤
│                   ACTIVATION MEMORY                         │
├─────────────────────────────────────────────────────────────┤
│ • Input tensors: batch_size × features × 4 bytes            │
│ • Output tensors: batch_size × features × 4 bytes           │
│ • Intermediate results during forward pass                  │
│ • Total: Grows with batch size and layer width              │
├─────────────────────────────────────────────────────────────┤
│                   TEMPORARY MEMORY                          │
├─────────────────────────────────────────────────────────────┤
│ • Dropout masks: batch_size × features × 1 byte             │
│ • Computation buffers for matrix operations                 │
│ • Total: Peak during forward/backward passes                │
└─────────────────────────────────────────────────────────────┘
```

### Computational Complexity Overview
```
Layer Operation Complexity:
┌─────────────────────────────────────────────────────────────┐
│ Linear Layer Forward Pass:                                  │
│   Matrix Multiply: O(batch × in_features × out_features)    │
│   Bias Addition: O(batch × out_features)                    │
│   Dominant: Matrix multiplication                           │
├─────────────────────────────────────────────────────────────┤
│ Multi-layer Forward Pass:                                   │
│   Sum of all layer complexities                             │
│   Memory: Peak of all intermediate activations              │
├─────────────────────────────────────────────────────────────┤
│ Dropout Forward Pass:                                       │
│   Mask Generation: O(elements)                              │
│   Element-wise Multiply: O(elements)                        │
│   Overhead: Minimal compared to linear layers               │
└─────────────────────────────────────────────────────────────┘
```
"""
