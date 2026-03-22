# Introduction To Quantization Module

## Prerequisites
- This modules assumes you have gone through the following modules:
    - Tensor
    - Activations
    - Layers
    - Prolifing
    - Quantization

## Definition of Quantization
- Generally quantization is the process of mapping a large (often continuous) set of values into a saller set of discrete values.
- Meaning the we reduce precision to save space or computation.

### Memory Wall Problem
- Imagine your are trying to fit a library into your bag.
- Neural network face the same challenges whereby model become so huge to a point its impossible to store them in devices this is because devices have limited memory.
- This is where **Quantization** comes in.

### The Precision Paradox
- Modern neural networks use 32-bit floating point numbers (FP32) with incredible precision:

```
FP32 Number: 3.14159265359...
             ^^^^^^^^^^^^^^^^
             32 bits = 4 bytes per weight
```
- But here's the thing we do not neeed all that precision for most AI task.

### The Growing Memory Crisis

```
Model Memory Requirements (FP32):
┌─────────────────────────────────────────────────────────────┐
│ BERT-Base:   110M params ×  4 bytes = 440MB                 │
│ GPT-2:       1.5B params ×  4 bytes = 6GB                   │
│ GPT-3:       175B params × 4 bytes = 700GB                  │
│ Your Phone:  Available RAM = 4-8GB                          │
└─────────────────────────────────────────────────────────────┘
                        ↑
                    Problem!
```

### The Quantization Solution
- What is there was a strategy we could use to represent each weight with 8 bits instead of 32?

```
Before Quantization (FP32):
┌───────────────────────────────┐
│  3.14159265   │  2.71828183   │  32 bits each
└───────────────────────────────┘

After Quantization (INT8):
┌────────┬────────┬────────┬────────┐
│   98   │   85   │   72   │   45   │  8 bits each
└────────┴────────┴────────┴────────┘
         ↑
    4× less memory!
```

### Real-World Impact of achieved because of Quantization.

**Memory Reduction:**
- BERT-Base: 440MB → 110MB (4× smaller)
- Fits on mobile devices!
- Faster loading from disk
- More models in GPU memory

**Speed Improvements:**
- 2-4× faster inference (hardware dependent)
- Lower power consumption
- Better user experience

**Accuracy Preservation:**
- <1% accuracy loss with proper techniques
- Sometimes even improves generalization!

**Why This Matters:**
- **Mobile AI:** Deploy powerful models on phones
- **Edge Computing:** Run AI without cloud connectivity
- **Data Centers:** Serve more users with same hardware
- **Environmental:** Reduce energy consumption by 2-4×


## Fundementals (The Mathematics of Compression)

### Understanding the Core Challenge
- Think of quantization like converting a smooth analog signal to digital steps.
- Here in our scenario we need to map infinite precision (FP32) to just 256 possible values (INT8).

### The Quantization Mapping

```
The Fundamental Problem:

FP32 Numbers (Continuous):        INT8 Numbers (Discrete):
    ∞ possible values         →      256 possible values

  ...  -1.7  -1.2  -0.3  0.0  0.8  1.5  2.1  ...
         ↓     ↓     ↓    ↓    ↓    ↓    ↓
      -128  -95   -38    0   25   48   67   127
```

### The formula
- Every quantization system uses this fundemental relationship:

```
Quantization (FP32 → INT8):
┌─────────────────────────────────────────────────────────┐
│  quantized = round((float_value - zero_point) / scale)  │
└─────────────────────────────────────────────────────────┘

Dequantization (INT8 → FP32):
┌─────────────────────────────────────────────────────────┐
│  float_value = (quantized - zero_point) × scale         │
└─────────────────────────────────────────────────────────┘
```

### The Two Ciritical Parameters

**1. Scale(s)**- how big each INT8 step is in FP32 space:
```
Small Scale (high precision):       Large Scale (low precision):
 FP32: [0.0, 0.255]                 FP32: [0.0, 25.5]
   ↓     ↓     ↓                       ↓     ↓     ↓
 INT8:  0    128   255              INT8:  0    128   255
        │     │     │                      │     │     │
      0.0   0.127  0.255                 0.0   12.75  25.5

 Scale = 0.001 (very precise)        Scale = 0.1 (less precise)
```

**2. Zero Point (z)** - which INT8 value represents FP32 zero:

```
Symmetric Range:                    Asymmetric Range:
 FP32: [-2.0, 2.0]                  FP32: [-1.0, 3.0]
   ↓     ↓     ↓                       ↓     ↓     ↓
 INT8: -128    0   127              INT8: -128  -64   127
        │     │     │                      │     │     │
     -2.0    0.0   2.0                  -1.0   0.0   3.0

 Zero Point = 0                     Zero Point = -64
```

### Quantization Error Analysis
```
Perfect Reconstruction (Impossible):  Quantized Reconstruction (Reality):

Original: 0.73                       Original: 0.73
    ↓                                     ↓
INT8: ? (can't represent exactly)     INT8: 93 (closest)
    ↓                                     ↓
Restored: 0.73                        Restored: 0.728
                                           ↑
                                    Error: 0.002
```

**The Quantization Trade-off**
- **More bits** = higher precision,large memory
- **Fewer bits** = lower precision, smaller memory
- **Goal:** - find thw sweet spot where error is acceptable

### Why INT8 is the Sweet Spot

```
Precision vs Memory Trade-offs:

FP32: ████████████████████████████████ (32 bits) - Overkill precision
FP16: ████████████████ (16 bits)                  - Good precision
INT8: ████████ (8 bits)                           - Sufficient precision ← Sweet spot!
INT4: ████ (4 bits)                               - Often too little

Memory:    100%    50%    25%    12.5%
Accuracy:  100%   99.9%  99.5%   95%
```

- INT8 gives us 4x memory reduction with <1% accuracy loss, the perfect balance for production systems.

## Implementation of a Quantization Engine
- Details about the implementation of the Quantization Engine are available in **quantization.py**.

## INT8 Quantization - The Foundation
- This is the core function that converts any FP32 tensor to INT8.
- We can think of it as a smart compression algorithm that preserves the most important information.

```
Quantization Process Visualization:

Step 1: Analyze Range              Step 2: Calculate Parameters       Step 3: Apply Formula
┌─────────────────────────┐    ┌─────────────────────────┐  ┌─────────────────────────┐
│ Input: [-1.5, 0.2, 2.8] │    │ Min: -1.5               │  │ quantized = round(      │
│                         │    │ Max: 2.8                │  │   value / scale + zp)   │
│ Find min/max values     │ →  │ Range: 4.3              │ →│                         │
│                         │    │ Scale: 4.3/255 = 0.017  │  │                         │
│                         │    │ Zero Point: -39         │  │ Result: [-128,-27, 127] │
└─────────────────────────┘    └─────────────────────────┘  └─────────────────────────┘
```

**Key Challenges This Function Solves:**
- **Dynamic Range:** Each tensor has different min/max values
- **Precision Loss:** Map 4 billion FP32 values to just 256 INT8 values
- **Zero Preservation:** Ensure FP32 zero maps exactly to an INT8 value
- **Symmetric Mapping:** Distribute quantization levels efficiently

**Why This Algorithm:**
- **Linear mapping** preserves relative relationships between values
- **Symmetric quantization** works well for most neural network weights
- **Clipping to [-128, 127]** ensures valid INT8 range
- **Round-to-nearest** minimizes quantization error


## INT8 Dequantization (Restoring Precision)
- Dequantization is the inverse process, i.e converting compressed INT8 values back to usable FP32.
- This is where "decompress" our quantized data.

```
Dequantization Process:

INT8 Values + Parameters → FP32 Reconstruction

┌───────────────────────────────────┐
│ Quantized: [-128, -27, 127]       │
│ Scale: 0.017                      │
│ Zero Point: -39                   │
└───────────────────────────────────┘
                 │
                 ▼ Apply Formula
┌───────────────────────────────────┐
│ FP32 = (quantized - zero_point)   │
│        × scale                    │
└───────────────────────────────────┘
                 │
                 ▼
┌───────────────────────────────────┐
│ Result: [-1.501, 0.202, 2.799]    │
│ Original: [-1.5, 0.2, 2.8]        │
│ Error: [0.001, 0.002, 0.001]      │
└───────────────────────────────────┘
```

**Why This Step Is Critical:**
- **Neural networks expect FP32** - INT8 values would confuse computations
- **Preserves computation compatibility** - works with existing matrix operations
- **Controlled precision loss** - error is bounded and predictable
- **Hardware flexibility** - can use FP32 or specialized INT8 operations

**When Dequantization Happens:**
- **During forward pass** - before matrix multiplications
- **For gradient computation** - during backward pass
- **Educational approach** - production uses INT8 GEMM directly
"""

## QuantizedLinear 

### Why we Need Quantized Layers
- A quantized model isn't just storing weights in INT8, we need layers that can work  efficiently with quantized data.

```
Regular Linear Layer:              QuantizedLinear Layer:

┌─────────────────────┐            ┌─────────────────────┐
│ Input: FP32         │            │ Input: FP32         │
│ Weights: FP32       │            │ Weights: INT8       │
│ Computation: FP32   │    VS      │ Computation: Mixed  │
│ Output: FP32        │            │ Output: FP32        │
│ Memory: 4× more     │            │ Memory: 4× less     │
└─────────────────────┘            └─────────────────────┘
```

### The Quantized Forward Pass

```
Quantized Linear Layer Forward Pass:

    Input (FP32)                  Quantized Weights (INT8)
         │                               │
         ▼                               ▼
┌─────────────────┐              ┌─────────────────┐
│    Calibrate    │              │   Dequantize    │
│   (optional)    │              │   Weights       │
└─────────────────┘              └─────────────────┘
         │                               │
         ▼                               ▼
    Input (FP32)                  Weights (FP32)
         │                               │
         └───────────────┬───────────────┘
                         ▼
                ┌─────────────────┐
                │ Matrix Multiply │
                │   (FP32 GEMM)   │
                └─────────────────┘
                         │
                         ▼
                   Output (FP32)

Memory Saved: 4× for weights storage!
Speed: Depends on dequantization overhead vs INT8 GEMM support
```

### Calibration (Finding Optimal Input Quantization)

```
Calibration Process:

 Step 1: Collect Sample Inputs    Step 2: Analyze Distribution    Step 3: Optimize Parameters
 ┌─────────────────────────┐      ┌─────────────────────────┐    ┌─────────────────────────┐
 │ input_1: [-0.5, 0.2, ..]│      │   Min: -0.8             │    │ Scale: 0.00627          │
 │ input_2: [-0.3, 0.8, ..]│  →   │   Max: +0.8             │ →  │ Zero Point: 0           │
 │ input_3: [-0.1, 0.5, ..]│      │   Range: 1.6            │    │ Optimal for this data   │
 │ ...                     │      │   Distribution: Normal  │    │ range and distribution  │
 └─────────────────────────┘      └─────────────────────────┘    └─────────────────────────┘
```

**Why Calibration Matters:**
- **Without calibration:** Generic quantization parameters may waste precision
- **With calibration:** Parameters optimized for actual data distribution
- **Result:** Better accuracy preservation with same memory savings

## QuantizedLinear Class 
- This class replaces regular Linear layers with quantized version that use 4x less memory while preserving functionality.

```
QuantizedLinear Architecture:

Creation Time:                       Runtime:
┌───────────────────────────────┐    ┌───────────────────────────────┐
│ Regular Linear Layer          │    │ Input (FP32)                  │
│ ↓                             │    │ ↓                             │
│ Quantize weights → INT8       │    │ Optional: quantize input      │
│ Quantize bias → INT8          │ →  │ ↓                             │
│ Store quantization params     │    │ Dequantize weights            │
│ Ready for deployment!         │    │ ↓                             │
└───────────────────────────────┘    │ Matrix multiply (FP32)        │
      One-time cost                  │ ↓                             │
                                     │ Output (FP32)                 │
                                     └───────────────────────────────┘
                                        Per-inference cost
```

**Key Design Decisions:**

1. **Store original layer reference** - for debugging and comparison
2. **Separate quantization parameters** - weights and bias may need different scales
3. **Calibration support** - optimize input quantization using real data
4. **FP32 computation** - educational approach, production uses INT8 GEMM
5. **Memory tracking** - measure actual compression achieved

**Memory Layout:**
- Regular Linear layers store weights in FP32 (4 bytes each), while QuantizedLinear stores them in INT8 (1 byte each) plus a small overhead for quantization parameters (scales and zero points).
- This achieve approximately 4x memory reduction with minimal overhead.

**Production vs Educational Trade-off**
- **Our approach:** Dequantize -> FP32 computation
- **Production:** INT8 GEMM operations (faster,more complex)
- **Both achieve:** same memory saving, similar accuracy.

## Scaling to Full Neural Netwoes

### The Model Quantization Challenge
- Quantizing individual tensors is useful, but real applications need to quantize entire neural networks with multiple layers, activations and complex data flows.
- The key is replacing standard layers (like Linear) with quantized equivalent (QuantizedLinear) while keeping activation functions unchanged since they have no parameters.

### Smart Layer Selection 
- Not all layers benefit equally from quantization.
- Linear and convolutional layers with many parameters see the largest benefits, while activation functions (which have no parameters) cannot be quantized.
- Some layers like input/output projection may be sensitve to quantization and should be kept in higher precision for critical application. 

### Calibration Data Flow
- Calibration runs sample data through the model layer-by-layer,  collecting activation statistics (min/max values, distributions) determine optimal quantization parameters for each layer, ensuring minimal accuracy loss during quantization.

### Memory Impact
- Quantization provides  consistent 4x memory reduction across all model sizes.
- The actual impact depends on model architecture, but the compression ratio remains constant since we're reducing precision from 32 bits to 8 bits per parameter.

##  Advanced Quantization Strategies - Production Techniques

This analysis compares different quantization approaches used in production systems, revealing the trade-offs between accuracy, complexity, and performance.

```
Strategy Comparison Framework:

┌──────────────────────────────────────────────────────────────────────────────────┐
│                          Three Advanced Strategies                             │
├──────────────────────────┬──────────────────────────┬──────────────────────────┤
│       Strategy 1         │       Strategy 2         │       Strategy 3         │
│    Per-Tensor (Ours)     │    Per-Channel Scale     │    Mixed Precision       │
├──────────────────────────┼──────────────────────────┼──────────────────────────┤
│                          │                          │                          │
│ ┌──────────────────────┐ │ ┌──────────────────────┐ │ ┌──────────────────────┐ │
│ │ Weights:             │ │ │ Channel 1: scale₁   │ │ │ Sensitive: FP32      │ │
│ │ [W₁₁ W₁₂ W₁₃]        │ │ │ Channel 2: scale₂   │ │ │ Regular: INT8        │ │
│ │ [W₂₁ W₂₂ W₂₃] scale  │ │ │ Channel 3: scale₃   │ │ │                      │ │
│ │ [W₃₁ W₃₂ W₃₃]        │ │ │                      │ │ │ Input: FP32          │ │
│ └──────────────────────┘ │ │ Better precision     │ │ │ Output: FP32         │ │
│                          │ │ per channel          │ │ │ Hidden: INT8         │ │
│ Simple, fast             │ └──────────────────────┘ │ └──────────────────────┘ │
│ Good baseline            │                          │                          │
│                          │ More complex             │ Optimal accuracy         │
│                          │ Better accuracy          │ Selective compression    │
└──────────────────────────┴──────────────────────────┴──────────────────────────┘
```

**Strategy 1: Per-Tensor Quantization (Our Implementation)**
```
Weight Matrix:                Scale Calculation:
┌─────────────────────────┐     ┌─────────────────────────┐
│ 0.1 -0.3  0.8  0.2      │     │ Global min: -0.5        │
│-0.2  0.5 -0.1  0.7      │ →   │ Global max: +0.8        │
│ 0.4 -0.5  0.3 -0.4      │     │ Scale: 1.3/255 = 0.0051 │
└─────────────────────────┘     └─────────────────────────┘

Pros: Simple, fast           Cons: May waste precision
```

**Strategy 2: Per-Channel Quantization (Advanced)**
```
Weight Matrix:                Scale Calculation:
┌─────────────────────────┐     ┌─────────────────────────┐
│ 0.1 -0.3  0.8  0.2      │     │ Col 1: [-0.2,0.4] → s₁  │
│-0.2  0.5 -0.1  0.7      │ →   │ Col 2: [-0.5,0.5] → s₂  │
│ 0.4 -0.5  0.3 -0.4      │     │ Col 3: [-0.1,0.8] → s₃  │
└─────────────────────────┘     │ Col 4: [-0.4,0.7] → s₄  │
                             └─────────────────────────┘

Pros: Better precision       Cons: More complex
```

**Strategy 3: Mixed Precision (Production)**
```
Model Architecture:            Precision Assignment:
┌─────────────────────────┐     ┌─────────────────────────┐
│ Input Layer  (sensitive) │     │ Keep in FP32 (precision) │
│ Hidden 1     (bulk)     │ →   │ Quantize to INT8        │
│ Hidden 2     (bulk)     │     │ Quantize to INT8        │
│ Output Layer (sensitive)│     │ Keep in FP32 (quality)   │
└─────────────────────────┘     └─────────────────────────┘

Pros: Optimal trade-off      Cons: Requires expertise
```

**Experimental Design:**
```
Comparative Testing Protocol:

1. Create identical test model   →  2. Apply each strategy        →  3. Measure results
   ┌───────────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
   │ 128 → 64 → 10 MLP      │     │ Per-tensor quantization │     │ MSE error calculation  │
   │ Identical weights       │     │ Per-channel simulation  │     │ Compression measurement│
   │ Same test input         │     │ Mixed precision setup   │     │ Speed comparison       │
   └───────────────────────┘     └───────────────────────┘     └───────────────────────┘
```

**Expected Strategy Rankings:**
1. **Mixed Precision** - Best accuracy, moderate complexity
2. **Per-Channel** - Good accuracy, higher complexity
3. **Per-Tensor** - Baseline accuracy, simplest implementation

This analysis reveals which strategies work best for different deployment scenarios and accuracy requirements.
 
 
