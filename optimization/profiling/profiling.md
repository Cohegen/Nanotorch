# Introduction to Profiling Submodule
- In this submodule, you'll learn about profiling in ML systems.

## Definition
- Profiling is the **systematic measurement of resource usage and performance** in our ML pipeline.
- It's not just timing how long a model trains, it' about understanding every step:
     - How long each layer or operation takes (forward/backward passes)
     - How much memory each tensor occupies
     - Whether GPUs or CPUs are underutilized
     - Where I/O or data preprocessing is slowing down things

- Think of it as a dectective for your ML workflow.
- Without profiling, we might spend hours training a model, but 50% of the time could be wasted on slow data loading or unnecessary computations.

## Why does Profiling Matter in ML systems?
- Imagine that we are detectives investigating a performance crime.
- Our model is running slowly,using too much memory, or burning thrugh compute budgets.
- Without profiling, we would be blind, that is we would be making guesses about what to optimize.
- With profiling, we have no evidence.


**The Performance Investigation Process:**
```
Suspect Model → Profile Evidence → Identify Bottleneck → Target Optimization
     ↓               ↓                    ↓                    ↓
   "Too slow"    "200 GFLOP/s"      "Memory bound"      "Reduce transfers"
```

**Questions Profiling Answers:**
- **How many parameters?** (Memory footprint, model size)
- **How many FLOPs?** (Computational cost, energy usage)
- **Where are bottlenecks?** (Memory vs compute bound)
- **What's actual latency?** (Real-world performance)

**Production Importance:**
In production ML systems, profiling isn't optional, it's survival. A model that's 10% more accurate but 100× slower often can't be deployed. Teams use profiling daily to make data-driven optimization decisions, not guesses.

### The Profiling Workflow Visualization
```
Model → Profiler → Measurements → Analysis → Optimization Decision
  ↓        ↓           ↓           ↓            ↓
 GPT   Parameter   125M params   Memory      Apply targeted
       Counter     2.5B FLOPs    bound       optimization
```

### From Implementation to Optimization: The Profiling Foundation
- In this module, we will build the measurement tools to discover optimization opportunities.
- Profiling insights guide targeted performance improvements, we can't optimize what we can't optimize.

**The Real ML Engineering Workflow**:

```
┌────────────────────────────────────────────────────────────────┐
│ Step 1: Measure (This Module!)    Step 2: Analyze              │
│   ↓                                 ↓                          │
│ Profile baseline → Find bottleneck → Understand cause          │
│ 40 tok/s          80% in attention    O(n^2) recomputation     │
│                                       ↓                        │
│ Step 4: Validate                    Step 3: Optimize (Future)  │
│   ↓                                   ↓                        │
│ Profile optimized ← Verify speedup ← Implement optimization    │
│ 500 tok/s (12.5x)   Measure impact    Design solution          │
└────────────────────────────────────────────────────────────────┘
```

**Without profiling**: We'd never know WHERE to optimize!
**Without measurement**: We couldn't verify improvements!


## Foundations : Performance Measurement Principles
- Before one builds a profiler, we need to understand what we're measuring and why each metric matters.

### Parameter Counting: Model Size
- Parameters determine our model's memory footprint and storage requirements.
- Every parameter is typically a 32-bit float (4 bytes),so counting them precisely predicts memory usage.

**Parameter Counting Formula:**
```
Linear Layer: (input_features × output_features) + output_features
               ↑              ↑                    ↑
            Weight matrix   Bias vector      Total parameters

Example: Linear(768, 3072) → (768 × 3072) + 3072 = 2,362,368 parameters
Memory: 2,362,368 × 4 bytes = 9.45 MB
```

### FLOP (Floating Point Operation) counting: (Computational Cost Analysis)
- FLOPS measure computational work.
- Unlike wall-clock time, FLOPs are hardware-independent and  predict compute costs across different systems.

**FLOP Formulas for Key Operations:**

```
Matrix Multiplication (M,K) @ (K,N):
   FLOPs = M x N x K x 2
           ↑   ↑   ↑   ↑
        Rows Cols Inner Multiply+Add

Linear Layer Forward:
   FLOPs = batch_size x input_features x output_features x 2
                      ↑                  ↑                 ↑
                  Matmul cost        Bias add        Operations

Linear Layer FLOP Breakdown:
┌────────────────────────────────────────────────────────────────┐
│ Input (batch=32, features=768) × Weight (768, 3072) + Bias     │
│                         ↓                                       │
│ Matrix Multiplication: 32 × 768 × 3072 × 2 = 150,994,944 FLOPs │
│ Bias Addition:         32 × 3072 × 1      =      98,304 FLOPs  │
│                         ↓                                       │
│ Total FLOPs:                                 151,093,248 FLOPs │
└────────────────────────────────────────────────────────────────┘

Convolution (simplified):
   FLOPs = output_H x output_W x kernel_H x kernel_W x in_channels x out_channels x 2

Convolution FLOP Breakdown:
┌────────────────────────────────────────────────────────────────┐
│ Input (batch=1, channels=3, H=224, W=224)                      │
│ Kernel (out=64, in=3, kH=7, kW=7)                             │
│                         ↓                                       │
│ Output size: (224×224) → (112×112) with stride=2              │
│ FLOPs = 112 × 112 × 7 × 7 × 3 × 64 × 2 = 236,027,904 FLOPs    │
└────────────────────────────────────────────────────────────────┘
```
#### FLOP Counting Strategy 
Different operations require different FLOP calculations:
- **Matrix operations**: M x N x K x 2 (multiply + add)
- **Convolutions**: Output spatial x kernel spatial x channels
- **Activations**: Usually 1 FLOP per element

### Memory Profiling
- ML models use memory in three distinct ways, each with different optimization strategies:

**Memory Type Breakdown:**
```
Total Training Memory = Parameters + Activations + Gradients + Optimizer State
                           ↓            ↓           ↓            ↓
                        Model         Forward     Backward     Adam: 2×params
                        weights       pass cache  gradients    SGD: 0×params

Example for 125M parameter model:
Parameters:    500 MB (125M × 4 bytes)
Activations:   200 MB (depends on batch size)
Gradients:     500 MB (same as parameters)
Adam state:  1,000 MB (momentum + velocity)
Total:      2,200 MB (4.4× parameter memory!)
```

### Latency Measurement
- Latency measurement is tricky because systems have variance, warmup effects and measurement overhead.
- **Latency** is the time it takes for a single input(or batch) to go through the ML system, from input arrival to output prediction.
- Key difference from throughput
   - Latency : time per sample (ms)
   - Throughput: number of samples processed per second.
- Professional profiling requires statistical rigor.

```
Measurement Protocol:
┌────────────────────────────────────────────────────────────────┐
│ 1. Warmup runs (10+)  → CPU/GPU caches warm up                │
│ 2. Timed runs (100+)  → Statistical significance              │
│ 3. Outlier handling   → Use median, not mean                  │
│ 4. Memory cleanup     → Prevent contamination                 │
└────────────────────────────────────────────────────────────────┘

Timeline:
Warmup: [run][run][run]...[run]     <- Don't time these
Timing: [run][run]...[run]          <- Time these
Result: median(all_times)           <- Robust to outliers
```
#### Latency Measurement Challenges
```
Timing Challenges:
┌─────────────────────────────────────────────────┐
│                 Time Variance                   │
├─────────────────┬─────────────────┬─────────────┤
│  System Noise   │   Cache Effects │   Thermal   │
│                 │                 │  Throttling │
├─────────────────┼─────────────────┼─────────────┤
│ Background      │ Cold start vs   │ CPU slows   │
│ processes       │ warm caches     │ when hot    │
│ OS scheduling   │ Memory locality │ GPU thermal │
│ Network I/O     │ Branch predict  │ limits      │
└─────────────────┴─────────────────┴─────────────┘

Solution: Statistical Approach
Warmup → Multiple measurements → Robust statistics (median)
```

## Profile Architecture

```
Profiler Class Structure:
┌─────────────────────────────────────────────────────────────┐
│ Core Measurement Methods:                                   │
│ • count_parameters() → Model size analysis                 │
│ • count_flops() → Computational cost estimation            │
│ • measure_memory() → Memory usage tracking                 │
│ • measure_latency() → Performance timing                   │
├─────────────────────────────────────────────────────────────┤
│ Advanced Profiling Methods:                                 │
│ • profile_layer() → Layer-wise analysis                    │
│ • profile_forward_pass() → Complete forward analysis       │
│ • profile_backward_pass() → Training analysis              │
├─────────────────────────────────────────────────────────────┤
│ Integration:                                                 │
│ All methods work together for comprehensive insights        │
└─────────────────────────────────────────────────────────────┘
```