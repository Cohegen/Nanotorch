# Introduction to the Quantization Module

This document walks through quantization **in order**: from *why* it exists, to the *math*, to *tensor* → *layer* → *full model* behavior, and finally to *production-style* variants. Read top to bottom once for a full picture, or jump using the roadmap below.

## How to read this guide (step-by-step path)

1. **Motivation** — memory pressure and what quantization buys you.
2. **Vocabulary** — scale, zero point, INT8 range, quantization error.
3. **One tensor, end-to-end** — compute scale and zero point, quantize, dequantize (matches `quantize_int8` / `dequantize_int8` in `quantization.py`).
4. **QuantizedLinear** — what happens when you wrap a `Linear` layer.
5. **Calibration** — how sample inputs set **input** quantization parameters (stored for analysis; the educational forward pass still matmuls in FP32).
6. **Whole models** — replacing linears while keeping activations as-is.
7. **Beyond the baseline** — per-channel and mixed precision (concepts).

---

## Prerequisites

This module assumes you have already worked through:

- **Tensor** — shapes, dtypes, element-wise ops.
- **Activations** — nonlinearities (no weights to quantize).
- **Layers** — especially `Linear` and how `forward` uses weights and bias.
- **Profiling** — measuring memory and time (helps interpret compression gains).

---

## Step 1 — What quantization is

Quantization maps values from a **large** set (often many possible FP32 numbers) to a **small discrete** set (for example 256 INT8 levels). You **reduce precision** to save **memory** and often **compute**.

![Alt text](https://github.com/Cohegen/Nanotorch/blob/main/assets/quantization_conversion_graph.webp)

Neural networks can outgrow device RAM (the “memory wall”). If each weight uses 32 bits, a large model is measured in gigabytes; using 8 bits per weight cuts storage for weights by about **4×** (plus small overhead for scale/zero-point metadata).
![Alt text](https://github.com/Cohegen/Nanotorch/blob/main/assets/quantization_net.webp)

**Precision vs need:** FP32 has far more precision than most inference tasks require. INT8 is a common “sweet spot”: large memory savings with modest accuracy impact when scales are chosen well.

---

## Step 2 — The growing memory picture (illustrative)

```
Model Memory Requirements (FP32 weights, rough order of magnitude):
┌─────────────────────────────────────────────────────────────┐
│ BERT-Base:   110M params × 4 bytes ≈ hundreds of MB       │
│ Large LMs:   billions of params × 4 bytes → many GB       │
│ Your phone:  on the order of a few GB RAM total             │
└─────────────────────────────────────────────────────────────┘
```

Quantization is one lever to shrink **stored weights** (and, in full stacks, activations) so models fit and load faster.

---

## Step 3 — The core idea in one diagram

```
Before (FP32):     each weight ≈ 32 bits
After (INT8):      each stored weight ≈ 8 bits  →  ~4× smaller weight storage
```

**Typical benefits (when done well):**

- **Memory:** fewer bytes for weights; more models or batch size in RAM.
- **Speed:** hardware-dependent; INT8 paths can be faster when available.
- **Accuracy:** often within a small margin of FP32 with good calibration and schemes.

---

## Step 4 — Mapping FP32 → INT8 (the vocabulary)

![Alt text](https://github.com/Cohegen/Nanotorch/blob/main/assets/fp32_to_int8.webp)

Think of quantization like **analog → digital**: infinitely many FP32 values must land on **256** signed INT8 codes (`-128` … `127`).

```
FP32 (continuous)              INT8 (256 levels)
  ... -1.2  0.0  0.8 ...   →   ... -95   0   25 ...
```

Two numbers **fully describe** the affine mapping used in this project:

| Term | Role |
|------|------|
| **Scale** `s` | How much FP32 value each step between INT8 levels represents. |
| **Zero point** `z` | Which INT8 code corresponds to FP32 `0` (after rounding). |

**Dequantization** (INT8 → approximate FP32) is always:

```
x_approx = (q - z) × s
```

where `q` is the stored INT8 (used as a real number for the formula).

**Quantization** (FP32 → INT8) is the nearest-integer assignment consistent with that relationship. In **this codebase** the forward quantization is implemented as:

```
q = round( x / s + z )
```

then `q` is **clipped** to `[-128, 127]` and stored as `int8`. (Algebraically this pairs with `x ≈ (q - z) × s`.)

Some textbooks write an equivalent form with `(x - offset) / s`; the **pairing** of encode/decode matters. When reading `quantization.py`, use the formulas above.

---

## Step 5 — How `scale` and `zero_point` are computed (per tensor)

For a **single FP32 tensor**, the engine looks at **min** and **max** over all elements:

1. **Range:** `r = max_val - min_val`.
2. **Edge case:** If the tensor is (almost) constant, the code uses `s = 1`, `z = 0`, and all quantized values `0`.
3. **Otherwise:**
   - `s = r / 255` — the FP32 span of one step between adjacent INT8 levels (255 steps from min code to max code in the unsigned span of indices; here the codes span `-128…127`).
   - `z = round(-128 - min_val / s)`, then **clamp** `z` into `[-128, 127]`.
4. **Apply:** `q = round(x / s + z)`, then clip to INT8 range.

Constants in code: `INT8_MIN_VALUE = -128`, `INT8_MAX_VALUE = 127`, `INT8_RANGE = 256`.

**Intuition:**

- **`s`** stretches or shrinks FP32 so the observed `[min, max]` maps into the available INT8 codes.
- **`z`** shifts the mapping so that **zero in FP32** lands near a valid INT8 index (as closely as rounding allows).

---

## Step 6 — Walkthrough: one small tensor

Take FP32 values `[-1.5, 0.2, 2.8]` (same spirit as the diagrams earlier).

**6.1 — Observe range**

- `min_val = -1.5`, `max_val = 2.8`, `r = 4.3`.

**6.2 — Scale**

- `s = 4.3 / 255 ≈ 0.01686`.

**6.3 — Zero point**

- `z = round(-128 - (-1.5) / s) = round(-128 + 88.88…) ≈ -39` (then clamp if needed; `-39` is in range).

**6.4 — Quantize each value**

- `q_i = round(x_i / s + z)`; clip to `[-128, 127]`.

You should see codes clustered across the range; dequantizing with `(q - z) * s` gives back numbers **close to** the originals, not always exact.

---

## Step 7 — Quantization error (what the learner should expect)

```
Original FP32:     0.73
       ↓ round-trip through INT8 with fixed (s, z)
Restored FP32:     e.g. 0.728
       ↓
Small error:       |0.73 - 0.728| is quantization noise
```

**Trade-off:**

- More bits → higher fidelity, more memory.
- Fewer bits → more noise, less memory.

The goal is an acceptable error on **downstream metrics** (accuracy, perplexity), not zero per-weight error.

---

## Step 8 — Dequantization only (restore approximate FP32)

Given stored `q`, `s`, and `z`:

```
x_approx = (q - z) × s
```

**Why it matters here:** The educational `QuantizedLinear` **dequantizes weights (and bias) to FP32** before `matmul`, so the rest of the stack still looks like a normal floating-point layer from the outside.

**Production contrast:** Many deployed stacks fuse **INT8 matrix multiply** with scales to avoid explicit FP32 weights; memory savings are similar, but kernels and numerics are more involved.

---

## Step 9 — From tensors to layers: `QuantizedLinear`

### 9.1 — Why a special layer?

Storing weights as INT8 is not enough—you need code that **loads INT8**, **reconstructs or uses** them in multiply-accumulate, and keeps the **same interface** as `Linear` for the rest of the model.

### 9.2 — What happens at construction (one-time)

When you build `QuantizedLinear` from a `Linear` layer (`quantization.py`):

1. Run `quantize_int8` on **weights** → `q_weight`, `weight_scale`, `weight_zero_point`.
2. If bias exists, run `quantize_int8` on **bias** → `q_bias`, `bias_scale`, `bias_zero_point`.
3. Initialize `input_scale` / `input_zero_point` to `None` until calibration.

### 9.3 — What happens on `forward` (each call)

1. `weight_fp32 = dequantize_int8(q_weight, weight_scale, weight_zero_point)`.
2. `result = x.matmul(weight_fp32)` with **FP32 activations** `x`.
3. If bias exists: dequantize bias and add.

So: **weights compressed in memory**, **compute path uses FP32** in this tutorial implementation.

```
Input x (FP32)     q_weight (INT8) + (s_w, z_w)
      │                      │
      │                      ▼
      │            dequantize → weight_fp32
      │                      │
      └────────── matmul ────┘
                  │
                  ▼
            output (FP32)  [+ dequantized bias if any]
```

---

## Step 10 — Calibration (input statistics)

**Purpose:** Run **representative inputs** through the layer (or model) and record how activations behave so you can choose good **input** scales and zero points—**if** you quantize activations or analyze sensitivity.

In this module, `calibrate(sample_inputs)`:

1. Flattens all sample tensors and finds **global min/max** over those values.
2. Computes `input_scale` and `input_zero_point` with the **same recipe** as weight quantization (constant-tensor guard included).

**Important for readers of the current code:** `forward` does **not** quantize `x` using `input_scale` / `input_zero_point`; it only uses **dequantized weights**. Calibration still teaches the **pipeline** used in fuller systems and matches how input parameters *would* be set.

```
Collect many input tensors  →  min/max over all seen values  →  (input_scale, input_zero_point)
```

---

## Step 11 — Scaling to a full network

### 11.1 — The challenge

Real models stack many layers. You want **consistent replacement**: e.g. each `Linear` → `QuantizedLinear`, while **ReLU** (no parameters) stays unchanged.

### 11.2 — Layer selection

- **Big wins:** `Linear` / conv layers with large weight tensors.
- **No weight tensors:** activations, dropout, etc.—nothing to quantize as “weights.”
- **Sensitivity:** first/last layers are sometimes kept in FP32 in production **mixed-precision** policies (concept below).

### 11.3 — Calibration data flow (model-wide idea)

For each quantized layer, you may pass calibration data through **all preceding layers** so the inputs to that layer reflect real activation ranges—then set per-layer input quantization parameters. That mirrors how deployment tools build histograms or min/max stats per tensor.

### 11.4 — Memory impact

Weight storage drops by about **4×** for INT8 vs FP32, plus small overhead for scales/zero points. Total model size also includes non-quantized pieces (e.g. batch norm buffers if any, or FP32 layers you chose not to quantize).

---

## Step 12 — Advanced strategies (production landscape)

Three common patterns (your implementation is **per-tensor** for each weight/bias tensor):

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Strategy 1: Per-tensor (this repo)                                         │
│   One (s, z) for the entire weight matrix — simple, fast baseline.         │
├────────────────────────────────────────────────────────────────────────────┤
│ Strategy 2: Per-channel                                                  │
│   Separate scales (often per output channel) — better fit, more metadata.  │
├────────────────────────────────────────────────────────────────────────────┤
│ Strategy 3: Mixed precision                                               │
│   Keep sensitive layers in FP32, quantize the bulk — strong accuracy/cost. │
└────────────────────────────────────────────────────────────────────────────┘
```

**Per-tensor (ours):** one global min/max over the whole tensor → one scale/zero point.

**Per-channel:** min/max along each channel (e.g. each column of a weight matrix) → many scales; often better SNR for conv/linear weights.

**Mixed precision:** quantize only part of the graph; typical for attention projections or heads in transformers.

**Rough expectation:** mixed precision and per-channel often beat pure per-tensor on accuracy, at the cost of complexity and extra storage for scales.

---

## Step 13 — Where to look in code

| Step in this guide | Primary code |
|--------------------|--------------|
| Tensor quantize/dequantize | `quantize_int8`, `dequantize_int8` |
| Layer | `QuantizedLinear` |
| Full model | `quantize_model` and helpers in `quantization.py` |

Reading this file **together** with `quantization.py` ties the narrative steps to function names and tensor dtypes (`int8` weights after quantization).

---

## Quick reference card

```
INT8 codes:  -128 … 127

Dequantize:  x ≈ (q - z) × s

Quantize (as implemented):  q = clip( round(x / s + z), -128, 127 )

Per tensor:  s from (max - min) / 255, z from -128 - min/s (clamped)
```

This closes the loop: you now have an ordered path from **problem** → **math** → **tensor** → **layer** → **model** → **stronger industrial variants**.
