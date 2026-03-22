# Introduction to the Profiling Submodule

This guide explains **why** we profile ML code, **what** we measure, and **how** those measurements are produced and interpreted. It is aligned with the `Profiler` class and helpers in `profiling.py` so narrative and implementation stay in sync.

---

## How to use this document (learning path)

1. **Motivation** — why guessing fails; inference vs training cost; what “evidence” means.
2. **Three lenses** — structural facts, analytical work (FLOPs), and empirical observation (time, memory).
3. **Metrics (deep dive)** — for each metric: *why it exists*, *how we obtain it here*, *what it cannot tell you*.
4. **FLOPs vs latency** — why more FLOPs do not always mean slower runs (memory vs compute).
5. **Measurement procedure** — warmup, repeated trials, median; what `perf_counter` and `tracemalloc` actually do.
6. **Using this codebase** — API order for layer vs model vs training estimates.
7. **Reading results** — bottlenecks, efficiency placeholders, sanity checks.
8. **Misconceptions** — common traps learners hit.
9. **Closing the loop** — measure → change → measure again.

---

## Part 1 — Motivation: why profile?

**Profiling** is the *systematic* measurement of **resource use** (time, memory) and **work** (parameters, FLOPs) in a model or pipeline. It is not the same as “it feels slow” or “training loss improved.”

### 1.1 The core “why”

Engineering questions are usually **quantitative**:

- Will this model **fit in RAM** on the target device?
- Will inference meet a **latency SLA** (e.g. under 50 ms per request)?
- If training **OOMs**, is it activations, optimizer state, or batch size?
- Did my refactor **actually** speed things up, or was the first run a fluke?

**Profiling exists because the computer’s real behavior is a combination of math, memory traffic, libraries, and the OS.** Big-O intuition (“attention is quadratic”) is necessary but not sufficient: constants, layout, and bottlenecks dominate in practice.

### 1.2 Inference vs training (different “whys”)

| Phase | You often care about | Profiling helps by… |
|--------|----------------------|---------------------|
| **Inference** | Latency, peak RAM, model size on disk | Separating parameter memory from activation memory; timing stable forward passes |
| **Training** | Throughput, peak RAM, step time | Reasoning about gradients + optimizer memory (this repo **estimates** backward/optimizer costs) |

The same forward pass may be “fast enough” for inference but still **too memory-heavy** once you add gradients and Adam state—that is a *different* question profiling answers by expanding what you count.

### 1.3 The problem with optimizing blind

- You might spend hours on a micro-optimization while most time sits in **one layer**, the **wrong batch size**, or **data loading** (this submodule focuses on **model** forward/backward; I/O profiling is separate).
- Without measurements, you cannot tell **compute-bound** (arithmetic throughput limits you) from **memory-bound** (moving tensors limits you). The **same** architecture can flip between the two when you change batch size or hardware.

### 1.4 What profiling enables

| Without profiling | With profiling |
|-------------------|----------------|
| Guess where time goes | Relate parameters, FLOPs, latency, and memory in one pass |
| Debate complexity abstractly | Compare **concrete** models on the **same** input shape |
| Hope a change helped | Re-run the **same protocol** and compare **medians** |

**Production angle:** A model that is slightly more accurate but violates latency or memory budgets often **cannot ship**. Profiling turns trade-offs into **evidence-based** decisions.

### 1.5 One-line mental model

```
Suspect (slow / big / OOM) → Measure → Evidence → Bottleneck hypothesis → One targeted change → Validate (measure again)
```

---

## Part 2 — Three lenses: how profiling thinks

It helps to separate **three kinds** of information:

1. **Structural (what is built)** — parameter count, shapes. Derived from the **graph and tensors**, not from a stopwatch. **Why:** capacity, disk size, and a floor for “how much stuff exists.”
2. **Analytical work (what the math implies)** — FLOP estimates. **Why:** compare algorithms **before** or **independently of** a specific machine’s quirks.
3. **Empirical (what actually happened)** — wall-clock latency; allocator/tracemalloc peaks. **Why:** users and SLAs live in **seconds and megabytes**, not FLOPs alone.

Good profiling **triangulates**: if FLOPs are low but latency is high, suspect overhead, Python dispatch, or memory bandwidth—not “more math.”

---

## Part 3 — What we measure (concepts + how + limits)

### 3.1 Parameters (model size)

**What:** Count of learnable scalars (weights and biases).

**Why:**

- **Storage:** FP32 ≈ 4 bytes per parameter → rough **disk / RAM** for weights.
- **Training floor:** Gradients are typically the **same shape** as parameters; optimizers add **extra** state (e.g. Adam).

**How (this repo):** `count_parameters()` sums `Tensor` elements from `layer.parameters()` on `Sequential`, or reads `weight` / `bias` on a single layer.

**Limits:** Shared weights, embeddings tied across layers, or frozen params need **semantic** interpretation—the code counts **stored** tensors, not “trainable in the business sense.”

**Typical `Linear` count:**

```
Linear(in_features, out_features):
  weights: in_features × out_features
  bias:    out_features
  total:   in_features × out_features + out_features
```

---

### 3.2 FLOPs (computational work)

**What:** A scalar estimate of floating-point **work** in a forward pass (multiply–add conventions vary slightly across literature; here, one mac ≈ **2 FLOPs**).

**Why:**

- **Hardware-agnostic comparison** of layer types and depths.
- **Scaling intuition:** doubling hidden size often **quadruples** some matmul costs—counting makes that explicit.
- **Energy / cost models** in large systems are often FLOP- or mac-based.

**How (this repo):**

- `Linear`: `_count_linear_flops` → `in_features × out_features × 2` (uses `input_shape[-1]` and `weight.shape[1]`). **Batch is not multiplied** in this helper—treat it as **one logical matmul step** per forward call pattern; for per-batch totals, multiply by batch in your head or extend the counter.
- `Conv2d`: simplified `out_H × out_W × kernel² × in_C × out_C × 2` with `out = input // stride`.
- `Sequential`: sum per layer, **propagate** `current_shape` after each layer with weights.

**Limits:** Real kernels fuse ops, skip zeros, or use lower precision; **analytic FLOPs ≠ billed GPU instructions.** Unknown layer types fall back to `prod(input_shape)`—a **placeholder**, not physics.

**Matrix view (reference):** `(M×K) @ (K×N)` is often counted as `2 M N K` FLOPs.

---

### 3.3 Memory

**What:** Bytes held or touched so the program can run.

**Why:**

- **OOM** is the hard stop in training; **peak** matters more than “average.”
- **Deployment:** mobile / edge devices have a hard RAM cap.

**Roles (conceptual, especially training):**

```
parameters + activations + gradients + optimizer state (+ framework overhead)
```

**How (this repo):** `measure_memory` starts **`tracemalloc`**, records **parameter** MB from the parameter count, uses a **rough** activation estimate from the **dummy input** buffer (`nbytes * 2`), runs `forward`, reads **peak** traced allocation, and returns `memory_efficiency` as useful-vs-peak ratio.

**Why tracemalloc:** It attributes **CPython object allocations** over an interval—**pedagogical** and **reproducible** in pure Python. It is **not** a GPU allocator; it does not see CUDA memory pools.

**Limits:** Peaks miss some native buffers; **fragmentation** and **garbage collection** can distort repeats. Compare runs with the **same** Python version and input shape.

---

### 3.4 Latency (wall-clock time)

**What:** Elapsed real time for a forward (here: `model.forward`), in **milliseconds**.

**Why:** Users and servers experience **latency** and **throughput** (samples/sec), not FLOPs.

**How (this repo):** `time.perf_counter()` around `forward` in a loop; **warmup** runs untimed; report **median** over `iterations`.

**Limits:** OS scheduling, other processes, **CPU frequency**, caches, and **Python overhead** all contribute. That is **why** we warm up and repeat.

**Latency vs throughput:**

- **Latency:** time for **one** forward (often quoted per batch).
- **Throughput:** batches or samples **per second**; can improve with pipelining or larger batch even when per-batch latency rises.

---

### 3.5 How the metrics relate (why you need more than one)

```
Parameters  →  “How big is the static model?”
FLOPs       →  “How much arithmetic does forward imply?”
Latency     →  “How long did forward actually take on this machine?”
Memory      →  “How much did the runtime need to allocate / hold?”
```

**High FLOPs + low latency** → machine is well-fed with work; good **arithmetic intensity** or efficient kernels (in big frameworks).

**Low FLOPs + high latency** → overhead, Python bound, or memory stalls—**profile empirically** and look outside raw FLOPs.

---

## Part 4 — Why FLOPs and latency diverge (memory vs compute)

**FLOPs count math operations in a model definition. Latency measures time on a real stack.**

Even in idealized hardware, **speed of light** is: you must **read** inputs and weights and **write** outputs. If memory bandwidth is saturated, adding more cores does not help—**you are memory-bound**.

**Compute-bound** roughly means: if the hardware could supply data instantly, you would still be busy doing arithmetic.

This repo’s `_analyze_bottleneck` uses a **simple heuristic**: compare derived **GFLOP/s** to **MB/s** memory bandwidth. It is a **teaching label**, not a hardware characterization certificate. Use it to **ask the right next question**, not as ground truth.

---

## Part 5 — Procedure: how to measure fairly

Follow this **protocol** when comparing two models or two commits.

### 5.1 Step A — Fix the scenario

1. **Same `input_shape` / tensor** for both sides.
2. **Same machine load** where possible (close heavy apps).
3. **Same code path** (e.g. both `forward` only, no accidental prints).

**Why:** Latency noise often swamps small improvements if the scenario drifts.

### 5.2 Step B — Warmup

Run **several** forwards **without** recording time.

**Why:**

- **CPU caches** (instruction + data) stabilize after first touches.
- First allocations can be slower; **tracemalloc** and allocator behavior **settle**.

Defaults: `measure_latency(..., warmup=10, iterations=100)`. `profile_forward_pass` uses **fewer** repeats for speed—tighten manually for serious benchmarks.

### 5.3 Step C — Many timed iterations

Loop `iterations` times with `perf_counter` around `forward`.

**Why:** Single samples catch **GC pauses**, scheduler blips, or background spikes.

### 5.4 Step D — Use the median

This profiler uses **median** latency.

**Why median, not mean?**

- **Outliers** (one slow run) skew the **mean** heavily.
- The **median** answers: “what is **typical**?” for a noisy process.

Reporting **min / max / p90** alongside is good practice for serious work.

### 5.5 What `time.perf_counter()` is (and is not)

- It is a **monotonic** wall clock suitable for **intervals** on the same machine.
- It is **not** CPU time only (no automatic “subtract other threads”).
- It is **not** GPU kernel time (CUDA kernels can overlap asynchronously).

**Why we still use it here:** For a **CPU-only educational forward**, it gives a **reproducible** latency story without vendor tools.

### 5.6 Checklist

```
[ ] Identical input shape and batch
[ ] Warmup before timing
[ ] Enough iterations (tens to hundreds)
[ ] Summarize with median (optionally min/max)
[ ] Same environment when comparing A vs B
```

---

## Part 6 — Workflow in this repository (`profiling.py`)

### 6.1 Create a profiler

Use the same `sys.path` setup as other scripts (see `testing_module.py`: adds `optimization/`). Then:

```python
from profiling import Profiler, quick_profile

profiler = Profiler()
```

### 6.2 Profile one layer

`profiler.profile_layer(layer, input_shape)` — parameters, FLOPs, memory, latency, `gflops_per_second`.

**Note:** internal `warmup=3`, `iterations=10` — quick snapshot, not a publication benchmark.

### 6.3 Profile full forward

`profiler.profile_forward_pass(model, input_tensor)` chains parameter count, FLOPs, memory, latency (warmup=5, iterations=20), derived throughput, and bottleneck hint.

`quick_profile(model, input_tensor)` prints a summary.

### 6.4 Training-oriented estimates

`profile_backward_pass` **reuses** forward profiling, then **estimates** backward FLOPs/latency as **2× forward** (simplification for teaching) and adds **optimizer memory** guesses (e.g. Adam ~2× gradient memory).

**Why estimates?** Real backward depends on **which tensors require grad**, kernel fusion, and implementation. Here we **teach the cost stack**, not replace PyTorch’s autograd profiler.

### 6.5 Optional: weight distribution

`analyze_weight_distribution` — links magnitudes to **quantization** / **regularization** discussions.

---

## Part 7 — How to interpret results

### 7.1 Bottleneck label

**Memory vs compute** is inferred from **relative** GFLOP/s vs MB/s. Treat as a **hypothesis**: validate with knowledge of layer sizes, batch, and hardware.

### 7.2 `computational_efficiency`

Compared to a **fixed placeholder** peak (`100` GFLOP/s in code). **Why it exists:** gives a **normalized** scalar for teaching. **How to use:** compare **before vs after** on the **same machine**, not the absolute percentage.

### 7.3 `memory_efficiency`

Useful memory vs **tracemalloc** peak. Low values suggest **overhead** or **temporaries** during forward—qualitative on CPython.

---

## Part 8 — Worked micro-example (mental arithmetic)

**Setup:** `Linear(16, 32)`, no bias in your head or with bias in code.

**Parameters:** `16 × 32 + 32 = 544` (with bias).

**FP32 weight memory (order of magnitude):** `544 × 4` bytes ≈ **2.1 KB** (weights alone).

**FLOPs (this repo’s linear formula):** `16 × 32 × 2 = 1024` FLOPs per **logical** matmul step as counted (remember batch handling in your interpretation).

**Why do this on paper?** You build **expectations** before you trust a profiler’s printout—you catch shape bugs and unit mistakes faster.

---

## Part 9 — Common misconceptions

1. **“Low parameter count ⇒ fast.”** Not if activations are huge or Python overhead dominates.
2. **“High FLOPs ⇒ slow.”** Not if the implementation is memory-bound or dominated by non-FLOP work.
3. **“One timing is enough.”** Always suspect **noise**; use warmup + many runs + median.
4. **“tracemalloc peak = GPU memory.”** No—this submodule is **CPU / CPython** oriented.
5. **“Backward is exactly 2× forward.”** A useful **pedagogical** rule of thumb here, not a theorem for every op.
6. **“Bottleneck label is always correct.”** It is a **heuristic**; corroborate with reasoning.

---

## Part 10 — Close the loop: the engineering cycle

```
1. Baseline profile   →  2. Form hypothesis (which knob: width, depth, batch, precision, fusion…)
        ↑                           ↓
4. Record outcome     ←  3. Change **one** major variable when possible
```

- **Control variables:** if you change batch size **and** architecture, you may not know **which** moved latency.
- **Always** re-run the **same** measurement procedure after a change.

---

## Part 11 — Prerequisites and limits

**Prerequisites:** `Tensor`, `Linear` / `Sequential`, and `forward` passes.

**Limits:**

- **CPU** timing and **CPython** `tracemalloc`; GPUs need vendor profilers for deep insight.
- **Analytic FLOPs** for a subset of layers; others are rough.
- **Backward** is **estimated**, not executed and timed in `profile_backward_pass`.

---

## Quick reference — API map

| Goal | Call |
|------|------|
| Parameter count | `profiler.count_parameters(model)` |
| Forward FLOPs (supported layers) | `profiler.count_flops(model, input_shape)` |
| Memory snapshot (forward) | `profiler.measure_memory(model, input_shape)` |
| Latency (custom rigor) | `profiler.measure_latency(model, input_tensor, warmup=..., iterations=...)` |
| One layer, combined | `profiler.profile_layer(layer, input_shape)` |
| Full forward summary | `profiler.profile_forward_pass(model, input_tensor)` |
| Forward + backward *estimates* | `profiler.profile_backward_pass(model, input_tensor)` |
| Pretty print | `quick_profile(model, input_tensor)` |

---

Profiling is **meaningful** when you know **why** each number exists and **how** it was produced. Structural counts, analytic FLOPs, and empirical latency/memory answer **different** questions; together they form the **evidence** you need to optimize without guessing.
