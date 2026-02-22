### 1. Introduction To Training

Training is the process that transforms a randomly initialized neural network into an intelligent system that makes predictions which in turn solve problems.

The training process follows a consistent sequence of actions across all machine learning:

1.  **Forward Pass**: Input flows through the model to produce predictions.
2.  **Loss Calculation**: We compare predictions to the true answers.
3.  **Backward Pass**: We compute the gradients showing how to improve.
4.  **Parameter Update**: We adjust the model's weights using an optimizer.
5.  **Repeat**: We continue until the model learns the pattern.



Production training systems need more than this basic loop:
* **Learning Rate Scheduling**: Should start high for rapid progress, then decay for stable convergence.
* **Gradient Clipping**: Gradients sometimes explode (become too large) and need to be capped.
* **Checkpointing**: Long training runs require saving states to survive crashes.
* **Mode Switching**: Models need separate train and evaluation modes.

This module builds all this infrastructure into a complete **Trainer class** that mirrors the PyTorch Lightning and Hugging Face training systems.

---

### 2. Training Loop Mathematics

The core training loop implements gradient descent with sophisticated improvements.

#### **Basic Update Rule**

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

Where:
* $\theta$: Parameters (weights)
* $\eta$: Learning rate
* $\nabla L$: Loss gradient

#### **Learning Rate Scheduling**

Why do we need to apply learning rate scheduling? Recall the gradient descent update:
```
updated_weight = old_weight - learning_rate x gradient_of_loss
```
The learning rate ($\eta$) determines how big steps we take, how fast we learn, and whether we converge or diverge.

* **Constant Learning Rate Risks**:
    * **Large $\eta$**: Overshoots the minimum, oscillates, and potentially diverges.
    * **Small $\eta$**: Training is too slow; takes tiny steps downhill.



**The Ideal Approach**:
1.  **Start Large**: Rapidly descend downhill.
2.  **End Small**: Precise convergence to the global minimum without oscillating.

A recommended learning rate scheduler is **Cosine Annealing**. For cosine annealing over $T$ total epochs:

$$\eta(t) = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min}) \left(1 + \cos\left(\frac{\pi t}{T}\right)\right)$$

#### **Gradient Clipping**

Gradient clipping prevents gradients from becoming too large, which destabilizes learning. During backpropagation (chain rule), repeated multiplication can cause gradients to grow exponentially.

This exponential growth causes:
1.  Unstable training.
2.  Huge weight updates.
3.  Loss becoming NaN (Not a Number).
4.  Failure to converge.

**The Solution**: Limit the maximum size of gradients before updating weights.

**Clipping by Norm**: Scale the entire gradient vector if its magnitude exceeds a threshold.
$\[
\nabla L \leftarrow
\begin{cases}
\nabla L & \text{if } \lVert \nabla L \rVert \le \text{max\_norm} \\
\nabla L \cdot \frac{\text{max\_norm}}{\lVert \nabla L \rVert} & \text{if } \lVert \nabla L \rVert > \text{max\_norm}
\end{cases}
\]$


#### **Gradient Accumulation**

Gradient accumulation is a technique used when we want the effect of a large batch size but cannot fit that large batch into GPU memory.

**How it Works**:
Suppose we want an effective batch size of $B_{eff} = 64$ but can only fit $B_{actual} = 16$ in memory.

1.  **Initialize**: Set gradients to zero.
2.  **Loop**: For each mini-batch of size 16:
    * Compute loss.
    * Backpropagate to get gradients.
    * **Accumulate**: Add gradients to existing gradients, but *do not* update weights yet.
3.  **Update**: After 4 mini-batches ($16 \times 4 = 64$):
    * Update weights once.
    * Zero the gradients.



**Mathematical View**:
Let mini-batch losses be $l_1, l_2, \dots, l_n$ where $n = B_{eff} / B_{actual}$.

1.  Compute gradients: $g_i = \nabla L_i$
2.  Accumulate gradients: $G = \sum g_i$
3.  Update weights: $\theta = \theta - \eta \cdot \frac{G}{n}$

Dividing by $n$ keeps the learning rate consistent with the actual batch size. The formula for accumulated gradient is:

$$\nabla L_{accumulated} = \frac{1}{\text{accumulation\_steps}} \sum_{i=1}^{n} \nabla L_{\text{batch\_i}}$$

### 3.Train vs Eval Modes
Many layer behave differently during training vs inference:
- **Dropout**:it's active during training but disabled during evaluation.
- **BatchNorm**:it updates statistics during training, it uses fixed statistics during evaluation.
- **Gradient computation**: it is enabled during training, its disabled during evaluation for efficient efficiency

### 4. The Trainer Class
The Trainer class manages the entire training process so we don't have to write boilerplate code.
It coordinates all the components i.e model,optimizer,loss function and schedular to conduct succesful training process.

#### Training Loop Architecture
The training loop follows a consistent pattern across all machine:
```
Training Loop Structure:

for epoch in range(num_epochs):
    ┌─────────────────── TRAINING PHASE ───────────────────┐
    │                                                      │
    │  for batch in dataloader:                            │
    │      ┌─── Forward Pass ───────┐                      │
    │      │ 1. input → model       │                      │
    │      │ 2. predictions         │                      │
    │      └────────────────────────┘                      │
    │               ↓                                      │
    │      ┌─── Loss Computation ───┐                      │
    │      │ 3. loss = loss_fn()    │                      │
    │      └────────────────────────┘                      │
    │               ↓                                      │
    │      ┌─── Backward Pass ──────┐                      │
    │      │ 4. loss.backward()     │                      │
    │      │ 5. gradients           │                      │
    │      └────────────────────────┘                      │
    │               ↓                                      │
    │      ┌─── Parameter Update ───┐                      │
    │      │ 6. optimizer.step()    │                      │
    │      │ 7. zero gradients      │                      │
    │      └────────────────────────┘                      │
    └──────────────────────────────────────────────────────┘
             ↓
    ┌─── Learning Rate Update ───┐
    │ 8. scheduler.step()        │
    └────────────────────────────┘
```
#### Key Features 
-**train/eval modes** : different behavior during training vs evaluation
-**gradient accumulation**:effective large batch sizes with limited memory.
-**checkpointing**:save/resume training state for long experiments
-**progress tracking**:monitor loss, learning rate and other metrics.

### 5. System Analysis
Training systems have significant recourse requirements.
Understanding memory usage is crucial in ML systems.

#### Training Memory Breakdown
```
Training Memory Requirements:

Forward Pass Memory:
┌─────────────────┐
│ Activations     │ ← Stored for backward pass
├─────────────────┤
│ Model Params    │ ← Network weights
└─────────────────┘

Backward Pass Memory:
┌─────────────────┐
│ Gradients       │ ← Same size as params
├─────────────────┤
│ Optimizer State │ ← 2-3× params (momentum, Adam buffers)
└─────────────────┘

Checkpoint Memory:
┌─────────────────┐
│ Model State     │ ← Full parameter snapshot
├─────────────────┤
│ Optimizer State │ ← All momentum/Adam buffers
├─────────────────┤
│ Training Meta   │ ← Epoch, history, scheduler
└─────────────────┘

Total Training Memory ≈ 5-6× Model Parameters
```

#### Key system insighets
**Gradient Accumulation Trade-off:**
-Effective batch size = accumulation_steps x actual_batch_size
-Memory:Fixed (only 1 batch in memory at a time)
-Time: Increases linearly with accumulation steps

**Checkpoint size**
Base model: 1x parameters
With optimizer(Adam): ~3x parameters
With full history: additional metadata 
Compression:pickle overhead ~10-20%
