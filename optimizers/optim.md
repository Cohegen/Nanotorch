"""
We will get the know how we enable neural netwoeks to learn from gradients using sophisticated algorithms.
It's simply asking ourselves like what happens after we compute gradient do we just let the model come up with its own strategy of what to do with them?
We are going to answer this question right here..
A good traveling map for our quest here is:

Gradients → Optimizers → 
(from autograd)  (from this directory)

**Why this matter**
1.**Learning**: complete optimization system for modern
neural network training.
2.**Production**:proper orgainization like Pytorch's torch.optim with all optimizers.



"""

"""
## Intoduction to Optimizers
Optimizers are the engines that drive neural network learning.
They take gradients computed from your loss function and use them to update to model parameters towards better solutions.
Think of optimization as a way of navigating across a complex landscape of high terrain and trying to find the lowest point of the landscape.

## Optimizers we use
**SGD(Stochastic Gradient Descent)**
-Strategy: Always step downhill
-Problem: Can get stuck oscillating in narrow valleys.
-Solution: Add momentum to "coast" through oscillations.

**Adam (Adaptive Moment Estimation)**
-Strategy:Adapt step size for each parameter individually.
-Advantage: different learning rates for different dimensions.
Note: Some directions need big steps,other need small steps.

**AdamW(Adam with Weight Decay)**
-Strategy:Adam + proper regularization
-Fix:seperates optimization from regularization
-Result:better generalization and training stability


## The Mathematics Behind Movement

At its core, optimization follows: **θ_new = θ_old - α * direction**

Where:
- θ` = parameters (your position in the landscape)
- `α` = step size (learning rate)
- `direction` = where to step (gradient-based)

But sophisticated optimizers do much more than basic gradient descent.
"""

"""
Foundations:Mathematical Background

##Understanding Momentum: The Physics of Optimization

Momentum in optimization works like momentum in Physics.
A ball rolling down a hill doesn't immediately change direction when it hits a small bump - it has momentum that carries it forward.

**SGD with Momentum Formula:**
```
velocity = β * previous_velocity + (1-β) * current_gradient
parameter = parameter - learning rate * velocity

Where β ≈ 0.9 means "90% memory of previous direction"

```

##Adam:Adaptive Learning for Each Parameter

Adam solves a key problem: different parameters need different learning rates.  
Imagine adjusting the focus and zoom on a camera- you need fine control for focus but coarse control for zoom.

Adam Solution: Automatic step size per parameter.


**Adam's Two Memory System:**

1. **First Moment(m)** : "Which direction am I usually going?"
    - `m = β₁ * old_m + (1-β₁) * gradient`
   - Like momentum, but for direction

2. **Second Moment (v)**: "How big are my gradients usually?"
    - `v = β₂ * old_v + (1-β₂) * gradient²`
   - Tracks gradient magnitude

3. **Adaptive Update**:
     - `step_size = m / √v`
    - Big gradients → smaller steps
    - Small gradients → relatively bigger steps


## AdamW: Fixing Weight Decay

Adam has subtle bug in how it applies weight decay (regularization). AdamW fices this 

```
Adam (incorrect):               AdamW (correct):
gradient += weight_decay * param    [compute gradient update]
update_param_with_gradient()        param -= learning_rate * gradient_update
                                   param *= (1 - weight_decay)  ← separate!

Why it matters:
- Adam: Weight decay affected by adaptive learning rates
- AdamW: Weight decay is consistent regardless of gradients
```
"""

"""
##Stochastic Gradient Descent 
SGD is the foundation of neural network perf.
It implements the simple but powerful idea i.e "move in the direction opposite to the gradient."

##Why SGD Works 

Gradients point uphill i.e  torward higher loss.
To minimize loss, we go downhill:

## Why it is  called Stochastic?
"Stochastic" means random.

Here instead of using entire dataset to calculate the gradient which can be slow.
SGD updates weights using one random data point at a time.

##The Oscillation Problem
Pure SGD can get trapped osillating in narrows valleys:
 
##Moment Solution

Momentum remebers the direction you were going and continues in that direction.
"""

"""

## Adam - Adaptive Moment Estimation

Adam solves the fundemental problem with SGD i.e different parameters often need different  learning rate.
Think of it as a complex system which tries to adjust swtiches, some switches may need more adjustments than the other.

Consider a neural network with both embedding weights and output weights:

```
Output_weight : loss graph is steep hence take tiny steps when updating weights.
Embedding_weight: loss graph is gentle hence take big steps while updating weights.

Same learning rate is a disaster
Since:
   Small LR: output weights learn fast, embeddings crawl
   Large LR: embeddings learn well, output weights explode
```

### Adam's Adaptive Solution

Adam automatically adjusts learning rates by tracking two statistics:

```
1.MOMENTUM (first moment): "which way am I usally going?"
  m = 0.9* old_direction + 0.1* current_gradient

  Visualization:
   old: →→→→
   new:     ↗️
   m:   →→→↗️  (weighted average)

2.SCALE (second moment): "How big are my steps usually?"
v = 0.990 * old_scale + 0.0001 * (current_gradient)**2

Big gradients -> bigger v -> smaller effective steps
Small gradients -> smaller v -> bigger effective steps

3.ADAPTIVE UPDATE:
step = momentum / √scale
param = param - learning_rate * step 
```

### Bias Correction:

Adam starts with m=0 and v=0, which creates a bias toward zero initially:

```
Without bias correction:    With bias correction:

Step 1: m = 0.9*0 + 0.1*g    Step 1: m̂ = m / (1-0.9¹) = m / 0.1
       = 0.1*g (too small!)           = g (correct!)

Step 2: m = 0.9*0.1*g + 0.1*g Step 2: m̂ = m / (1-0.9²) = m / 0.19
       = 0.19*g (still small)         ≈ g (better!)
```

"""

"""
## AdamW ( Adam with Decoupled weight Decay)

AdamW fixes a subtle but importnat bug in Adam's weight decay implementation.

### The Adam Weight Decay Bug

In standard Adam, weight decay is added to gradients before adaptive scaling :

```
Adam's approach:
1.gradient = computed_gradient + weight_decay * parameter
2.m = β₁ * m + (1-β₁) * gradient
3.v = β₂ * v + (1-β₂) * gradient**2
4.step = m / sqrt(v)
5.parameter = parameter - learning_rate * step

Problem here is that Weight decay gets 'adapted' by learning rate scaling.
```
##Why this matters

Weight decay should be consistent regularization force, but Adam makes it inconsistent:

```
Parameter Update Comparison

Large gradients -> small adaptive LR-> weak weight decay effect
Small gradients-> large adaptive LR-> strong weight decay effect


```

## AdamW's fix (Decoupled Weight Decay)
AdamW seperates gradient-based updates from weight decay: 

```
AdamW's approach:
1.m = m = β₁ * m + (1-β₁) * pure_gradient 
2. v = β₂ * v + (1-β₂) * pure_gradient**2 
3. step = m/ sqrt(v)
4. parameter = parameter - learning_rate * step
5. parameter = parameter * (1 - weight_decay_rate)

This results in consistent regularization of gradient magnitude
```
"""

"""
## System Analysis
Different optimizers have very different resource requirements.

### Memory Usage Patterna

```
Optimizer Memory Requirements (per parameter):

SGD:           Adam/AdamW:
┌────────┐     ┌────────┐
│ param  │     │ param  │
├────────┤     ├────────┤
│momentum│     │   m    │ ← first moment
└────────┘     ├────────┤
               │   v    │ ← second moment
               └────────┘

2× memory       3× memory
```

### Computational Complexity

```
Per-step Operations:

SGD:                     Adam:
• 1 multiplication       • 3 multiplications
• 1 addition            • 4 additions
• 1 subtraction         • 1 subtraction
                        • 1 square root
                        • 1 division

O(n) simple ops         O(n) complex ops
```
"""
