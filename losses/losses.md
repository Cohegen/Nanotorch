"""
## Introduction to Loss Functions
Loss functions are mathematical conscience of machine learning. They measure the distance betweeen what the model predicts and what actually happened. 

## Three Essential Loss functions
The loss functions measures how wrong our model is on each prediction it has made. The essential loss functions include:
**MSELoss (Mean Squared Error)**: used for regression
- Calculation: the average of squared distances between predictions and targets is calculated.
- Properties: it heavily penalizes large errors  and smooth gradients.

**CrossEntropyLoss**
- Used for multiclass classification e.g image recognition in short the model asks itself "how confident am I i've classified this object in its true class."
- It is calculated by taking the Negative log-likelihood of correct class probability.
- One of its notably properties is that it encourages confident correct predictions, punishes confident incorrect ones.

**BinartCrossEntropyLoss**
Used for binary classification e.g spam detection
- It is property is that it makes symmetric penalties for false positives and false negatives.

```
Binary Decision Boundary:
     Target=1 (Positive)    Target=0 (Negative)
     ┌─────────────────┬─────────────────┐
     │  Pred → 1.0     │  Pred → 1.0     │
     │  Loss → 0       │  Loss → ∞       │
     ├─────────────────┼─────────────────┤
     │  Pred → 0.0     │  Pred → 0.0     │
     │  Loss → ∞       │  Loss → 0       │
     └─────────────────┴─────────────────┘
```
"""

"""
## Mathematical Foundations

## Mean Squared Error
The foundation of regression, MSE measures the average squared distance between predictions and targets:

```
MSE = (1/N) * Σ(prediction_i - target_i)²
```

**Why square the differences**
-Mkaes all errors positive.
-Heavily penalizes large errors.
-Creates smooth gradients for optimization


## Cross-Entropy Loss
For classification, we need to meausre how wrong the probability distributions are: 

```
CrossEntropy = -Σ target_i * log(prediction_i)
```

**The Log-Sum-Exp Trick**
Computing softmax directly can cause numerical overflow. The log-sum-exp trick provides stability.

```
log_softmax(x) = x - log(Σ exp(x_i))
                = x - max(x) - log(Σ exp(x_i - max(x)))
```
This prevent exp(large_number) from exploding to infinity

## Binary Cross-Entropy
A specialized case where we have only two classes:
```
BCE = -(target * log(prediction) + (1-target) * log(1-prediction))
```

The mathematics naturally handles both "positive" and "negative" cases in a single formula.
"""
## Log-Softmax
**Why Log-Softmax matters**

Naive softmax can explode with large numbers:

```
Naive approach:
    logits = [100,200,300]
    exp(300) = 1.97x10^130 is too large hence breaks computers!


Stable approach"
     max_logit = 300
     shifted = [-200,-100,0] <- subract each term with maximum logit
     exp(0) = 1.0 <- Managable numbers.
```

### The log-Sum-exp trick visualization

```
Original Computation:           Stable Computation:

logits: [a, b, c]              logits: [a, b, c]
   ↓                              ↓
exp(logits)                    max_val = max(a,b,c)
   ↓                              ↓
sum(exp(logits))               shifted = [a-max, b-max, c-max]
   ↓                              ↓
log(sum)                       exp(shifted)  ← All ≤ 1.0
   ↓                              ↓
logits - log(sum)              sum(exp(shifted))
                                  ↓
                               log(sum) + max_val
                                  ↓
                               logits - (log(sum) + max_val)
```
Both give the same result, but the stable version never overflows
"""

"""
 ### MSELoss - Measuring Continous Prediction Quality
Mean Squared Erro is a crucial loss function for regression problems. It measures how far from the true values.

### When to use MSE
**MSE loss is perfect for:**
- House price prediction (ksh2000 vs ks1500)
- Temperature forecasting 
- Stock price prediction ($150 vs $ 160)
- Any coninous value where "distance" matters.

### How MSE shapes Learning

```
Prediction vs Target Visualization:

Target = 100

Prediction: 80   90   95   100  105  110  120
Error:     -20  -10   -5    0   +5  +10  +20
MSE:       400  100   25    0   25  100  400


Quadratic penalty: Large errors are MUCH more costly than small errors.

```

### Why Squaring the errors matters
1. **Positive penalties**: (-10)^2 = 100, smae as (+10)^2 = 100
2. **Heavy punishment for large errors**: Error of 20 -> penalty of 400.
3. **Smooth gradients**: Quadratic function has nice derivatives for optimization.
4. **Statistical foundation** : Maximum likelihood for Gaussian noise.


### MSE vs Other Regression Losses

```
Error Sensitivity Comparison:

 Error:   -10    -5     0     +5    +10
 MSE:     100    25     0     25    100  ← Quadratic growth
 MAE:      10     5     0      5     10  ← Linear growth
 Huber:    50    12.5   0    12.5    50  ← Hybrid approach

 MSE: More sensitive to outliers
 MAE: More robust to outliers
 Huber: Best of both worlds
```
"""

"""
CrossEntropyLoss 
It measres classification confidence
Cross-entropy loss is the gold standars for multi-class classification. It measures how wrong your probability predictions are and heavily penalizes confident mistakes.

**Perfect for:**
- Image classification (cat,dog,bird)
- Text classification (spam,ham,promotion)
- Language modelling (next word prediction)
- Any problem with mutually exclusive classes

### Undestanding Cross-Entropy Through Examples

```
Scenario: Image Classification (3 classes: cat, dog, bird)

Case 1: Correct and Confident
Model Output (logits): [5.0, 1.0, 0.1]  ← Very confident about "cat"
After Softmax:        [0.95, 0.047, 0.003]
True Label:           cat (class 0)
Loss: -log(0.95) = 0.05  ← Very low loss ✅

Case 2: Correct but Uncertain
Model Output:         [1.1, 1.0, 0.9]  ← Uncertain between classes
After Softmax:        [0.4, 0.33, 0.27]
True Label:           cat (class 0)
Loss: -log(0.4) = 0.92  ← Higher loss (uncertainty penalized)

Case 3: Wrong and Confident
Model Output:         [0.1, 5.0, 1.0]  ← Very confident about "dog"
After Softmax:        [0.003, 0.95, 0.047]
True Label:           cat (class 0)
Loss: -log(0.003) = 5.8  ← Very high loss ❌
```

### Cross-Entropy's Learning Signal

```
What Cross-Entropy Teaches the Model:

┌─────────────────┬─────────────────┬───────────────────────────┐
│ Prediction      │ True Label      │ Learning Signal           │
├─────────────────┼─────────────────┼───────────────────────────┤
│ Confident ✅    │ Correct ✅      │ "Keep doing this"         │
│ Uncertain ⚠️    │ Correct ✅      │ "Be more confident"       │
│ Confident ❌    │ Wrong ❌        │ "STOP! Change everything" │
│ Uncertain ⚠️    │ Wrong ❌        │ "Learn the right answer"  │
└─────────────────┴─────────────────┴───────────────────────────┘
Message: "Be confident when you're right!"
```

### Why Cross-Entropy Works So Well

1. **Probabilistic interpretation**: Measures quality of probability distributions
2. **Strong gradients**: Large penalty for confident mistakes drives fast learning
3. **Smooth optimization**: Log function provides nice gradients
4. **Information theory**: Minimizes "surprise" about correct answers

### Multi-Class vs Binary Classification

```
Multi-Class (3+ classes):          Binary (2 classes):

Classes: [cat, dog, bird]         Classes: [spam, not_spam]
Output:  [0.7, 0.2, 0.1]         Output:  0.8 (spam probability)
Must sum to 1.0                   Must be between 0 and 1 
Uses: CrossEntropyLoss            Uses: BinaryCrossEntropyLoss
```
"""

"""
## BinaryCrossEntropyLoss - it measures Yes/No decision Quality
Binary Cross-Entropy is specialized for yes/no decisions. It's like a regular cross-entropy loss but for exactly two classes.

**Perfect for:**
- Spam detection (spam vs not spam)
- Medical diagnosis (disease vs healthy)
- Fraud detection (fraud vs legitimate)
- Content moderation (toxic vs sage)
- Any two-class decision problem

### Understanding Binary CrossEntropy
```
Binary Classification Decision Matrix:

                 TRUE LABEL
              Positive  Negative
PREDICTED  P    TP       FP     ← Model says "Yes"
           N    FN       TN     ← Model says "No"

BCE Loss for each quadrant:
- True Positive (TP): -log(prediction)    ← Reward confident correct "Yes"
- False Positive (FP): -log(1-prediction) ← Punish confident wrong "Yes"
- False Negative (FN): -log(prediction)   ← Punish confident wrong "No"
- True Negative (TN): -log(1-prediction)  ← Reward confident correct "No"
```

### Binary Cross-Entropy Behavior Examples

```
Scenario: Spam Detection

Case 1: Perfect Spam Detection
Email: "Buy now! 50% off! Limited time!"
Model Prediction: 0.99 (99% spam probability)
True Label: 1 (actually spam)
Loss: -log(0.99) = 0.01  ← Very low loss 

Case 2: Uncertain About Spam
Email: "Meeting rescheduled to 2pm"
Model Prediction: 0.51 (slightly thinks spam)
True Label: 0 (actually not spam)
Loss: -log(1-0.51) = -log(0.49) = 0.71  ← Moderate loss

Case 3: Confident Wrong Prediction
Email: "Hi mom, how are you?"
Model Prediction: 0.95 (very confident spam)
True Label: 0 (actually not spam)
Loss: -log(1-0.95) = -log(0.05) = 3.0  ← High loss 
```

### Binary vs Multi-Class Cross-Entropy

```
Binary Cross-Entropy:              Regular Cross-Entropy:

Single probability output         Probability distribution output
Predict: 0.8 (spam prob)         Predict: [0.1, 0.8, 0.1] (3 classes)
Target: 1.0 (is spam)            Target: 1 (class index)

Formula:                         Formula:
-[y*log(p) + (1-y)*log(1-p)]    -log(p[target_class])

Handles class imbalance well     Assumes balanced classes
Optimized for 2-class case      General for N classes
```

### Why Binary Cross-Entropy is Special

1. **Symmetric penalties**: False positives and false negatives treated equally
2. **Probability calibration**: Output directly interpretable as probability
3. **Efficient computation**: Simpler than full softmax for binary cases
4. **Medical-grade**: Well-suited for safety-critical binary decisions
"""

"""
## Real-World Loss Function usage patterns
Understanding when and why we use loss functions is crucial for all ML engineers.

```
Problem Type Decision Tree:

What are you predicting?
         │
    ┌────┼────┐
    │         │
Continuous   Categorical
 Values       Classes
    │         │
    │    ┌───┼───┐
    │    │       │
    │   2 Classes  3+ Classes
    │       │       │
 MSELoss   BCE Loss  CE Loss



Examples:
MSE: House prices,temperature,stock prices
BCE: Spam detection,fraud detection,medical diagnosis.
CE : Image Classificatiion,language modelling,multiclass text classification
```

## Loss Function Behavior Comparison

Each loss function creates different learning pressures on the model.

```
Error Sensitivity Comparison

Small Error (0.1):     Medium Error (0.5):     Large Error (2.0):

MSE:     0.01         MSE:     0.25           MSE:     4.0
BCE:     0.11         BCE:     0.69           BCE:     ∞ (clips to large)
CE:      0.11         CE:      0.69           CE:      ∞ (clips to large)

MSE: Quadratic growth, managable with outliers.
BCE/CE: Logarithmic growth, explodes with confident wrong predictions.
```

"""

"""
## System Analysis
Loss functions seem simple, but they have important computational and numerical properties that affect training performance.

## Computational Complexity Analysis

Different loss functions have different computational costs, especially at scale:

```
Computational Cost Comparison (Batch Size B, Classes C):

MSELoss:
┌────────────────┬────────────────┐
│ Operation      │ Complexity     │
├────────────────┼────────────────┤
│ Subtraction    │ O(B)           │
│ Squaring       │ O(B)           │
│ Mean           │ O(B)           │
│ Total          │ O(B)           │
└────────────────┴────────────────┘

CrossEntropyLoss:
┌────────────────┬────────────────┐
│ Operation      │ Complexity     │
├────────────────┼────────────────┤
│ Max (stability)│ O(B*C)         │
│ Exponential    │ O(B*C)         │
│ Sum            │ O(B*C)         │
│ Log            │ O(B)           │
│ Indexing       │ O(B)           │
│ Total          │ O(B*C)         │
└────────────────┴────────────────┘

Cross-entropy is C times more expensive than MSE!
For ImageNet (C=1000), CE is 1000x more expensive than MSE.
```

## Memory Layout and Access Patterns

```
Memory Usage Patterns:
MSE Forward Pass:              CE Forward Pass:

Input:  [B] predictions       Input:  [B, C] logits
       │                             │
       │ subtract                    │ subtract max
       v                             v
Temp:  [B] differences        Temp1: [B, C] shifted
       │                             │
       │ square                      │ exponential
       v                             v
Temp:  [B] squared            Temp2: [B, C] exp_vals
       │                             │
       │ mean                        │ sum along C
       v                             v
Output: [1] scalar            Temp3: [B] sums
                                     │
Memory: 3*B*sizeof(float)            │ log + index
                                     v
                              Output: [1] scalar

                              Memory: (3*B*C + 2*B)*sizeof(float)
```
"""

"""
## Production Context - How Loss functions Scale

Understanding how loss functions behave in production helps make informed engineering decisions about model architecture and training strategies

## Loss function scaling challenges
As models grow larger, loss function bottlenecks become critical:

```
Scaling Challenge Matrix:

                    │ Small Model     │ Large Model      │ Production Scale  │
                    │ (MNIST)         │ (ImageNet)       │ (GPT/BERT)        │
────────────────────┼─────────────────┼──────────────────┼───────────────────┤
Classes (C)         │ 10              │ 1,000            │ 50,000+           │
Batch Size (B)      │ 64              │ 256              │ 2,048             │
Memory (CE)         │ 2.5 KB          │ 1 MB             │ 400 MB            │
Memory (MSE)        │ 0.25 KB         │ 1 KB             │ 8 KB              │
Bottleneck          │ None            │ Softmax compute  │ Vocabulary memory │

Memory grows as B*C for cross-entropy!
At scale, vocabulary (C) dominates everything.
```

##  Engineering Optimizations in Production

```
Common Production Optimizations:

1. Hierarchical Softmax:
   ┌─────────────────────┐     ┌─────────────────────┐
   │ Full Softmax:       │     │ Hierarchical:       │
   │ O(V) per sample     │ →   │ O(log V) per sample │
   │ 50k classes = 50k   │     │ 50k classes = 16    │
   │ operations          │     │ operations          │
   └─────────────────────┘     └─────────────────────┘

2. Sampled Softmax:
   Instead of computing over all 50k classes,
   sample 1k negative classes + correct class.
   50× speedup for training!

3. Label Smoothing:
   Instead of hard targets [0, 0, 1, 0],
   use soft targets [0.1, 0.1, 0.7, 0.1].
   Improves generalization.

4. Mixed Precision:
   Use FP16 for forward pass, FP32 for loss.
   2× memory reduction, same accuracy.
```
"""
