### Computational Cost Analysis

Different activations have different computational profiles:

**ReLU: O(n) comparisons**
- Simple element-wise comparison: max(0, x)
- Fastest activation function (baseline)
- No exponentials, no divisions
- Ideal for large hidden layers

**Sigmoid/Tanh: O(n) exponentials**
- Each element requires exp() computation
- 3-4× slower than ReLU
- Exponentials are expensive operations
- Use sparingly in hidden layers

**GELU: O(n) exponentials + multiplications**
- Approximation involves sigmoid (exponential)
- 4-5× slower than ReLU
- Worth the cost in transformers (better gradients)
- Trade-off: performance vs. optimization quality

**Softmax: O(n) exponentials + O(n) sum + O(n) divisions**
- Most expensive: exp, sum, divide for entire vector
- Use only for output layers (not hidden layers)
- Requires synchronization across dimension
- Numerical stability tricks add overhead

### Numerical Stability Considerations

Activations can fail catastrophically without proper handling:

**Sigmoid/Tanh overflow:**
```
Problem: exp(1000) = inf, exp(-1000) = 0
Solution: Clip inputs to reasonable range (±500)
Our implementation: Uses stable computation for Sigmoid
```

**Softmax catastrophic overflow:**
```
Problem: exp(1000) = inf, causing NaN
Solution: Subtract max before exp (doesn't change result)
Your implementation: Uses max subtraction in Softmax.forward()
```

**ReLU dying neurons:**
```
Problem: Large negative gradient → weights become negative → ReLU always outputs 0
Solution: Monitor dead neuron percentage, use LeakyReLU variants

```

### Gradient Behavior Preview

understanding gradient characteristics helps:

**ReLU gradient: Sharp discontinuity**
- Gradient = 1 if x > 0, else 0
- Sharp corner at zero
- Dead neurons never recover (gradient = 0 forever)


**Sigmoid/Tanh gradient: Vanishing problem**
- Gradient ≈ 0 for large |x|
- Deep networks struggle (gradients die in early layers)
- Why ReLU replaced sigmoid in hidden layers

**GELU gradient: Smooth everywhere**
- No sharp corners (unlike ReLU)
- No vanishing at extremes (like sigmoid)
- Best of both worlds (modern architectures use this)

**Softmax gradient: Coupled across dimension**
- Changing one input affects all outputs
- Jacobian matrix (not element-wise)
- More complex backward pass than others

  
### Activation Selection Guide

**When to Use Each Activation:**

**Sigmoid**
- **Use case**: Binary classification output layers, gates in LSTMs/GRUs
- **Production example**: Spam detection (output: probability of spam)
- **Why**: Outputs valid probabilities in (0, 1)
- **Avoid**: Hidden layers in deep networks (vanishing gradients)

**ReLU**
- **Use case**: Hidden layers in CNNs, feedforward networks
- **Production example**: Image classification networks (ResNet, VGG)
- **Why**: Fast computation, prevents vanishing gradients, creates sparsity
- **Avoid**: Output layers (can't output negative values or probabilities)

**Tanh**
- **Use case**: RNN hidden states, when zero-centered outputs matter
- **Production example**: Sentiment analysis RNNs, time series prediction
- **Why**: Zero-centered helps with gradient flow in recurrent networks
- **Avoid**: Very deep networks (still suffers from vanishing gradients)

**GELU**
- **Use case**: Transformer models, modern architectures
- **Production example**: GPT, BERT, modern language models
- **Why**: Smooth approximation of ReLU, better gradient flow, state-of-the-art results
- **Avoid**: When computational efficiency is critical (slightly slower than ReLU)

**Softmax**
- **Use case**: Multi-class classification output layers
- **Production example**: ImageNet classification (1000 classes), NLP token prediction
- **Why**: Converts logits to valid probability distribution (sums to 1)
- **Avoid**: Hidden layers (loses information through normalization)

