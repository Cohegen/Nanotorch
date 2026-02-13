
"""
### Introduction To Training
Training is the process that transforms a randomly intialized neural network into an intelligent system that makes predictions which in turn solve problems.

The training process follows a consistent sequence of actions across all machine learning:
1.**Forward Pass**:Input flows through the model to produce predictions.
2.**Loss Calculation**: here we compare predictions to the true answers
3.**Backward pass**: this is the stage where we compute the gradients showing how to improve
4.**Parameter Update**: here we adjust the model's weights using an optimizer.
5.**Repeat**: we continue until the model learns the pattern.

Production training systems need more than this basic loop.
Learning rate should start high for rapid progress, then decay for stable convergence.
Gradients sometimes explode (become to large) and need clipping.
Long training runs require checkpointing to survive crashes.
Models need seperate train and evaluation  models.
This module builds all this infrastructure into a complete Trainer class that mirrors the Pytorch Lightning and Hugging Face training systems.
"""

"""
### Training Loop Mathematics
The core training loop implements gradient descent with sophisticated improvements:

**Basic Update Rule:**
```
θ(t+1) = θ(t) - η ∇L(θ(t))
```
Where θ are parameters, η is learning rate, and ∇L is the loss gradient.

**Learning Rate Scheduling:**
Why do we need to apply learning rate scheduling in the first place?
Recall from optimizers, gradient descent update was :
```
updated_weight = old_weight - learning_rate*gradient_of_loss_wrt_weight
```
Here the learning rate determines: how big steps we take, how fast we learn, whether you converge or diverge.
Normally we at times used a constant learning rate.
However, this method has some potential risks.
Like for instance if we used a large learning rate, it would overshoot to minimum, the oscillation downhill would be large and instead of converging it would be large.
If the learning rate was small, the training would be slow since we would converge to the global minimum slowly since small learning rate results to tiny steps downhill.
So to solve this our ideal approach would be:
1.Make the learning rate large at the beginning which results to fast descending downhill.
2.Make the learning rate small at the end so to have precise convergence i.e reach the global minimum without oscillating over it. 
This is what learning rate scheduling actual does.
So learning rate scheduling is the systematic process of changing the learning rate during training.
A optimal option learning rate schedular to use is the cosine annealing
For cosine annealing over T epochs:
```
η(t) = η_min + (η_max - η_min) * (1 + cos(πt/T)) / 2
```

**Gradient Clipping:**
When ||∇L|| > max_norm, rescale:
```
∇L ← ∇L * max_norm / ||∇L||
```
"""