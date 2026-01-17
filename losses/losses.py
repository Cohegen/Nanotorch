import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
from activations.activations import ReLU
from layers.layers import Linear
import numpy as np

#constant for numerical stability
EPILSON = 1e-7 #small value to prevent log(0) which is infinity and numerically unstable

def log_softmax(x:Tensor,dim: int =-1) -> Tensor:
    """
    Computes log-softmax so as to ensure numerical stability.
    """
    
    #finding the maximum element along a dimension
    max_val = np.max(x.data,axis=dim,keepdims=True)

    #2.Subracting max to prevent overflow
    shifted = x.data - max_val

    #3. Computing log(sum(exp(shifted)))
    log_sum_exp = np.log(np.sum(np.exp(shifted),axis=dim,keepdims=True))

    #4. Returning log_softmax = input- max -log_sum_exp
    result = x.data - max_val - log_sum_exp

    #5. Return as Tensor
    return Tensor(result)

class MSELoss:
    """Mean Squared Loss for regression tasks."""

    def __init__(self):
        """Initialize MSE loss function"""
        pass

    def forward(self,predictions: Tensor, targets: Tensor) -> Tensor:
        """Compute mean squared error between predictions and targets."""

        #1. Computing element-wise difference
        diff = predictions.data - targets.data

        #2. Squaring the differences
        squared_diff = diff**2

        #3. Take mean across all elemets
        mse = np.mean(squared_diff)

        return Tensor(mse)


    def __call__(self,predictions: Tensor, targets: Tensor) -> Tensor:
        """Allows the loss function to be called like a function"""
        return self.forward(predictions,targets)

    def backward(self)->Tensor:
        """Computes gradients"""
        pass

    

class CrossEntropyLoss:
    """CrossEntropy loss for multi-class classification."""

    def __init__(self):
        """Intialize cross-entropy loss function."""
        pass

    def forward(self,logits:Tensor,targets:Tensor) -> Tensor:
        """Computes cross-entropy loss between logits and target class indices"""

        #1.Compute log-softamx for numerical stability
        log_probs = log_softmax(logits,dim=-1)

        #2.Selecting log-probabilities for correct classes
        batch_size = logits.shape[0]
        target_indices = targets.data.astype(int)

        #selecting correct class log-probabilities using advanced indexing
        selected_log_probs = log_probs.data[np.arange(batch_size),target_indices]

        #3.Returning negative mean (cross-entropy is negative log-likelihood)
        cross_entropy = -np.mean(selected_log_probs)

        return Tensor(cross_entropy)

    def __call__(self,logits:Tensor,targets:Tensor)->Tensor:
        """Allows the loss function to be called like a function"""
        return self.forward(logits,targets)

    def backward(self):
        """Computes gradients"""
        pass


class BinaryCrossEntropyLoss:
    """Binary cross-entropy loss for binary classification."""

    def __init__(self):
        """Initialize binary cross-entropy loss function"""
        pass

    def forward(self,predictions:Tensor,targets:Tensor) -> Tensor:
        """Computes binary cross-entropy loss."""

        #1.Clamping predictions to avoid numerical issues with log(0) and log(1)
        eps = EPILSON
        clamped_preds = np.clip(predictions.data,eps,1-eps)

        #2.computing binary cross-entropy
        #BCE = -(targets*log(preds) + (1-targets)*log(1-preds))
        log_preds = np.log(clamped_preds)
        log_one_minus_preds = np.log(1- clamped_preds)

        bce_per_sample = -(targets.data*log_preds + (1-targets.data)*log_one_minus_preds)

        #3.Returning mean across all samples
        bce_loss = np.mean(bce_per_sample)

        return Tensor(bce_loss)

    def __call__(self,predictions:Tensor,targets:Tensor)->Tensor:
        """Allows the loss function to be called like a function"""
        return self.forward(predictions,targets)

    def backward(self)->Tensor:
        """Computes gradient"""
        pass

