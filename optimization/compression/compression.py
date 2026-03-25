import numpy as np
import copy 
from typing import List,Dict,Any,Tuple,Optional
import time 
import os
import sys

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Tensor import Tensor 
from layers.layers import Linear,Sequential
from activations.activations import ReLU

#constants for memory calculations
BYTES_PER_FLOAT32 =4 #standard float32 size bytes
MB_TO_BYTES = 1024*1024 #megabytes to bytes conversion


def measure_sparsity(model) ->float:
    """
    Calculates the percentage of zero weights 
    in a model.

    Args:
         model: Model with .parameters() method

    Returns:
         Sparsity percentage

    EXAMPLE:
    >>> # Create test model with explicit composition
    >>> layer1 = Linear(10, 5)
    >>> layer2 = Linear(5, 2)
    >>> model = Sequential(layer1, layer2)
    >>> sparsity = measure_sparsity(model)
    >>> print(f"Model sparsity: {sparsity:.1f}%")
    Model sparsity: 0.0%  # Before pruning
    """

    total_params = 0
    zero_params = 0

    for param in model.parameters():
        #only counting weight matrices (2D), not biases (1D)
        #biases are often intialized to zero, which would skew sparsity
        if len(param.shape) > 1:
            total_params += param.size 
            zero_params += np.sum(param.data == 0)

        if total_params == 0:
            return 0.0

        return (zero_params /total_params) * 100.0

    
def magnitude_prune(model,sparsity=0.9):
    """
    Removes weights with smalles magnitudes to achieve target sparsity

    EXAMPLE:
    >>> # Create model with explicit layer composition
    >>> layer1 = Linear(100, 50)
    >>> layer2 = Linear(50, 10)
    >>> model = Sequential(layer1, layer2)
    >>> original_params = sum(p.size for p in model.parameters())
    >>> magnitude_prune(model, sparsity=0.8)
    >>> final_sparsity = measure_sparsity(model)
    >>> print(f"Achieved {final_sparsity:.1f}% sparsity")
    Achieved 80.0% sparsity
    """

    ##collecting all weights (excluding biases)
    all_weights = []
    weights_params = []

    for param in model.parameters():
        #skipping biases
        if len(param.shape) >1 :
            all_weights.extend(param.data.flatten())
            weights_params.append(param)

        if not all_weights:
            return model

        #calculates magnitude threshold
        magnitudes = np.abs(all_weights)
        threshold = np.percentile(magnitudes,sparsity*100)

        #apply pruning to each weight parameter
        for param in weights_params:
            mask = np.abs(param.data) >= threshold
            param.data= param.data * mask

        return model

def structured_prune(model,prune_ratio=0.5):
    """
    Removes entire channels/neurons based on L2 norm importance.

     EXAMPLE:
    >>> # Create model with explicit layers
    >>> layer1 = Linear(100, 50)
    >>> layer2 = Linear(50, 10)
    >>> model = Sequential(layer1, layer2)
    >>> original_shape = layer1.weight.shape
    >>> structured_prune(model, prune_ratio=0.3)
    >>> # 30% of channels are now completely zero
    >>> final_sparsity = measure_sparsity(model)
    >>> print(f"Structured sparsity: {final_sparsity:.1f}%")
    Structured sparsity: 30.0%
    """
    #all linear layers have .weight attributes
    for layer in model.layers:
        if isinstance(layer,Linear):
            weight = layer.weight.data 

            #calculating L2 norm for each output channel (column)
            channel_norms = np.linalg.norm(weight,axis=0)

            #finding channels to prune (lowest importance)
            num_channels = weight.shape[1]
            num_to_prune = int(num_channels * prune_ratio)

            if num_to_prune > 0:
                #getting indices of channels to prune (smalles norm)
                prune_indices = np.argpartition(channel_norms,num_to_prune)[:num_to_prune]

                #zeros out entire channels
                weight[:,prune_indices] =0

                #also zeroing corresponding bias element if bias exists
                if layer.bias is not None:
                    layer.bias.data[prune_indices] =0

                
    return model

def low_rank_approximate(weight_matrix,rank_ratio=0.5):
    """
    Approximates weight matrix using low-rank decompisition (SVD)

    EXAMPLE:
    >>> weight = np.random.randn(100, 50)
    >>> U, S, V = low_rank_approximate(weight, rank_ratio=0.3)
    >>> # Original: 100*50 = 5000 params
    >>> # Compressed: 100*15 + 15*50 = 2250 params (55% reduction)

    """
    m,n =weight_matrix.shape

    #perform SVD
    U,S,V = np.linalg.svd(weight_matrix,full_matrices=False)

    #determining target rank
    max_rank = min(m,n)
    target_rank = max(1,int(rank_ratio*max_rank))

    #truncating to target rank
    U_truncated = U[:,:target_rank]
    S_truncated = S[:target_rank]
    V_truncated = V[:target_rank,:]

    return U_truncated,S_truncated, V_truncated


class KnowledgeDistillation:
    """
    Knowledge distillation for model compression.

    Trains a smaller student model to mimic a larger teacher model.
    """

    def __init__(self,teacher_model,student_model,temperature=3.0,alpha=0.7):
        """
        Initializes knowledge distillation.


        Args:
           teacher_model:Large,pre-trained model
           student_model: smaller model to train
           temperature: softening parameter for distributions
           alpha:weight for soft target loss (1-alpha for hard targets)

         EXAMPLE:
        >>> # Create teacher with more capacity (explicit layers)
        >>> teacher_l1 = Linear(100, 200)
        >>> teacher_l2 = Linear(200, 50)
        >>> teacher = Sequential(teacher_l1, teacher_l2)
        >>>
        >>> # Create smaller student (explicit layer)
        >>> student = Sequential(Linear(100, 50))
        >>>
        >>> kd = KnowledgeDistillation(teacher, student, temperature=4.0, alpha=0.8)
        >>> print(f"Temperature: {kd.temperature}, Alpha: {kd.alpha}")
        Temperature: 4.0, Alpha: 0.8
        """
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.temperature = temperature
        self.alpha = alpha
        

    def distillation_loss(self,student_logits,teacher_logits,true_labels):
            """
            Calculates combined distillation loss.

             EXAMPLE:
        >>> kd = KnowledgeDistillation(teacher, student)
        >>> loss = kd.distillation_loss(student_out, teacher_out, labels)
        >>> print(f"Distillation loss: {loss:.4f}")

            """
            #extracting numpy array from Tensors
            #student_logits and teacher_logits are always Tensor from forward passes
            student_logits = student_logits.data 
            teacher_logits = teacher_logits.data 

            #true_labels might be numpy array or Tensor 
            if isinstance(true_labels,Tensor):
                true_labels= true_labels.data 

            #soften distributions with temperature 
            student_soft = self._softmax(student_logits / self.temperature)
            teacher_soft = self._softmax(teacher_logits / self.temperature)

            #soft target loss(KL divergence)
            # KL divergence must be computed over probability distributions.
            soft_loss = self._kl_divergence(student_soft, teacher_soft)

            #hard target loss (cross-entropy)
            student_hard = self._softmax(student_logits)
            hard_loss =self._cross_entropy(student_hard,true_labels)

            #combined loss
            total_loss = self.alpha * soft_loss + (1-self.alpha) * hard_loss

            return total_loss 

    def _softmax(self,logits):
        """
        computes softmax with numerical stability
        """
        exp_logits = np.exp(logits - np.max(logits,axis=-1,keepdims=True))
        return exp_logits / np.sum(exp_logits,axis=-1,keepdims=True)


    def _kl_divergence(self,p,q):
        """
        Computes KL divergence between distributions.
        """
        return np.sum(p*np.log(p/(q+1e-8) + 1e-8))

    def _cross_entropy(self,predictions,labels):
        """
        Computes cross-entropy loss.
        """
        #simple implementation for integer labels
        if labels.ndim == 1:
            return -np.mean(np.log(predictions[np.arange(len(labels)),labels]+ 1e-8))
        else:
            return -np.mean(np.sum(labels*np.log(predictions + 1e-8),axis=1))
