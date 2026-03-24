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

