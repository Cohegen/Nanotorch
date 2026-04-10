import os 
import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from compression import measure_sparsity,magnitude_prune
import numpy as np

def testing_magnitude_prune():
    """
    This function intends to test whether the magnitude_prune
    function works
    """
    #creating testing model with explicit composition
    layer1 = Linear(4,3)
    layer2 = Linear(3,2)
    model = Sequential(layer1,layer2)

    layer1.weight.data = np.array([
        [1.0, 2.0, 3.0],    # Large weights - should survive pruning
        [0.1, 0.2, 0.3],    # Medium weights
        [4.0, 5.0, 6.0],    # Large weights - should survive pruning
        [0.01, 0.02, 0.03]  # Tiny weights - will be pruned
    ])

    initial_sparsity = measure_sparsity(model)
    assert initial_sparsity < 1.0, "Model should start with minimal sparsity (<1%)"

    #applying 50% pruning -removes smallest 50% of weights
    magnitude_prune(model,sparsity=0.5)
    final_sparsity = measure_sparsity(model)

    #should achieve approximately 50% sparsity
    assert 40 <= final_sparsity <= 60, f"Expected ~50% sparsity, got {final_sparsity}%"

    #verifying if largest weight survived
    remaining_weights =layer1.weight.data[layer1.weight.data !=0]
    assert len(remaining_weights) > 0, "Some weights should remain"
    assert np.all(np.abs(remaining_weights) >= 0.1), "Large weights should survive"

    print("magnitude_prune works")

if __name__ == "__main__":
    testing_magnitude_prune()