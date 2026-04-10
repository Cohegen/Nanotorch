import os 
import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from compression import measure_sparsity,structured_prune
import numpy as np

def testng_structured_prune():
    """
    This function intends to test structured_prune function and
    to find out whether it works correctly
    """

    #creating test model with explicit layers
    layer1 = Linear(4,6)
    layer2 = Linear(6,2)
    model = Sequential(layer1,layer2)

    #setting predictable weights for testing
    layer1.weight.data = np.array([
        [1.0, 0.1, 2.0, 0.05, 3.0, 0.01],  # Channels with varying importance
        [1.1, 0.11, 2.1, 0.06, 3.1, 0.02],  # Large values in columns 0,2,4
        [1.2, 0.12, 2.2, 0.07, 3.2, 0.03],  # Small values in columns 1,3,5
        [1.3, 0.13, 2.3, 0.08, 3.3, 0.04]   # Pruning removes small channels
    ])

    initial_sparsity = measure_sparsity(model)
    assert initial_sparsity <1.0 ,"Model should start with minimal sparsity (<1%)"

    #Applying 33% structured pruning (2 out of 6 channels)
    #This removes entire channels, not scattered weights
    structured_prune(model, prune_ratio=0.33)
    final_sparsity = measure_sparsity(model)

    #checking that some channels are completely zero
    weight = layer1.weight.data
    zero_channels = np.sum(np.all(weight == 0, axis=0))
    assert zero_channels >= 1, f"Expected at least 1 zero channel, got {zero_channels}"

    #checking that non-zero channels are completely preserved
    # this is structured pruning whereby entire channels are zero or non-zero
    for col in range(weight.shape[1]):
        channel = weight[:,col]
        assert np.all(channel==0) or np.all(channel !=0), "Channels should be fully zero or fully non-zero"

    print("structured_prune works correctly")

if __name__ == "__main__":
    testng_structured_prune()