from bz2 import compress
import os 
import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from compression import measure_sparsity,structured_prune,low_rank_approximate,compress_model
import numpy as np

def testing_model_compression():
    """
    This function intends to test whether the compress_model
    helper function works properly
    """
    #creating testing model with explicit layers
    layer1 = Linear(20,15)
    layer2 =Linear(15,10)
    layer3 = Linear(10,5)
    model = Sequential(layer1,layer2,layer3)

    #defining compression configuration
    config = {
        'magnitude_prune':0.7, #removing 70% of smallest weights
        'structured_prune':0.2,#removing 20% of least important channels
    }

    #applying compression pipeline
    stats = compress_model(model,config)

    #verifying statistics
    assert 'original_params' in stats , "Shouls track original parameter count"
    assert 'final_sparsity' in stats, "Should track final sparsity"
    assert 'applied_techniques' in stats, "Should track applied techniques"

    #verifying if compression was applied successfully
    assert stats['final_sparsity'] > stats['original_sparsity'],"Sparsity should increase "
    assert len(stats['applied_techniques']) == 2,"Should apply both techniques"

    #verifying model still has reasonable structure after compression
    remaining_params = sum(np.count_nonzero(p.data) for p in model.parameters())
    assert remaining_params >0,"Model should retain some parameters"

    print("compress_model works correctly")

if __name__ == "__main__":
    testing_model_compression()
