import os 
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
import numpy as np
from transformers import LayerNorm 

def testing_layer_norm():
    """
    This function intends to test whether LayerNorm works properly
    """
    #test basic normalization
    layer_norm = LayerNorm(4)
    x = Tensor([[1.0,2.0,3.0,4.0],[5.0,6.0,7.0,8.0]]) #(2,4)

    normalized = layer_norm.forward(x)

    #check output shape 
    assert normalized.shape == (2,4)

    #checking normalization properties (approximately)
    #for each sample, mean should be close to 0 , std close to 1
    for i in range(2):
        sample_mean = np.mean(normalized.data[i])
        sample_std = np.std(normalized.data[i])
        assert abs(sample_mean) < 1e-5, f"Mean should be ~0 got {sample_mean}"
        assert abs(sample_std-1.0) < 1e-4, f"Std should be ~1, got {sample_std}"

    #Test parameter shapes
    params = layer_norm.parameters()
    assert len(params) == 2
    assert params[0].shape == (4,) #gamma
    assert params[1].shape == (4,) #beta 

    print("LayerNorm works correctly")

if __name__== "__main__":
    testing_layer_norm()


    