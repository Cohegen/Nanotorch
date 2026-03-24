import os 
import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from compression import measure_sparsity
import numpy as np

def testing_measure_sparsity():
    """
    This function intends to test whether
    the measure_sparsity function works
    """

    #testing with dense models 
    layer1 = Linear(4,3)
    layer2 = Linear(3,2)
    model = Sequential(layer1,layer2)

    initial_sparsity = measure_sparsity(model)
    assert initial_sparsity <1.0 ,f"Expected <1% sparsity (dense model), got {initial_sparsity}%"

    #testing with manually sparse models
    layer1.weight.data[0,0] =0 #zeroes out specific weight
    layer1.weight.data[1,1] = 0#zeros out another weight
    sparse_sparsity = measure_sparsity(model)
    assert sparse_sparsity > 0,  f"Expected >0% sparsity, got {sparse_sparsity}%"

    print("measure_sparsity works correctly")

if __name__ == "__main__":
    testing_measure_sparsity()