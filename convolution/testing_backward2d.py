import os
import sys

from convolutions import BatchNorm2d
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from Tensor import Tensor

def testing_batchNorm2d():
    """
    This function validates batch normalization implementation

    """

    #Test 1: Basic forward pass shape
    print("Testing basic forward pass...")
    bn = BatchNorm2d(num_features=16)
    x = Tensor(np.random.randn(4,16,8,8))
    y = bn(x)

    assert y.shape == x.shape,f"Output shape should match input, got {y.shape}"

    #Test2: Training mode normalization
    print("Testing training mode normallization...")
    bn2 = BatchNorm2d(num_features=8)
    bn2.train() #ensuring training mode

    #create input with known statistics per channel
    x2 = Tensor(np.random.randn(32,8,4,4)*10 + 5) #mean~5, std~10
    y2 = bn2(x2)

    #After normalization, each channel should have mean=0,std=1
    #before gamma/beta are applied, since gamma=1,beta=0
    for c in range(8):
        channel_mean = np.mean(y2.data[:,c:,:])
        channel_std = np.std(y2.data[:,c,:,:])
        assert abs(channel_mean) < 0.1,f"Channel {c} mean should ~0, got {channel_mean:.3f}"
        assert abs(channel_std -1.0) < 0.1,f"Channel {c} std should be ~1, got {channel_std:.3f}"

    #Testing Running statistics update
    print("  Testing running statistics update...")
    intial_running_mean = bn2.running_mean.copy()

    #Forward pass updates running stats
    x3 = Tensor(np.random.randn(16,8,4,4)+3) #offset mean
    _ = bn2(x3)

    #Running mean should have moved toward batch mean
    assert not np.allclose(bn2.running_mean,intial_running_mean), \
        "Running mean should update during training"

    #Test4:Eval mode uses running statistics
    print("  Testing eval mode behavior")
    bn3 = BatchNorm2d(num_features=4)

    #train on some data to establish running stats
    for _ in range(10):
        x_train = Tensor(np.random.randn(8,4,4,4)* 2 + 1)
        _ = bn3(x_train)

    saved_running_mean = bn3.running_mean.copy()
    saved_running_var = bn3.running_var.copy()

    #Switching to eval mode
    bn3.eval()

    #processing different data 
    x_eval = Tensor(np.random.randn(2,4,4,4)*5) #different distribution
    _ = bn3(x_eval)

    assert np.allclose(bn3.running_mean,saved_running_mean),\
        """
        Running mean should not change in eval mode
        """
    assert np.allclose(bn3.running_var,saved_running_var),\
        "Running var should not change in eval mode"
    
    #Test 5:Parameter counting
    print(" Testing parameter counting...")
    bn4 = BatchNorm2d(num_features=64)
    params = bn4.parameters()

    assert len(params) == 2, f"Should have 2 parameters (gamma,beta), got{len(params)}"
    assert params[0].shape == (64,),f"Gamma shape should be (64,), got {params[0].shape}"
    assert params[1].shape == (64,),f"Beta shape be (64,), got {params[1].shape}"

    print("BatchNorm2d works correctly!")

if __name__ == "__main__":
    testing_batchNorm2d()

    