import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
from convolutions import BatchNorm2d
import numpy as np

def testing_batchnorm2d_get_stats():

    bn = BatchNorm2d(num_features=4)
    x = Tensor(np.random.randn(8,4,6,6))

    #training mode: should return batch stats and update running stats
    bn.train()
    running_mean_before = bn.running_mean.copy()
    mean,var = bn._get_stats(x)

    assert mean.shape == (4,),f"Expected per-channel mean shape (4,), got {mean.shape}"
    assert var.shape == (4,),f"Expected per-channel var shape (4,), got {var.shape}"
    assert not np.allclose(bn.running_mean,running_mean_before),  \
        "Running mean should be updated in training mode"

    #eval mode: should return running stats(frozen)
    bn.eval()
    running_mean_snapshot = bn.running_mean.copy()
    mean_eval , var_eval = bn._get_stats(x)

    assert np.allclose(mean_eval,running_mean_snapshot),\
        "Eval mode should return running mean"
    assert np.allclose(bn.running_mean,running_mean_snapshot),\
        "Running mean should not change in eval mode"

    print("BatchNorm2d._get_stats work correctly!")

if __name__ == "__main__":
    testing_batchnorm2d_get_stats()