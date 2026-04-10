from activations import Sigmoid
from activations import TOLERANCE
import sys 
import os
## adding parent directory to path to allow importing Tensor module
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from  Tensor import Tensor
import numpy as np

def testing_sigmoid():

    sigmoid = Sigmoid()

    #testing basic use cases
    x = Tensor([0.0])
    result = sigmoid.forward(x)
    assert np.all(result.data > 0) and np.all(result.data < 1), "All sigmoid outputs should be in the range(0,1)"

    #test specific values
    x = Tensor([-1000,1000]) #extreme values
    result = sigmoid.forward(x)
    assert np.allclose(result.data[0],0,atol=TOLERANCE), "sigmoid(-inf) should approach zero"
    assert np.allclose(result.data[1],1,atol=TOLERANCE),"sigmoid(+inf) should approach 1"

    print("Sigmoid works correctly")

if __name__ == "__main__":
    testing_sigmoid()
