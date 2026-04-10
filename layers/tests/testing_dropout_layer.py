from multiprocessing import Value
import os
import sys
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
from activations.activations import ReLU, Softmax
from layers.layers import XAVIER_SCALE_FACTOR, Layer,Linear,Dropout

def testing_dropout_layer():
    """This function tests the dropout layer"""
    print("Testing the dropout layer")

    dropout = Dropout(0.5)
    assert dropout.p == 0.5

    #testing inference mode
    x = Tensor([1,2,3,4])
    y_inference = dropout.forward(x,training=False)
    assert np.array_equal(x.data,y_inference.data),"Inference should pass through unchanged"

    #testing training mode with full dropout
    dropout_full = Dropout(1.0)
    y_full = dropout_full.forward(x,training=True)
    assert np.allclose(y_full.data,0),"Full dropout should zero everything"

    #testing training mode with partial dropout
    np.random.seed(42)
    x_large = Tensor(np.ones(1000,)) #creating a large tensor for statistical significance
    y_train = dropout.forward(x_large,training=True)

    #counting non-zero elements i.e approximately 50% should survive
    non_zero_count = np.count_nonzero(y_train.data)
    expected = 500

    #using 3-sigma bounds: std = sqrt(n*p*(1-p)) = sqrt(1000*0.5*0.5) = 15.8
    std_error = np.sqrt(1000*0.5*0.5)
    lower_bound = expected - 3 *std_error
    upper_bound = expected + 3 *std_error
    assert lower_bound < non_zero_count < upper_bound, \
        f"Expected{expected}+_{3*std_error:.0f} survivors, got {non_zero_count}"

    #testing scaling (surviving elements should be scaled by 1/(1-p)=2.0)
    surviving_values = y_train.data[y_train.data != 0]
    expected_value = 2.0 #1.0/ (1-0.5)
    assert np.allclose(surviving_values,expected_value), f"Surviving values should be {expected_value}"

    #testing no parameters
    params = dropout.parameters()
    assert len(params) == 0 , "Dropout should have no parameters"

    #testing invalid probability
    try:
        Dropout(-0.1)
        assert False, "Should raise ValueError for negative probability"
    except ValueError:
        pass

    try:
        Dropout(1.1)
        assert False, "Should raise ValueError for probability > 1"
    except ValueError:
        pass

    print("Dropout layer works correctly")

if __name__ == "__main__":
    testing_dropout_layer()
