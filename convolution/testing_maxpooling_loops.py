import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 
from convolutions import Conv2d,MaxPool2d
import numpy as np

def testing_maxpooling_loops():
    """
    This function is intended to test that `_maxpool_loops`
    in the MaxPool2d correctly finds the maximum value
    in each pooling window
    """

    pool = MaxPool2d(kernel_size=2,stride=2)

    #known 4x4 input
    padded = np.array([[[[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0],
                          [9.0, 10.0, 11.0, 12.0],
                          [13.0, 14.0, 15.0, 16.0]]]])

    output = pool._maxpool_loops(padded,batch_size=1,channels=1,out_h=2,out_w=2)

     # Window maxes:
    # top-left: max(1,2,5,6) = 6
    # top-right: max(3,4,7,8) = 8
    # bottom-left: max(9,10,13,14) = 14
    # bottom-right: max(11,12,15,16) = 16
    expected = np.array([[[[6.0, 8.0], [14.0, 16.0]]]])
    assert np.allclose(output, expected), f"Expected:\n{expected}\nGot:\n{output}"

    #testing with negative values
    padded_neg = np.array([[[[-5.0, -1.0],
                              [-3.0, -2.0]]]])

    pool_small = MaxPool2d(kernel_size=2,stride=2)
    output_neg = pool_small._maxpool_loops(padded_neg,1,1,1,1)
    assert output_neg[0,0,0,0] == -1.0,f"Max of negatives: expected -1.0, got {output_neg[0,0,0,0]}"

    print("MaxPool2d loops work correctly")

if __name__ == "__main__":
    testing_maxpooling_loops()