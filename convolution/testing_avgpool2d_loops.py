import os 
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(__file__)))
from Tensor import Tensor 
from convolutions import AvgPool2d,MaxPool2d
import numpy as np

def testing_avgpool2d_loops():
    pool = AvgPool2d(kernel_size=2,stride=2)

    #known 4x4 input 
    padded = np.array([[[[1.0, 2.0, 3.0, 4.0],
                          [5.0, 6.0, 7.0, 8.0],
                          [9.0, 10.0, 11.0, 12.0],
                          [13.0, 14.0, 15.0, 16.0]]]])

    output = pool._avgpool_loops(padded,batch_size=1,channels=1,out_h=2,out_w=2)

    # Window averages:
    # top-left: (1+2+5+6)/4 = 3.5
    # top-right: (3+4+7+8)/4 = 5.5
    # bottom-left: (9+10+13+14)/4 = 11.5
    # bottom-right: (11+12+15+16)/4 = 13.5
    expected = np.array([[[[3.5, 5.5], [11.5, 13.5]]]])
    assert np.allclose(output, expected), f"Expected:\n{expected}\nGot:\n{output}"

    # Test that avg is always <= max for same data
    pool_max = MaxPool2d(kernel_size=2, stride=2)
    max_output = pool_max._maxpool_loops(padded, 1, 1, 2, 2)
    assert np.all(output <= max_output), "Average should always be <= maximum"

    print("AvgPool2d loops work correctly!")

if __name__ == "__main__":
    testing_avgpool2d_loops()

