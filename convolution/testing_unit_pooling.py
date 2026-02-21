import os 
#from select import EPOLLEXCLUSIVE
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
from convolutions import AvgPool2d, MaxPool2d
import numpy as np

def testing_pooling():
    """
    This test validates both max and average pooling
    implementation.
    """

    #Test 1 #MaxPool2d and AvgPool2d 
    print("  Testing MaxPoll2d...")
    maxpool = MaxPool2d(kernel_size=2,stride=2)
    x1 = Tensor(np.random.randn(1,3,8,8))
    out1 = maxpool(x1)

    expected_shape = (1,3,4,4) # 8/2 = 4
    assert out1.shape == expected_shape,f"MaxPool expected {expected_shape}, got {out1.shape}"

    #Test 2:AvgPool2d basic functionality
    print("Testing AvgPool2d")
    avgpool = AvgPool2d(kernel_size=2,stride=2)
    x2 = Tensor(np.random.randn(2,16,16,16))
    out2 = avgpool(x2)

    expected_shape = (2,16,8,8) #16/2 =8
    assert out2.shape == expected_shape,f"AvgPool expected {expected_shape}, got {out2.shape}"

    #Test3: MaxPool vs AvgPool on known data 
    print("  Testing max vs avg behavior...")
    #create simple test case with known values
    test_data = np.array([[[[1, 2, 3, 4],
                           [5, 6, 7, 8],
                           [9, 10, 11, 12],
                           [13, 14, 15, 16]]]], dtype=np.float32)

    x3 = Tensor(test_data)

    maxpool_test = MaxPool2d(kernel_size=2,stride=2)
    avgpool_test = AvgPool2d(kernel_size=2,stride=2)

    max_out = maxpool_test(x3)
    avg_out = avgpool_test(x3)

    # For 2x2 windows:
    # Top-left: max([1,2,5,6]) = 6, avg = 3.5
    # Top-right: max([3,4,7,8]) = 8, avg = 5.5
    # Bottom-left: max([9,10,13,14]) = 14, avg = 11.5
    # Bottom-right: max([11,12,15,16]) = 16, avg = 13.5

    expected_max = np.array([[[[6, 8], [14, 16]]]])
    expected_avg = np.array([[[[3.5, 5.5], [11.5, 13.5]]]])

    assert np.allclose(max_out.data,expected_max),f"MaxPool values incorrect:{max_out.data} vs {expected_max}"
    assert np.allclose(avg_out.data,expected_avg),f"AvgPool values incorrect:{avg_out.data} vs {expected_avg}"


    #Testing overlapping pooling (stride < kernel_size)
    print("  Testing overlapping pooling...")
    overlap_pool = MaxPool2d(kernel_size=3,stride=1)
    x4 = Tensor(np.random.randn(1,1,5,5))
    out4 = overlap_pool(x4)

    #Output : (5-3) / 1 + 1 = 3
    expected_shape = (1, 1, 3, 3)
    assert out4.shape == expected_shape, f"Overlapping pool expected {expected_shape}, got {out4.shape}"

    # Testing No parameters in pooling layers
    print("  Testing parameter counts...")
    assert len(maxpool.parameters()) == 0, "MaxPool should have no parameters"
    assert len(avgpool.parameters()) == 0, "AvgPool should have no parameters"

    print(" Pooling operations work correctly!")


if __name__ == "__main__":
    testing_pooling()  