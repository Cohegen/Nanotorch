import numpy as np
import os 
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Tensor import Tensor
from activations.activations import Softmax, TOLERANCE


def testing_softmax():
    """Testing Softmax implementation."""
    print("Testing Softmax")

    softmax = Softmax()

    #test basic probability properties
    x = Tensor([1,2,3])
    result = softmax.forward(x)

    #should sum to 1
    assert np.allclose(np.sum(result.data),1.0),f"Softmax should sum to 1 got{np.sum(result.data)} instead."

    #all values should be postive
    assert np.all(result.data > 0), "All softmax values be postive"

    #all values should be less than one
    assert np.all(result.data < 1), "All softmax values should be less than 1"

    #largest input should get largest output
    max_input_idx = np.argmax(x.data)
    max_output_idx = np.argmax(result.data)

    #test numerical stability with large numbers
    x = Tensor([1000,1001,1002]) #would overflow without max subtraction
    result = softmax.forward(x)
    assert np.allclose(np.sum(result.data),1.0), "Sotmax should handle large numbers."
    assert not np.any(np.isnan(result.data)), "Softmax should not be Nan"
    assert not np.any(np.isinf(result.data)), "Softmax should not be infinite"

    #testing with 2D tensor (batch dimension)
    x = Tensor([[1,2],[3,4]])
    result = softmax.forward(x,dim=-1) #softmax along last dimension
    assert result.shape == (2,2), "Softmax should preserve input shape"

    #each row should sum to 1
    row_sums = np.sum(result.data,axis=-1)
    assert np.allclose(row_sums,[1.0,1.0]),"Each row should sum to 1"

    print("Softmax works correctly")

if __name__ == "__main__":
    testing_softmax()
