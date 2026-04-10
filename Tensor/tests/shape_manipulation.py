"""Python script that attempts shape manipulation of tensors"""

from tensor import Tensor
import numpy as np

def testing_shape_manipulation():
    print("Testing shape manipulation.")

    tensor = Tensor([1,2,3,4,5,6])
    reshaped = tensor.reshape(2,3)
    assert reshaped.shape == (2,3)
    expected = np.array([[1,2,3],[4,5,6]],dtype=np.float32)
    assert np.array_equal(reshaped.data,expected)

    #testing with tuple
    reshaped2 = tensor.reshape((3,2)) #shape : (3,2)
    assert reshaped2.shape == (3,2)
    expected2 = np.array([[1,2],[3,4],[5,6]],dtype=np.float32)
    assert np.array_equal(reshaped2.data,expected2)

    #test reshape validation - should raise error for incompatible size
    try:
        tensor.reshape(2,2)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "Total elements must match" in str(e)
        #assert "6 is not equal to 4" in str(e)

    # test matrix transpose 
    matrix = Tensor([[1,2,3],[4,5,6]]) #shape(2,3)
    transposed = matrix.transpose()
    assert transposed.shape == (3,2)
    expected = np.array([[1,4],[2,5],[3,6]],dtype=np.float32)
    assert np.array_equal(transposed.data,expected)

    #testing 1 dimensional tranpose
    vector = Tensor([1, 2, 3])
    vector_t = vector.transpose()
    assert np.array_equal(vector.data, vector_t.data)


    #test specific dimension transpose
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
    swapped = tensor_3d.transpose(0, 2)  # Swap first and last dimensions
    assert swapped.shape == (2, 2, 2)  # Same shape but data rearranged

    #test neural network reshape pattern (flatten for MLP)
    batch_images = Tensor(np.random.rand(2,3,4)) #(batch=2,height=3,width=4)
    flattened = batch_images.reshape(2,-1) #(batch=2,features=12)
    assert flattened.shape == (2,12)

    print("Shape manipulation works correctly!")

if __name__ == "__main__":
    testing_shape_manipulation()

