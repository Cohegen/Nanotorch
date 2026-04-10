from tensor import Tensor
import numpy as np

def testing_reduction_operations():
    """Test reduction operations."""
    print("Testing reduction operations")

    matrix = Tensor([[1,2,3],[4,5,6]])

    #test sum all elements
    total = matrix.sum()
    assert total.data == 21.0
    assert total.shape == ()

    #testing sum across columns (axis=0)
    cols_sum = matrix.sum(axis=0)
    expected_col = np.array([5,7,9],dtype=np.float32)
    assert np.array_equal(cols_sum.data,expected_col)
    assert cols_sum.shape == (3,)

    #testing sum across rows(axis=1)
    row_sum = matrix.sum(axis=1)
    expected_row = np.array([6,15],dtype=np.float32)
    assert np.array_equal(row_sum.data,expected_row)
    assert row_sum.shape == (2,)

    #mean across column(axis=0)
    col_mean = matrix.mean(axis=0)
    expected_mean = np.array([2.5,3.5,4.5],dtype=np.float32)
    assert np.allclose(col_mean.data,expected_mean)

    #testing max
    maximum = matrix.max()
    assert maximum.data == 6.0
    assert maximum.shape == ()

    #testing max along axis
    row_max = matrix.max(axis=1)
    expected_max = np.array([3,6],dtype=np.float32)
    assert np.array_equal(row_max.data,expected_max)

    #test keepdims
    sum_keepdims = matrix.sum(axis=1,keepdims=True)
    assert sum_keepdims.shape == (2,1) #maintaining 2D shape
    expected_keepdims = np.array([[6],[15]],dtype=np.float32)

    # Test 3D reduction (simulating global average pooling)
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # (2, 2, 2)
    spatial_mean = tensor_3d.mean(axis=(1, 2))  # Average across spatial dimensions
    assert spatial_mean.shape == (2,)  # One value per batch item

    print("Reduction operations work correctly!")

if __name__ == "__main__":
    testing_reduction_operations() 