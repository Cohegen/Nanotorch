import numpy as np
from tensor import Tensor

def testing_matrix_multiplication():
    """Testing matrix multiplication operations."""
    print("Testing matrix multiplication.")

    #testing 2x2 matrix multiplication
    a = Tensor([[1,2],[3,4]])
    b = Tensor([[5,6],[7,8]])
    result = a.matmul(b)

    expected = np.array([[19,22],[43,50]],dtype=np.float32)
    assert np.array_equal(result.data,expected)

    #testing rectangular matrices
    c = Tensor([[1,2,3],[4,5,6]]) #2x3
    d = Tensor([[7,8],[9,10],[11,12]]) #3x2
    result = c.matmul(d)

    expected2 = np.array([[58,64],[139,154]],dtype=np.float32)
    assert np.array_equal(result.data,expected2)

    #test matrix-vector multiplication
    matrix = Tensor([[1,2,3],[4,5,6]])
    vector = Tensor([1,2,3]) #shape (3,)
    result = matrix.matmul(vector)

    expected3 = np.array([14,32],dtype=np.float32)
    assert np.array_equal(result.data,expected3)

    #testing shape
    try:
        incompatible_a = Tensor([[1,2]])
        incompatible_b = Tensor([[1],[2],[3]]) #3x1
        incompatible_a.matmul(incompatible_b) #1x2 @ 3x1 should fail
        assert False,"Should have raised ValueError for incompatible shapes"
    except ValueError as e:
        assert "Inner dimensions must match" in str(e)
        assert "2 not equal to 3" in str(e) #should show specific dimensions

    print("Matrix multiplication works correctly!")

if __name__ == "__main__":
    testing_matrix_multiplication()