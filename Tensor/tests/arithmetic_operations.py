from tensor import Tensor
import numpy as np


def testing_arithmetic_operations():
    print(f"Testing arithmetic operation")

    a = Tensor([1,2,3])
    b = Tensor([4,5,6])

    result = a + b
    assert np.array_equal(result.data,np.array([5,7,9],dtype=np.float32))

    #testing broadcasting with different shapes (matrix+vector)
    matrix = Tensor([[1,2],[3,4]])
    vector = Tensor([10,20])

    result = matrix + vector
    assert np.array_equal(result.data,np.array([[11,22],[13,24]],dtype=np.float32))

    ##testing subtraction
    result = b-a
    assert np.array_equal(result.data,np.array([3,3,3],dtype=np.float32))

    #testing multiplication
    result = a*2
    assert np.array_equal(result.data,np.array([2,4,6],dtype=np.float32))

    #testing division
    result = b/2
    assert np.array_equal(result.data,np.array([2.0,2.5,3.0],dtype=np.float32))

    #testing chaining operations
    normalized = (a-2) / 2
    expected = np.array([-0.5,0.0,0.5],dtype=np.float32)
    assert np.allclose(normalized.data,expected)

    print("Arithmetic operations work correctly!")

if __name__ == "__main__":
    testing_arithmetic_operations()
