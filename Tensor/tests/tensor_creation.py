from tensor import Tensor
import numpy as np

##testing 
def  tensor_creation():
    """Testing tensor creating with various data types."""

    #testing scalar scalar creation
    scalar = Tensor(5.0)
    assert scalar.data == 5.0
    assert scalar.shape == ()
    assert scalar.size == 1 
    assert scalar.dtype == np.float32

    #testing vector creation

    vector = Tensor([1,2,3])
    assert np.array_equal(vector.data,np.array([1,2,3],dtype=np.float32))
    assert vector.shape == (3,)
    assert vector.size == 3


    ##testing matrix creation
    matrix = Tensor([[1,2],[3,5]])
    assert np.array_equal(matrix.data,np.array([[1,2],[3,5]],dtype=np.float32))
    assert matrix.shape == (2,2)
    assert matrix.size == 4

    #test 3D tensor creation
    tensor_3d = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert tensor_3d.shape == (2,2,2)
    assert tensor_3d.size == 8

    print("Tensor creation works correctly")

if __name__ == "__main__":
    tensor_creation()




