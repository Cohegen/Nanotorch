from tensor import Tensor
from tensor_creation import tensor_creation
from arithmetic_operations import testing_arithmetic_operations
from shape_manipulation import testing_shape_manipulation
from matrix_multiplication import testing_matrix_multiplication
from reduction_operation import testing_reduction_operations

import numpy as np

def test_module():
    """
    Testing the whole module by attempting a Complete intergration
    
    
    """
    print("  RUNNING MODULE INTERGRATION TEST")
    print("="*50)

    #running all unit tests
    print("Running unit tests...")
    tensor_creation()
    testing_arithmetic_operations()
    testing_matrix_multiplication()
    testing_shape_manipulation()
    testing_reduction_operations()

    print("Running integration scenarios...")

    #testing realistic neural network computation
    print(" Testing a Two-Layer Neural Network...")

    #creating a tensor of shape(2 samples, 3 features)
    x = Tensor([[1,2,3],[4,5,6]])

    #first layer : 3 inputs -> 4 hidden units
    w_1 = Tensor([
        [0.1,0.2,0.3,0.4],
        [0.5,0.6,0.7,0.8],
        [0.9,1.0,1.1,1.2]
    ])
    b1 = Tensor([0.1,0.2,0.3,0.4])

    #forward pass
    hidden = x.matmul(w_1)+b1
    assert hidden.shape == (2,4), f"Expected (2,4), got {hidden.shape}"

    #second layer: 4 hidden -> hiddenW2 + b2
    w_2 = Tensor([[0.1,0.2],[0.3,0.4],[0.5,0.6],[0.7,0.8]])
    b2 = Tensor([0.1,0.2])

    #output layer: output = hiddenw2 + b2
    output = hidden.matmul(w_2)+b2
    assert output.shape == (2,2), f"Expected (2,2), got {output.shape}"

    #verifying data flows correctly 
    assert not np.isnan(output.data).any(), "Output contains NaN values"
    assert np.isfinite(output.data).all(), "Ouputs contans infinite values"

    print("Two-layered neural network computation works")

    #testing complex shape manipulation
    print(" Intergration Test: Complex Shape Operations...")
    data = Tensor([1,2,3,4,5,6,7,8,9,10,11,12])

    #reshape to 3D tensor i.e batch processing
    tensor_3d = data.reshape(2,2,3) #(batch=2,height=2,width=2)
    assert tensor_3d.shape == (2,2,3)

    #global average pooling simulation
    pooled = tensor_3d.mean(axis=(1,2))#average across spatial dimensions
    assert pooled.shape == (2,), f"Expected (2,), got {pooled.shape}"

    #flatten for MLP
    flattened = tensor_3d.reshape(2,-1) #(batch,features)
    assert flattened.shape == (2,6)

    #transpose for different operations
    transposed = tensor_3d.transpose() #should transpose last two dims
    assert transposed.shape == (2,3,2)

    print(" Complex shape operations work")


    #testing broadcasting edge cases
    print("Testing broadcasting operations")

    #scalar operations
    scalar =Tensor(5.0)
    vector = Tensor([1,2,3])
    result = scalar + vector
    expected = np.array([6,7,8],dtype=np.float32)
    assert np.array_equal(result.data,expected)

    #matrix + vector broadcasting
    matrix = Tensor([[1,2],[3,4]])
    vec = Tensor([10,20])
    result = matrix + vec
    expected = np.array([[11,22],[13,24]],dtype=np.float32)
    assert np.array_equal(result.data,expected)

    print("Broadcating works")

if __name__ == "__main__":
    test_module()

