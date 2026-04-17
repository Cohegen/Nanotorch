import numpy as np
import os 
import sys


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Tensor import Tensor
from activations.activations import ELU, GELU, LeakyReLU, Mish, PReLU, ReLU, SiLU, Sigmoid, Softmax, SwiGLU, Tanh, TOLERANCE
from activations.tests.testing_extended_activations import testing_extended_activations
from activations.tests.testing_gelu import testing_gelu
from activations.tests.testing_relu import testing_relu
from activations.tests.testing_sigmoid import testing_sigmoid
from activations.tests.testing_softmax import testing_softmax
from activations.tests.testing_tanh import testing_tanh
##complete module test
def testing_module():
    print("Running module intergration test")
    print("=" * 50)

    #Run all unit tests
    print("Running unit tests...")
    testing_sigmoid()
    testing_relu()
    testing_tanh()
    testing_gelu()
    testing_softmax()
    testing_extended_activations()

    print("\n Running intergration scenarios")

    #1. All activations preserve tensor properties
    print("Intergration test: Tensor property preservation...")
    test_data = Tensor([[1,-1],[2,-2]])

    activations = [Sigmoid(),ReLU(),LeakyReLU(),SiLU(),Mish(),PReLU(),ELU(),Tanh(),GELU()]
    for activation in activations:
        result = activation.forward(test_data)
        assert result.shape == test_data.shape, f"Shape not preserved by{activation.__class__.__name}"
        assert isinstance(result,Tensor), f"Output not a Tensor from {activation.__class__.__name__}"

    print("All activations prserve tensor properties!")


    #2. Softmax works with different dimensions
    print("Intergration Test: Softmac dimension handling")
    data_3d = Tensor([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    softmax = Softmax()

    #testing different dimensions
    result_last = softmax(data_3d,dim=-1)
    assert result_last.shape == (2,2,3), "Softmax should preserve shape"

    #check that last dimension sums to 1
    last_dim_sums = np.sum(result_last.data,axis=-1)
    assert np.allclose(last_dim_sums,1.0),"Last dimension should sum to 1"

    print("Softmax handles different dimensions correctly")

    # SwiGLU reduces the split dimension by half
    swiglu = SwiGLU()
    swiglu_input = Tensor([[1, 2, 3, 4]])
    swiglu_output = swiglu.forward(swiglu_input)
    assert swiglu_output.shape == (1, 2), "SwiGLU should halve the split dimension"

    #3. Testing 3 Activation chaining (simulating neural network)
    print("Intergration Test: Activation chaining...")
    
    x = Tensor([[-1,0,1,2]])
    
    #aplying ReLU (hidden layer activation)
    relu = ReLU()
    hidden = relu.forward(x)

    #apllying softmax
    softmax = Softmax()
    output = softmax.forward(hidden)

    #verify the chain
    assert hidden.data[0,0]== 0, "ReLU should zero negative input"
    assert np.allclose(np.sum(output.data),1.0), "Final output should be a probability distribution"

    print("\n" + "=" * 50)
    print(" ALL TESTS PASSED! Module ready for export.")

if __name__ == "__main__":
    testing_module()
 
