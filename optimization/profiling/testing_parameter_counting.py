import os
import sys 
import numpy as np


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear

def testing_parameter_counting():

    """
    This function validates whether the count_parameters helper functions work correctly
    for different model types
    
    """

    profiler = Profiler()

    #testing simple model with known parameters
    class SimpleModel:
        def __init__(self):
            self.weight = Tensor(np.random.randn(10,5))
            self.bias = Tensor(np.random.randn(5))

        def parameters(self):
            return [self.weight,self.bias]

    simple_model = SimpleModel()
    param_count = profiler.count_parameters(simple_model)
    expected_count = 10*5 + 5 #weight + bias 
    assert param_count == expected_count,f"Expected {expected_count} parameters, got{param_count} "
    print(f"Simple model:{param_count} parameters")

    #testing model without parameters
    class NoParamModel:
        def __init__(self):
            pass 

    no_param_model = NoParamModel()
    param_count = profiler.count_parameters(no_param_model)
    assert param_count == 0,f"Expected 0 parameters, got {param_count}"
    print(f"No parameter model: {param_count} parameters")

    #testing direct tensor with no parameters 
    test_tensor = Tensor(np.random.randn(2,3))
    param_count = profiler.count_parameters(test_tensor)
    assert param_count == 0,f"Expected 0 parameters for tensor, got {param_count}"
    print(f"Direct tensor: {param_count} parameters")

    print("Parameter counting works correctly")

if __name__ == "__main__":
    testing_parameter_counting()