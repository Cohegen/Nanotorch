import os
import sys 
import numpy as np


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear

def testing_helper_functions():
    """
    This program is intended to test the 
    helper functions defined in profiling.py
    """

    #testing quick profile function
    test_model = Linear(16,8)
    test_input = Tensor(np.random.randn(8,16))
    profile = quick_profile(test_model,test_input,profiler=Profiler())

    #validate profile contains expected keys 
    assert 'parameters' in profile, "Quick profile should include parameters"
    assert 'flops' in profile ," Quick should include FLOPs"
    assert 'latency_ms' in profile, "Quick profile should include latency"
    print("Quick profile provides comprehensive metrics")

    #testing weight distribution analysis
    class SimpleModel:
        def __init__(self):
            self.weight = Tensor(np.random.randn(10,5)*0.1)#small weights

    model = SimpleModel()
    stats =analyze_weight_distribution(model)

    #validating stats structure
    assert 'total_weights' in stats,"Should count total weights"
    assert 'mean' in stats,"Should compute mean"
    assert 'std' in stats, "Should compute standard deviation"
    assert stats['total_weights'] == 50,f"Expected 50 weights, got {stats['total_weights']}"
    print(f"Weight distribution analysis: {stats['total_weights']} weights analyzed")

    #testing weight distribution with no weights
    class NoWeightModel:
        pass 

    no_weight_model = NoWeightModel()
    stats = analyze_weight_distribution(no_weight_model)
    assert 'error' in stats, "Should handle models without weights"
    print("Handles models without weights gracefully")
    print("Helper functions work correctly!")

if __name__ == "__main__":
    testing_helper_functions()