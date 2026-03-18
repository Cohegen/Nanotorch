import os
import sys 
import numpy as np


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear

def estimating_optimizer_memory():
    """
    This function intends to test the _estimate_optimizer_memory helper function 
    and whether it estimates memory requirements for different optimizers.

    """
    profiler= Profiler()

    #testing with 100mb gradient memory 
    estimates = profiler._estimate_optimizer_memory(gradient_memory_mb=100.0)

    assert estimates['sgd'] == 0, f"SGD should need 0 extra, got {estimates['sgd']}"
    assert estimates['adam'] == 200.0, f"Adam should need 200mb  got {estimates['adam']}"
    assert estimates['adamw'] == 200.0, f"AdamW should need 200 M, got {estimates['adamw']}"
    print(f"SGD: {estimates['sgd']} MB, Adam:{estimates['adam']} MB,AdamW: {estimates['adamw']} MB")

    #testing with zero gradients
    estimates_zero = profiler._estimate_optimizer_memory(gradient_memory_mb=0.0)
    assert estimates_zero['adam'] == 0.0, "Zero gradients -> zero optimizer memory"
    print("Zero gradient memory handled correctly")

    print("_estimate_optimizer_memory works correctly")

if __name__ == "__main__":
    estimating_optimizer_memory()