import os
import sys 
import numpy as np
import tracemalloc

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear

def testing_calculate_memory_efficiency():
    """
    This functions intends to test whether the _calculate_memory_effiency helper 
    works 
    Efficiency = useful_memory /peak_memory
    Analysing efficiency is crucial since low efficiency means that memory fragmentation or allocator overheard
    Values are expected to between 0 and 1
    """

    profiler = Profiler()

    #testing Perfect efficiency 
    eff = profiler._calculate_memory_efficiency(10.0,10.0)
    assert abs(eff-1.0) < 0.01, f"Expected 1.0, got {eff}"
    print(f"Perfect efficiency: {eff}")

    #testing Half efficiency 
    eff_half = profiler._calculate_memory_efficiency(5.0,10.0)
    assert abs(eff_half - 0.5) < 0.01, f"Expected 0.5, got {eff_half}"
    print(f"Half efficiency: {eff_half}")

    #testing whether eff is clamped at 1.0 (useful > peak shouldn't exceed 1.0)
    eff_clamped = profiler._calculate_memory_efficiency(20.0,10.0)
    assert eff_clamped <= 1.0, f"Efficiency should be clamped to 1.0, got {eff_clamped}"
    print(f"Clamped efficiecny:{eff_clamped}")

    #testing Division by zero safety
    eff_zero = profiler._calculate_memory_efficiency(5.0,0.0)
    assert eff_zero <= 1.0, f"Should handle zero peak safely, got {eff_zero}"
    print("Zero peak safety handled")

    print("_calculate_memory_efficiency works correctly ")

if __name__ == "__main__":
    testing_calculate_memory_efficiency()
