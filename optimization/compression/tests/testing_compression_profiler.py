import os 

import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from compression import measure_sparsity,magnitude_prune,structured_prune,low_rank_approximate
from optimization.profiling.profiling import Profiler
import numpy as np

def compression_with_profiler():
    """
    This function uses the Profiler described in the profiling module 
    to measure the actual parameter reduction from pruning.
    """
    print("Measuring Compression Impact with Profiler")
    print("="*70)

    profiler = Profiler()

    #creating a simple model
    model =Linear(512,256)
    model.name= "baseline_model"

    print("\nBefore: Dense Model")
    print("-"*70)

    #measuring baseline
    param_count_before = profiler.count_parameters(model)
    sparsity_before = measure_sparsity(model)
    input_shape = (32,512)
    memory_before = profiler.measure_memory(model,input_shape)

    print(f"   Parameters: {param_count_before:,}")
    print(f"   Sparsity: {sparsity_before:.1f}% (zeros)")
    print(f"   Memory: {memory_before['parameter_memory_mb']:.2f} MB")
    print(f"   Active parameters: {int(param_count_before * (1 - sparsity_before / 100)):,}")

    #measuring magnitude pruning
    target_sparsity = 0.7  # Removes 70% of parameters
    print(f"\n  Applying {target_sparsity*100:.0f}% Magnitude Pruning...")
    pruned_model = magnitude_prune(model, sparsity=target_sparsity)
    pruned_model.name = "pruned_model"

    print("\n AFTER: Pruned Model")
    print("-" * 70)

    #measuring after pruning
    param_count_after = profiler.count_parameters(pruned_model)
    sparsity_after = measure_sparsity(pruned_model)
    memory_after = profiler.measure_memory(pruned_model,input_shape)

    print(f"   Parameters: {param_count_after:,} (same, but many are zero)")
    print(f"   Sparsity: {sparsity_after:.1f}% (zeros)")
    print(f"   Memory: {memory_after['parameter_memory_mb']:.2f} MB (same storage)")
    print(f"   Active parameters: {int(param_count_after * (1 - sparsity_after / 100)):,}")

    print("\n COMPRESSION RESULTS")
    print("=" * 70)
    sparsity_gain = sparsity_after - sparsity_before
    active_before = int(param_count_before * (1 - sparsity_before / 100))
    active_after = int(param_count_after * (1 - sparsity_after / 100))
    reduction_ratio = active_before / active_after if active_after > 0 else 1
    params_removed = active_before - active_after

    print(f"   Sparsity increased: {sparsity_before:.1f}% → {sparsity_after:.1f}%")
    print(f"   Active params reduced: {active_before:,} → {active_after:,}")
    print(f"   Parameters removed: {params_removed:,} ({sparsity_gain:.1f}% of total)")
    print(f"   Compression ratio: {reduction_ratio:.1f}x fewer active parameters")

    print("\n Key Insight:")
    print(f"   Magnitude pruning removes {sparsity_gain:.0f}% of parameters")
    print(f"   With sparse storage formats, this means {reduction_ratio:.1f}x less memory!")
    print(f"   Critical for: edge devices, mobile apps, energy efficiency")
    print("\n This is the power of compression: remove what doesn't matter!")

if __name__ == "__main__":
    compression_with_profiler()