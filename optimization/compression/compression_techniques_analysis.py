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

def analyze_compression_techniques():
    """
    This function offers analysis of compression ratios across different techniques 
    """
    print("\nAnalyzing Compression Techniques")
    print("="*60)

    #creating baseline model
    model_configs = [
         ("Small MLP", [Linear(128, 64), Linear(64, 32)]),
        ("Medium MLP", [Linear(512, 256), Linear(256, 128)]),
        ("Large MLP", [Linear(1024, 512), Linear(512, 256)])
    ]

    print(f"\n{'Model':<15} {'Technique':<20} {'Sparsity':<12} {'Compression':<12}")
    print("-" * 60)

    for model_name,layers in model_configs:
        #creating model with explicit composition
        model = Sequential(*layers)
        baseline_params = sum(p.size for p in model.parameters())

        #testing magnitude pruning on copy of model
        #creating fresh layer for magnitude pruning
        mag_layers = [Linear(l.weight.shape[0], l.weight.shape[1]) for l in layers]
        for i, layer in enumerate(mag_layers):
            layer.weight = layers[i].weight
            
            layer.bias = layers[i].bias
        mag_model = Sequential(*mag_layers)
        magnitude_prune(mag_model, sparsity=0.8)
        mag_sparsity = measure_sparsity(mag_model)
        mag_ratio = 1.0 / (1.0 - mag_sparsity / 100) if mag_sparsity < 100 else float('inf')

        print(f"{model_name:<15} {'Magnitude (80%)':<20} {mag_sparsity:>10.1f}% {mag_ratio:>10.1f}x")

        #testing structured pruning on seperate copy
        #creating fresh layer for structured pruning test
        struct_layers = [Linear(l.weight.shape[0],l.weight.shape[1]) for l in layers ]
        for i,layer in enumerate(struct_layers):
            layer.weight = layers[i].weight
            layer.bias = layers[i].bias 
        struct_model = Sequential(*struct_layers)
        structured_prune(struct_model, prune_ratio=0.5)
        struct_sparsity = measure_sparsity(struct_model)
        struct_ratio = 1.0 / (1.0 - struct_sparsity / 100) if struct_sparsity < 100 else float('inf')


        print(f"{'':<15} {'Structured (50%)':<20} {struct_sparsity:>10.1f}% {struct_ratio:>10.1f}x")
        print()

if __name__ == "__main__":
    analyze_compression_techniques()