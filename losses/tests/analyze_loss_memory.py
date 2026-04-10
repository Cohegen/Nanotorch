import os
import sys
from unittest import result
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses import  log_softmax,CrossEntropyLoss,EPILSON,BinaryCrossEntropyLoss,MSELoss
from Tensor import Tensor
from activations.activations import ReLU
from layers.layers import Linear
import numpy as np

def analyze_loss_memory():
    """Analyzing memory usage patterns of different loss functions."""

    print("\n Analysis: Loss function Memory Usage....")

    batch_sizes = [32,128,512,1024]
    num_classes = 1000 #like in ImageNet

    print("\nMemory Usage by Batch Size:")
    print(f"Batch size | MSE(MB) |CrossEntropy (MB) | BCE(MB) | Notes")
    print("-" * 75)

    for batch_size in batch_sizes:
        #memory calculations
        bytes_per_float = 4

        #mse: predictions + targets 
        mse_elements = batch_size * 1 #regression usually has 1 output
        mse_memory = mse_elements *bytes_per_float * 2 / 1e6 #converting to MB

        #crossentropy: logits + targets + softmax + log_softmax
        ce_logits = batch_size * num_classes
        ce_targets = batch_size * 1 #target indices
        ce_softmax = batch_size *num_classes #intermediate softmax
        ce_total_elements = ce_logits + ce_targets + ce_softmax
        ce_memory = ce_total_elements * bytes_per_float / 1e6

        #BCE: predictions + targets (binary, so smaller)
        bce_elements = batch_size * 1
        bce_memory = bce_elements * bytes_per_float * 2 / 1e6

        notes = "Linear scaling" if batch_size == 32 else f"{batch_size//32}x fast"
        print(f"{batch_size:10} | {mse_memory:8.2f} | {ce_memory:13.2f} | {bce_memory:7.2f} | {notes}")
        
    print(f"\nMemory Insights:")
    print("   - CrossEntropy dominates due to large vocabulary (num_classes)")
    print("   - Memory scales linearly with batch size")
    print("   - Intermediate activations (softmax) double CE memory")
    print(f"   - For batch=1024, CE needs {ce_memory:.1f}MB just for loss computation")

    

if __name__ == "__main__":
    analyze_loss_memory()