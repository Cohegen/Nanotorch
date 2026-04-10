from multiprocessing import Value
import os
import sys
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
from activations.activations import ReLU, Softmax
from layers.layers import XAVIER_SCALE_FACTOR, Layer,Linear,Dropout

def analyze_memory():
    """Analyzing memory usage patterns in layer operations"""
    print("Analyzing layer memory usage...")

    #testing different layer sizes
    layer_configs = [
        (784,256),
        (256,256),
        (256,10),
        (2048,2048)
    ]
    
    print("\nLinear Layer Memory Analysis:")
    print("Configuration-> Weight Memory-> Bias Memory -> Total Memory")

    for in_feat, out_feat in layer_configs:
        #calculating memory usage
        weight_memory = in_feat*out_feat*4 #4 bytes per float32
        bias_memory = out_feat*4
        total_memory = weight_memory + bias_memory

        print(f"({in_feat:4d}, {out_feat:4d}) -> {weight_memory/1024:7.1f} KB ->{bias_memory/1024:6.1f} KB -> {total_memory/1024:7.1f} KB")

    print("\nMulti-layer Memory Scaling: ")
    hidden_sizes = [128,256,512,1024,2048]

    for hidden_size in hidden_sizes:
        # 3-layer MLP -> hidden -> hidden / 2 -> 10
        layer1_params = 784*hidden_size + hidden_size
        layer2_params = hidden_size*(hidden_size // 2) + (hidden_size // 2)
        layer3_params = (hidden_size // 2) * 10 + 10

        total_params = layer1_params + layer2_params + layer3_params
        memory_mb = total_params * 4 / (1024 * 1024)

        print(f"Hidden={hidden_size:4d}:, total_parameters={total_params:7,}, params={memory_mb:5.1f} MB")


if __name__ == "__main__":
    analyze_memory()