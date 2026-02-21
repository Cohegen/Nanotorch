from ntpath import abspath
import os
from os.path import dirname
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from convolutions import Conv2d
from Tensor import Tensor
import numpy as np
import time

def analyze_convolution_complexity():
    """
    This function is intended to understand the concept of computational complexity
    and memory trade-off  in spatial operations

    This functions gives a glimpse of why certain desing choices matter for real-world performance,
    and why modern CNNs use specific architectural patterns.

    """

    #testing configurations optimized for educational demonstration
    configs = [
        {"input": (1, 3, 16, 16), "conv": (8, 3, 3), "name": "Small (16×16)"},
        {"input": (1, 3, 24, 24), "conv": (12, 3, 3), "name": "Medium (24×24)"},
        {"input": (1, 3, 32, 32), "conv": (16, 3, 3), "name": "Large (32×32)"},
        {"input": (1, 3, 16, 16), "conv": (8, 3, 5), "name": "Large Kernel (5×5)"},
    ]

    print(f"{'Configuration':<20} {'FLOPs':<15} {'Memory (MB)':<12} {'Time (ms)':<10}")
    print("-" * 70)

    for config in configs:
        #Create convolution layer
        in_ch = config["input"][1]
        out_ch,k_size = config["conv"][0],config["conv"][1]
        conv = Conv2d(in_ch,out_ch,kernel_size=k_size,padding=k_size//2)

        #creating input tensor
        x= Tensor(np.random.randn(*config["input"]))

        #Calculating theoretical FLOPs
        batch,in_channels,h,w = config["input"]
        out_channels,kernel_size = config["conv"][0],config["conv"][1]

        #each output element requires in_channels * kernel_size**2 multiply-adds
        flops_per_output = in_channels*kernel_size*kernel_size*2 # 2 for MAC 
        total_outputs = batch * out_channels*h*w # assuming same size with padding 
        total_flops = flops_per_output * total_outputs

        #measuring memory usage
        input_memory = np.prod(config["input"]) * 4  # float32 = 4 bytes
        weight_memory = out_channels * in_channels * kernel_size * kernel_size * 4
        output_memory = batch * out_channels * h * w * 4
        total_memory = (input_memory + weight_memory + output_memory) / (1024 * 1024)  # MB

        #Measuring execution time
        start_time = time.time()
        _ = conv(x)
        end_time = time.time()
        exec_time = (end_time - start_time) * 1000 #ms

        print(f"{config['name']:<20} {total_flops:<15,} {total_memory:<12.2f} {exec_time:<10.2f}")
print("\n💡 Key Insights:")
print(" FLOPs scale as O(H×W×C_in×C_out×K²) - quadratic in spatial and kernel size")
print(" Memory scales linearly with spatial dimensions and channels")
print(" Large kernels dramatically increase computational cost")
print(" This motivates more efficient convolution variants that reduce computational cost")

if __name__ == "__main__":
    analyze_convolution_complexity()
