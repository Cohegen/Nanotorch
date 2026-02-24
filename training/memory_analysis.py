import os
import sys

from numpy.polynomial import test


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from Tensor import Tensor
from training import clip_grad_norm,Trainer
from training import CosineSchedule
from layers.layers import Linear
from optimizers.optimizers import SGD
from losses.losses import MSELoss
from dataloader.dataloader import Dataloader

def training_memory_usage():
    print("A function that analyzes Memory Overhead")

    #creating models of different sizes
    model_sizes = [
        ("Small",100), #100 params
        ("Medium",1000), # 1k parameters
        ("Large",10000) #10k parameters
    ]

    print("\nTraining Memory Analysis:")
    print("="*90)
    print(f"{'Model':<10} {'Params':<10} {'Gradients':<12} {'SGD State':<12} {'Adam State':<12} {'Total':<10}")
    print("-" * 90)

    for name,param_count in model_sizes:
        #base memory:parameters
        param_memory = param_count * 4 # 4 bytes per float32

        #gradients:same as parameters
        grad_memory = param_count * 4

        #SGD optimizer state: mometum buffer
        sgd_memory = param_count * 4

        #Adam optimizer state:2 buffers (m and v)
        adam_memory = param_count * 2* 4

        #total with Adam 
        total_memory = param_memory + grad_memory + adam_memory

        #convert to human-readable
        def format_memory(bytes):
            if bytes <1024:
                return f"{bytes}B"
                
