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
            elif bytes <1024*1024:
                return f"{bytes/1024:1f}KB"
            else:
                return f"{bytes/(1024*1024):.1f}MB"

        print(f"{name:<10} {format_memory(param_memory):<10} "
              f"{format_memory(grad_memory):<12} {format_memory(sgd_memory):<12} "
              f"{format_memory(adam_memory):<12} {format_memory(total_memory):<10}")

def analyze_checkpoint_overhead():
    """Analyzing checkpoint size and overhead."""

    #creating a simple model
    class NanoModel:
        def __init__(self,size):
            self.layer = Linear(size,size)
            self.training = True

        def forward(self,x):
            return self.layer.forward(x)
        
        def parameters(self):
            return self.layer.parameters()

    sizes = [10,50,100]
    
    print("\nCheckpoint Size Analysis:")
    print("=" * 70)
    print(f"{'Model Size':<12} {'Raw Params':<15} {'Checkpoint':<15} {'Overhead':<10}")
    print("-" * 70)

    for size in sizes:
        #creating model and trainer 
        model = NanoModel(size)
        optimizer = SGD(model.parameters(),lr=0.01)
        trainer = Trainer(model,optimizer,MSELoss())

        #estimating raw parameter size
        param_count = size*size + size 
        raw_size = param_count * 4 # 4bytes per float32

        #creating checkpoint and measuring size
        checkpoint_path = f"/tmp/checkpoint_test_{size}.pkl"
        trainer.save_checkpoint(checkpoint_path)

        checkpoint_size = os.path.getsize(checkpoint_path)
        overhead = (checkpoint_size/raw_size - 1)*100

        os.remove(checkpoint_path)

        def format_size(bytes):
            if bytes < 1024:
                return f"{bytes}B"
            return f"{bytes/1024:.1f}KB"

        print(f"{size}×{size:<8} {format_size(raw_size):<15} "
              f"{format_size(checkpoint_size):<15} {overhead:.1f}%")

if __name__ == "__main__":
    training_memory_usage()
    analyze_checkpoint_overhead()
                