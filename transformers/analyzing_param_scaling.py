import os 
import sys




sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from Tensor import Tensor 
import numpy as np
from transformers import LayerNorm ,MLP, TransformerBlock,GPT

def analyzing_parameter_scaling():
    """
    This function analyzes how parameter count scales with model dimensions
    """

    #testing different model sizes
    configs = [
          {"name": "Tiny", "embed_dim": 64, "num_layers": 2, "num_heads": 4},
        {"name": "Small", "embed_dim": 128, "num_layers": 4, "num_heads": 8},
        {"name": "Medium", "embed_dim": 256, "num_layers": 8, "num_heads": 16},
        {"name": "Large", "embed_dim": 512, "num_layers": 12, "num_heads": 16},
    ]

    vocab_size = 50000 

    for config in configs:
        model = GPT(
            vocab_size=vocab_size,
            embed_dim=config["embed_dim"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"]
        )

        #count parameters
        total_params = 0
        for param in model.parameters():
            total_params += param.size 

        #calculating memory requirements (4 bytes per float32 parameter)
        memory_mb = (total_params * BYTES_PER_FLOAT32) /MB_TO_BYTES

        print(f"{config['name']} Model:")
        print(f"  Parameters: {total_params:,}")
        print(f"  Memory: {memory_mb:.1f} MB")
        print(f"  Embed dim: {config['embed_dim']}, Layers: {config['num_layers']}")
        print()

    print("\nConclusion: Parameter scaling is roughly quadratic with embedding dimension")

if __name__ =="__main__":
    analyzing_parameter_scaling()