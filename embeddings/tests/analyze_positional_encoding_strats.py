import os
import sys


 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 

from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from embeddings import Embedding,PositionalEncoding,_compute_sinusoidal_table,create_sinusoidal_embeddings
from embeddings import EmbeddingLayer
import numpy as np
import math 

def analyze_positional_encoding_strategies():
    """
    This function Compares positional encoding approaches and tradeoffs

    """
    max_seq_len = 512 
    embed_dim = 256 

    #creating both types of positional encodings
    learned_pe = PositionalEncoding(max_seq_len,embed_dim)
    sinusoidal_pe = create_sinusoidal_embeddings(max_seq_len,embed_dim)

    #analyzing memory footprint
    learned_params = max_seq_len * embed_dim 
    learned_memory = learned_params * 4 / (1024*1024) #converting to megabytes 

    print(f"Learned PE:     {learned_memory:.2f} MB ({learned_params:,} parameters)")
    print(f"Sinusoidal PE:  0.00 MB (0 parameters)")

    #analyzing encoding patterns
    print("\nEncoding Pattern analysis: ")

    #testing sample sequences
    test_input=Tensor(np.random.randn(1,10,embed_dim))

    learned_output = learned_pe.forward(test_input)

    #For sinusoidal, we manually add to match learned interface 
    sin_encodings = sinusoidal_pe.data[:10][np.newaxis,:,:]#(1,10,embed_dim)
    sinusoidal_output = Tensor(test_input.data + sin_encodings)

    #analyzing variance across positions
    learned_var = np.var(learned_output.data,axis=1).mean()#variance across positions
    sin_var = np.var(sinusoidal_output.data,axis=1).mean()

    print(f"Position variance (learned):    {learned_var:.4f}")
    print(f"Position variance (sinusoidal): {sin_var:.4f}")

    #checking extrapolation capability
    print(f"\nExtrapolation Analysis: ")
    extended_length = max_seq_len + 100

    try:
        #learned PE cannit handle longer sequences 
        extended_learbed = PositionalEncoding(extended_length,embed_dim)
        print(f"Learned PE: Requires retraining for sequences > {max_seq_len}")

    except:
        print(f"Learned PE: Cannot handle sequences > {max_seq_len}")

    #Sinusoidal can extrapolate
    extended_sin = create_sinusoidal_embeddings(extended_length,embed_dim)
    print(f"Sinusoidal PE: Can extrapolate to length {extended_length} (smooth continuation)")

if __name__ == "__main__":
    analyze_positional_encoding_strategies()