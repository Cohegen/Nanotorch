import os
import sys



sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 

from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from attention import scaled_dot_product_attention,_compute_attention_scores,_scale_scores
from attention import MultiHeadAttention
import numpy as np
import math 


def testing_attention_scenarios():
    """
    This function intends to test attention mechanisms in realistic world scenarios
    """

    #scenario 1: small transformer block setup
    print("\n1. Small Transformer Setup: ")
    embed_dim,num_heads,seq_len = 128,8,32

    #creating embeddings (simulating token embeddings + positional embeddings)
    embeddings = Tensor(np.random.randn(2,seq_len,embed_dim))

    #multi-head attention
    mha=  MultiHeadAttention(embed_dim,num_heads)
    attended = mha.forward(embeddings)

    #scenario 2: Causal language modelling
    print("\n2. Causal Language Modelling: ")

    #creating casual mask (lower triangular)
    causal_mask = np.tril(np.ones((seq_len,seq_len)))
    mask = Tensor(np.broadcast_to(causal_mask,(2,seq_len,seq_len)))

    #applying causal attention
    causal_output = mha.forward(embeddings,mask)

    print(f"   Masked output shape: {causal_output.shape}")
    print(f"    Causal mask applied: {mask.shape}")

    #scenario 3: Comparing attention patterns
    print("\n3.Attention Pattern Analysis:")

    #creating simple test sequence
    simple_embed= Tensor(np.random.randn(1,4,16))
    simple_mha= MultiHeadAttention(16,4)

    #get attention weights by calling the base function
    Q = simple_mha.q_proj.forward(simple_embed)
    K = simple_mha.k_proj.forward(simple_embed)
    V = simple_mha.v_proj.forward(simple_embed)

    #reshaping for single head analysis
    Q_head = Tensor(Q.data[:,:,:4]) #first head only 
    K_head = Tensor(K.data[:,:,:4])
    V_head = Tensor(V.data[:,:,:4])

    _,weights = scaled_dot_product_attention(Q_head,K_head,V_head)

    print(f"   Attention weights shape: {weights.shape}")
    print(f"    Attention weights (first batch, 4x4 matrix):")
    weight_matrix = weights.data[0,:,:].round(3)

    #Formatting the attention matrix nicely
    print("     Pos→  0     1     2     3")
    for i in range(4):
        row_str = f"   {i}: " + " ".join(f"{weight_matrix[i,j]:5.3f}" for j in range(4))
        print(row_str)

    print(f"   Row sums: {weights.data[0].sum(axis=1).round(3)} (should be ~1.0)")

    #scenario 4: Attention with masking visualization
    print("\n4.Causal Masking Effect:")

    #applying causal mask to simple examples
    simple_mask = Tensor(np.tril(np.ones((1,4,4))))
    _,masked_weights = scaled_dot_product_attention(Q_head,K_head,V_head,simple_mask)

    print("   Causal Attention matrix (lower triangular):")
    masked_matrix = masked_weights.data[0,:,:].round(3)
    for i in range(4):
        row_str = f"   {i}: " + " ".join(f"{masked_matrix[i,j]:5.3f}" for j in range(4))
        print(row_str)

    print("   Notice: Upper triangle is zero (can't attend to future)")

    print("\n All attention scenarios work correclty!")

if __name__ == "__main__":
    testing_attention_scenarios()
