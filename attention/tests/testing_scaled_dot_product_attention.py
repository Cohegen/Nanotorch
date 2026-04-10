import os
import sys

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 

from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from attention import _compute_attention_scores,_scale_scores,_apply_mask,scaled_dot_product_attention 
import numpy as np
import math 

def testing_scaled_dot_product_attention():
    """
    This function is intended to test whether 
    scaled dot product attention implementation works correctly.

    """

    #testing basic functionality
    batch_size,seq_len,d_model = 2,4,8
    Q = Tensor(np.random.randn(batch_size,seq_len,d_model))
    K = Tensor(np.random.randn(batch_size,seq_len,d_model))
    V = Tensor(np.random.randn(batch_size,seq_len,d_model))

    output,weights = scaled_dot_product_attention(Q,K,V)

    #check output shapes
    assert output.shape == (batch_size,seq_len,d_model),f"Output  {output.shape} incorrect"
    assert weights.shape == (batch_size,seq_len,seq_len),f"Weights shape {weights.shape} incorrect"

    #checking attention weights sum to 1 (probability distribution)
    weights_sum = weights.data.sum(axis=2) #sum over last dimension
    expected_sum = np.ones((batch_size,seq_len))
    assert np.allclose(weights_sum,expected_sum,atol=1e-6),"Attention weights don't sum to 1"

    #testing with casual mask
    mask = Tensor(np.tril(np.ones((batch_size,seq_len,seq_len)),k=0)) #lower triangular
    output_masked,weights_masked = scaled_dot_product_attention(Q,K,V,mask)

    #checking that future positions have zero attention
    for b in range(batch_size):
        for i in range(seq_len):
            for j in range(i+1,seq_len): #future positions
                assert abs(weights_masked.data[b,i,j]) < 1e-6,f"Future attention not masked at ({i},{j}) "

    print("scaled_dot_product_attention works correctly")

if __name__ == "__main__":
    testing_scaled_dot_product_attention()