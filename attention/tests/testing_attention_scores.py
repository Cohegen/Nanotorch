import os
import sys

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 

from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from attention import _compute_attention_scores
import numpy as np
import math 

def testing_attention_scores():
    """
    This function tests Q @ K^T produces correct similarity matrix shape and values
    """

    Q = Tensor(np.ones((1,3,4)))
    K = Tensor(np.ones((1,3,4)))
    scores = _compute_attention_scores(Q,K)
    assert scores.shape == (1,3,3), "All-ones Q@K^T should give d_model=4"
    print("Attention scores:correct shape and values")

if __name__ == "__main__":
    testing_attention_scores()