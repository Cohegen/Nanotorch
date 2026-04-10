import os
import sys

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 

from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from attention import _compute_attention_scores,_scale_scores,_apply_mask
import numpy as np
import math 


def testing_apply_mask():
    scores = Tensor(np.ones((1,3,3)))
    mask = Tensor(np.tril(np.ones((1,3,3))))
    masked= _apply_mask(scores,mask)

    #future positions should be unchanged
    assert masked.data[0,0,1] < -1e8, "Future position not masked"
    #past positions should be unchanged
    assert np.allclose(masked.data[0,0,0],1.0),"Pat position was modified"
    print("Causal masking works correctly")

if __name__ == "__main__":
    testing_apply_mask()