import os
import sys

sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor 

from Tensor.tensor import BYTES_PER_FLOAT32, MB_TO_BYTES
from attention import _compute_attention_scores,_scale_scores
import numpy as np
import math 

def testing_scale_scores():
    print("Score Scaling")
    scores = Tensor(np.array([[[4.0,8.0]]]))
    scaled = _scale_scores(scores,d_model=4)
    assert np.allclose(scaled.data,[[[2.0,4.0]]]),f"Expected /sqrt(4)=2, got {scaled.dta}"
    print("Score scaling works correcty")

if __name__ == "__main__":
    testing_scale_scores()
