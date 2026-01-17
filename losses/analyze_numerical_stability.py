import os
import sys
from unittest import result
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses import  log_softmax,CrossEntropyLoss,EPILSON,BinaryCrossEntropyLoss,MSELoss
from Tensor import Tensor
from activations.activations import ReLU
from layers.layers import Linear
import numpy as np

def analyze_numerical_stability():
    """A function that demonstrates why numerical stability matters in loss function"""

    print("Analysing Numerical stability in  losss functions..")

    #testing with increasingly large logits
    test_cases = [
        ("small logits",[1.0,2.0,3.0]),
        ("medium logits",[10.0,20.0,30.0]),
        ("large logits",[100.0,200.0,300.0]),
        ("very large logits",[500.0,600.0,700.0])

    ]

    print("\nLog-Softmax Stability Test: ")
    print("Case      |Max Input | log-softmax min| Numerically Stability")
    print("-"*70)

    for case_name, logits in test_cases:
        x = Tensor([logits])

        #our stable implementation
        stable_result = log_softmax(x,dim=-1)

        max_input = np.max(logits)
        min_output = np.min(stable_result.data)
        is_stable = not(np.any(np.isnan(stable_result.data)) or np.any(np.isinf(stable_result.data)))

        print(f"{case_name:20} | {max_input:8.0f} | {min_output:15.3f}")

    print(f"\nKey Insight: Log-sum-exp trick prevent overflow")
    print("   Without it: exp(700) would cause overflow in standard softmax")
    print("   With it: We can handle arbitrarily large logits safely")

if __name__ == "__main__":
    analyze_numerical_stability()   