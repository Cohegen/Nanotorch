import os 
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
from convolutions import MaxPool2d,AvgPool2d
import numpy as np

def analyze_pooling_effects():
    """
    Analyzes pooling's impact on spatial dimensions ad features
    """

    #Creating ssample input with spatial stucture
    #simple edge pattern that pooling should preserve differently
    pattern = np.zeros((1,1,8,8))
    pattern[0,0,:,3:5] = 1.0 #vertical edge
    pattern[0,0,3:5,:] = 1.0 #horizontal edge
    x = Tensor(pattern)

    print("Original 8x8 pattern:")
    print(x.data[0,0])

    #testing different pooling strategies
    pools = [
        (MaxPool2d(2, stride=2), "MaxPool 2×2"),
        (AvgPool2d(2, stride=2), "AvgPool 2×2"),
        (MaxPool2d(4, stride=4), "MaxPool 4×4"),
        (AvgPool2d(4, stride=4), "AvgPool 4×4"),
    ]

    print(f"\n{'Operation':<15} {'Output Shape':<15} {'Feature Preservation'}")
    print("-" * 60)

    for pool_op,name in pools:
        result = pool_op(x)
        #measuring how much of the original pattern is preserved
        preservation = np.sum(result.data > 0.1) / np.prod(result.shape)
        print(f"{name:<15} {str(result.shape):<15} {preservation:<.2%}")

        print(f"  Output:")
        print(f"   {result.data[0,0]}")
        print()

    print("💡 Key Insights:")
    print(" MaxPool preserves sharp features better (edge detection)")
    print(" AvgPool smooths features (noise reduction)")
    print(" Larger pooling windows lose more spatial detail")
    print(" Choice depends on task: classification vs detection vs segmentation")

if __name__ == "__main__":
    analyze_pooling_effects()
