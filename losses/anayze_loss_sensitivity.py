import os
import sys
from unittest import result
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses import CrossEntropyLoss,EPILSON,BinaryCrossEntropyLoss,MSELoss
from Tensor import Tensor
from activations.activations import ReLU
from layers.layers import Linear
import numpy as np

def analyze_loss_sensitivity():
    """Analyzing how sensitive each loss function is to prediction errors."""

    print("\nAnalysis: Loss Function Sensitivity to Errors: ")

    #creating a range of prediction errors for analysis
    true_value = 1.0
    predictions = np.linspace(0.1,1.9,50)

    #intializing loss functions
    mse_loss = MSELoss()
    bce_loss = BinaryCrossEntropyLoss()

    mse_losses = []
    bce_losses = []

    for pred in predictions:
        #MSE analysis
        pred_tensor =Tensor([pred])
        target_tensor = Tensor([true_value])
        mse = mse_loss.forward(pred_tensor,target_tensor)
        mse_losses.append(mse.data)

        #BCE analysis (clamping predictions to valid probability range)
        clamped_pred = max(0.01,min(0.99,pred))
        bce_pred_tensor = Tensor([clamped_pred])
        bce_target_tensor = Tensor([1.0]) #target is "positive class"
        bce = bce_loss.forward(bce_pred_tensor,bce_target_tensor)
        bce_losses.append(bce.data)

    #finding minimum losses
    min_mse_idx = np.argmin(mse_losses)
    min_bce_idx = np.argmin(bce_losses)

    print(f"MSE Loss:")
    print(f"  Minimum at prediction  = {predictions[min_mse_idx]:.2f},loss={mse_losses[min_mse_idx]:.4f}")
    print(f"  At prediction =0.5: loss: {mse_losses[24]:.4f}") #middle of range
    print(f"  At prediction=0.1: loss={mse_losses[0]:.4f}")

    print(f"\nBinary Cross-Entropy Loss: ")
    print(f" Minimum at prediction = {predictions[min_bce_idx]:.2f}, loss= {bce_losses[min_mse_idx]:.4f}")
    print(f" At predictions=0.5: loss={bce_losses[24]:.4f}")
    print(f"  At prediction = 0.1: loss={bce_losses[0]:.4f}")

    print(f"\n Sensitivity Insights:")
    print("   - MSE grows quadratically with error distance")
    print("   - BCE grows logarithmically, heavily penalizing wrong confident predictions")
    print("   - Both encourage correct predictions but with different curvatures")

if __name__ == "__main__":
    analyze_loss_sensitivity()