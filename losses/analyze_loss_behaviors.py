import os
import sys
from unittest import result
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses import CrossEntropyLoss,EPILSON,BinaryCrossEntropyLoss,MSELoss
from Tensor import Tensor
from activations.activations import ReLU
from layers.layers import Linear
import numpy as np

def analyze_loss_behaviors():
    """This function compares how different loss functions behae with various predictions patterns."""

    #intiliazing loss functions
    mse_loss = MSELoss()
    ce_loss =CrossEntropyLoss()
    bce_loss = BinaryCrossEntropyLoss()

    print("\n1. Regression Scenario (House Predictions)")
    print("   predictions: [200k,250k,300k],Targets: [195k,260k,290k]")
    house_pred = Tensor([200.0,250.0,300.0]) #in thousands
    house_target = Tensor([195.0,260.0,290.0])
    mse = mse_loss.forward(house_pred,house_target)
    print(f"  MSE Loss: {mse.data:.2f} (thousand^2)")

    print("\n2. Multi-class Classification (Image Recognition)")
    print("  classes: [cat,dog,bird], Predicted: confident about cat, uncertain about dog")
    #logits: [2.0,0.5,0.1] suggests model is most confident about class 0 i.e cat
    image_logits = Tensor([[2.0,0.5,0.1],[0.3,1.8,0.2]])#two samples
    image_targets = Tensor([0,1])
    ce = ce_loss.forward(image_logits,image_targets)
    print(f" Cross-Entropy Loss: {ce.data:.3f}")

    print("\n3. Binary Classification i.e Spam Detection")
    print("  Predictions: [0.9,0.1,0.7,0.3] (spam probabilites)")
    spam_pred = Tensor([0.9,0.1,0.7,0.3])
    spam_target = Tensor([1.0,0.0,1.0,0.1])
    bce = bce_loss.forward(spam_pred,spam_target)
    print(f"  Binary Cross-Entropy Loss: {bce.data:.3f}")

    return mse.data, ce.data, bce.data

if __name__ == "__main__":
    analyze_loss_behaviors()