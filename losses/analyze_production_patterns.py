import os
import sys
from unittest import result
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from losses import CrossEntropyLoss,EPILSON,BinaryCrossEntropyLoss,MSELoss
from Tensor import Tensor
from activations.activations import ReLU
from layers.layers import Linear
import numpy as np

def analyze_production_patterns():
    """Analyze loss function patterns in production ML systems."""

    print("Production Analysis: Loss Function Engineering Patterns...")
    
    print("\n1. Loss Function Choice by Problem Type:")

    scenarios = [
        ("Recommender Systems", "BCE/MSE", "User preference prediction", "Billions of interactions"),
        ("Computer Vision", "CrossEntropy", "Image classification", "1000+ classes, large batches"),
        ("NLP Translation", "CrossEntropy", "Next token prediction", "50k+ vocabulary"),
        ("Medical Diagnosis", "BCE", "Disease probability", "Class imbalance critical"),
        ("Financial Trading", "MSE/Huber", "Price prediction", "Outlier robustness needed")
    ]

    print("System Type          | Loss Type    | Use Case              | Scale Challenge")
    print("-" * 80)
    for system, loss_type, use_case, challenge in scenarios:
        print(f"{system:20} | {loss_type:12} | {use_case:20} | {challenge}")

    print("\n2. Engineering Trade-offs:")

    trade_offs = [
        ("CrossEntropy vs Label Smoothing", "Stability vs Confidence", "Label smoothing prevents overconfident predictions"),
        ("MSE vs Huber Loss", "Sensitivity vs Robustness", "Huber is less sensitive to outliers"),
        ("Full Softmax vs Sampled", "Accuracy vs Speed", "Hierarchical softmax for large vocabularies"),
        ("Per-Sample vs Batch Loss", "Accuracy vs Memory", "Batch computation is more memory efficient")
    ]

    print("\nTrade-off                    | Spectrum              | Production Decision")
    print("-" * 85)
    for trade_off, spectrum, decision in trade_offs:
        print(f"{trade_off:28} | {spectrum:20} | {decision}")

    print("\n💡 Production Insights:")
    print("   - Large vocabularies (50k+ tokens) dominate memory in CrossEntropy")
    print("   - Batch computation is 10-100× more efficient than per-sample")
    print("   - Numerical stability becomes critical at scale (FP16 training)")
    print("   - Loss computation is often <5% of total training time")

if __name__ == "__main__":
    analyze_production_patterns()