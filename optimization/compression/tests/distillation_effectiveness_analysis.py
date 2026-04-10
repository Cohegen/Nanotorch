import os 

import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from compression import measure_sparsity,magnitude_prune,structured_prune,low_rank_approximate
from optimization.profiling.profiling import Profiler
import numpy as np

def distillation_analysis():
    """
    This function analyzes knowledge distillation compression and accuracy tradeoffs
    """
    print("\n Analyzing Knowlegde Distillation Effectiveness")
    print("="*60)

    #simulating teacher-sudent scenarios
    scenarios = [
        ("Large→Small", 100_000, 10_000, 0.95, 0.90, 10.0),
        ("Medium→Tiny", 50_000, 5_000, 0.92, 0.87, 10.0),
        ("Small→Micro", 10_000, 1_000, 0.88, 0.83, 10.0),
    ]
    print(f"\n{'Scenario':<15} {'Teacher':<12} {'Student':<12} {'Ratio':<10} {'Acc Loss':<10}")
    print("-" * 60)

    for name,teacher_params,student_params,teacher_acc,student_acc,compression in scenarios:
         acc_retention = (student_acc / teacher_acc) * 100
         acc_loss = teacher_acc - student_acc

         print(f"{name:<15} {teacher_params:>10,}p {student_params:>10,}p {compression:>8.1f}x {acc_loss*100:>8.1f}%")

    print("\n Knowledge Distillation Insights:")
    print("   • Achieves 10x+ compression with 5-10% accuracy loss")
    print("   • Student learns teacher's 'soft' predictions")
    print("   • More effective than naive pruning for large reductions")
    print("   • Requires retraining (unlike pruning/quantization)")
    print("\n Best Use Case:")
    print("   Deploy small student models on edge devices")
    print("   Train expensive teacher once, distill many students")
if __name__ == "__main__":
    distillation_analysis()