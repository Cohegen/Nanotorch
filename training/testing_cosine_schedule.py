import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from training import CosineSchedule 

def testing_cosine_schedule():
    """
    Testing CosineSchedule implementation
    """
    print("Testing CosineSchedule..")

    #testing basic schedule
    schedule = CosineSchedule(max_lr=0.1,min_lr=0.01,total_epochs=100)

    #testing start,middle and end
    lr_start = schedule.get_lr(0)
    lr_middle = schedule.get_lr(50)
    lr_end = schedule.get_lr(100)

    print(f"Learning rate at epoch 0: {lr_start:.4f}")
    print(f"Learning rate at epoch 50: {lr_middle:.4f}")
    print(f"Learning rate at epoch 100:{lr_end:.4f}")

    #validate behavior
    assert abs(lr_start - 0.1) < 1e-6, f"Expected 0.1 at start, got {lr_start}"
    assert abs(lr_end-0.01) < 1e-6, f"Expected 0.01 at end, got {lr_end}"
    assert 0.01 < lr_middle < 0.1, f"Middle LR should be between min and max, got{lr_middle}"

    #testing monotonic decrease in first half
    lr_quater = schedule.get_lr(25)
    assert lr_quater > lr_middle, "LR should decrease monotonically in first half"

    print("CosineSchedule works correctly.")

if __name__ == "__main__":
    testing_cosine_schedule()