import os 
import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from compression import measure_sparsity,structured_prune,low_rank_approximate,KnowledgeDistillation
import numpy as np

def testing_knowledge_distillation():
    """
    This function intendes to test whether the testing_knowledge_distillation
    works
    """
    #creating teacher model with more capacity
    teacher_l1= Linear(10,20)
    teacher_l2 = Linear(20,5)
    teacher = Sequential(teacher_l1,teacher_l2)

    #creates smaller student model
    student_l1 = Linear(10,5)
    student =Sequential(student_l1) #direct connection, no hidden layer

    #initialize knowledge distillation with temperature scaling
    kd = KnowledgeDistillation(teacher,student,temperature=3.0,alpha=0.7)

    #creating dummy data for testing
    input_data = Tensor(np.random.randn(8,10)) #batch of 8 samples
    true_labels = np.array([0, 1, 2, 3, 4, 0, 1, 2]) #class labels

    #forward passes 
    teacher_output = teacher.forward(input_data)  # Large model predictions
    student_output = student.forward(input_data)  # Small model predictions

    #calculate distillation loss
    loss = kd.distillation_loss(student_output,teacher_output,true_labels)

    #verifying loss is reasonable
    assert isinstance(loss, (float, np.floating)), f"Loss should be float, got {type(loss)}"
    assert loss > 0, f"Loss should be positive, got {loss}"
    assert not np.isnan(loss), "Loss should not be NaN"

    print("knowledge_distillation works correctly")

if __name__ =="__main__":
    testing_knowledge_distillation()
