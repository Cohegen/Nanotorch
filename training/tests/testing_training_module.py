""""
This script puts together every concept
in this module into a single use case.

"""
import os
import sys

from numpy.polynomial import test


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from Tensor import Tensor
from training import clip_grad_norm,Trainer
from training import CosineSchedule
from layers.layers import Linear
from optimizers.optimizers import SGD
from losses.losses import MSELoss
from dataloader.dataloader import Dataloader

def test_training_module():
    #creating a simple model
    class NanoModel:
        def __init__(self):
            self.layer = Linear(2,1)
            self.training = True 

        def forward(self,x):
            return self.layer.forward(x)

        def parameters(self):
            return self.layer.parameters()

    #creating integrated system
    model = NanoModel()
    optimizer = SGD(model.parameters(),lr=0.01)
    loss_fn = MSELoss()
    scheduler = CosineSchedule(max_lr=0.1,min_lr=0.001,total_epochs=3)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        grad_clip_norm=0.5
    )

    #testing data using tensors
    data = [
        (Tensor([[1.0,0.5]]),Tensor([[0.8]])),
        (Tensor([[0.5,1.0]]),Tensor([[0.2]]))
    ]

    #testing training
    intial_loss = trainer.train_epoch(data)
    assert isinstance(intial_loss,(float,np.floating)),"Evaluation should return float loss"
    assert trainer.epoch == 1, "Epoch should increment"
    
    #testing evaluation
    eval_loss,accuracy = trainer.evaluate(data)
    assert isinstance(eval_loss,(float,np.floating)),"Evaluation should return float loss"
    assert isinstance(accuracy,(float,np.floating)),"Evaluation should return float accuracy"



    #testing scheduling
    lr_epoch_0 =scheduler.get_lr(0)
    lr_epoch_1 = scheduler.get_lr(1)
    assert lr_epoch_0 > lr_epoch_1,"Learning rate should decrease"

    #testing gradient clipping with large gradients using real tensor
    large_param = Tensor([1.0,2.0],requires_grad=True)
    large_param.grad = np.array([100.0,200.0])
    large_params = [large_param]

    original_norm = clip_grad_norm(large_params,max_norm=1.0)
    assert original_norm >1.0,"Original norn should be large"

    if isinstance(large_params[0].grad,np.ndarray):
        grad_data = large_params[0].grad 
    else:
        grad_data = large_params[0].grad.data
    new_norm = np.linalg.norm(grad_data)
    assert abs(new_norm-1.0) <1e-6, "Clipped norm should equal max_norm"

    #testing checkpoint
    checkpoint_path = "/tmp/integration_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)

    original_epoch =trainer.epoch
    trainer.epoch = 999
    trainer.load_checkpoint(checkpoint_path)

    assert trainer.epoch == original_epoch, "Checkpoint should restore state"

    #cleaning memory 
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

if __name__ == "__main__":
    test_training_module()
