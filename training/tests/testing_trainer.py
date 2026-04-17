import os
import sys
import tempfile

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

def test_trainer():
    print("Test trainer")

    #creating a simple model using REAL Linear layer
    class SimpleModel:
        def __init__(self):
            self.layer=Linear(2,1) #real Linear from layers module
            self.training =True 

        def forward(self,x):
            return self.layer.forward(x)

        def parameters(self):
            return self.layer.parameters()

    
    #creating trainer with real components
    model = SimpleModel()
    optimizer = SGD(model.parameters(),lr=0.01) #real SGD from optimizers module
    loss_fn = MSELoss() #real MSELoss from losses module
    scheduler = CosineSchedule(max_lr=0.1,min_lr=0.01,total_epochs=10)

    trainer = Trainer(model,optimizer,loss_fn,scheduler,grad_clip_norm=0.1)

    #test training
    print("Testing training epoch...")
    #use real Tensors for data
    dataloader = [
        (Tensor([[1.0,0.5]]),Tensor([[2.0]])),
        (Tensor([[0.5,1.0]]),Tensor([[1.5]]))
    ]
    loss = trainer.train_epoch(dataloader)
    assert isinstance(loss,(float,np.floating)),f"Expected float loss, got {type(loss)}"
    assert trainer.epoch == 1, f"Expected epoch 1, got {trainer.epoch}"

    #Test evaluation
    print("Testing evaluation...")
    eval_loss,accuracy = trainer.evaluate(dataloader)
    assert isinstance(eval_loss,(float,np.floating)),f"Expected float eval_loss, got {type(eval_loss)}"
    assert isinstance(accuracy,(float,np.floating)),f"Expected float accuracy, got {type(accuracy)}"

    #test checkpointing
    print("Testing checkpointing...")
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pkl")
        trainer.save_checkpoint(checkpoint_path)

        #modify trainer state 
        original_epoch = trainer.epoch
        trainer.epoch = 999

        #loading checkpoint
        trainer.load_checkpoint(checkpoint_path)
        assert trainer.epoch == original_epoch, f"Checkpoint didn't restore epoch correctly"
    
    print(f"Trainer works correctly, Final loss: {loss:.4f}")

if __name__ == "__main__":
    test_trainer()
