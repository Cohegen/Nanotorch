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

def testing_full_training_pipeline():
    """
    full end to end training example using all components

    """

    #creating a model using Real Linear layer
    class SimpleNN:
        def __init__(self):
            self.layer1 = Linear(3,5)
            self.layer2 = Linear(5,2)
            self.training = True 

        def forward(self,x):
            x = self.layer1.forward(x)
            #simple relu like activation (max with 0)
            x = Tensor(np.maximum(0,x.data))
            x = self.layers2.forward(x)
            return x

        def parameters(self):
            return self.layer1.parameters() + self.layer2.parameters()

    print("Model created: 3-> 5->2 network")

    #creating optimizer
    model=SimpleNN()
    optimizer = SGD(model.parameters(),lr=0.1,momentum=0.9)
    print("Optimizer: SGD with momentum")

    #creating a loss function
    loss_fn = MSELoss()
    print("Loss function:MSE")

    #creating scheduler
    scheduler = CosineSchedule(max_lr=0.1,min_lr=0.001,total_epochs=5)
    print("Scheduler:Cosine annealing (0.1-> 0.001)")

    #creating trainer with gradient clipping
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        scheduler=scheduler,
        grad_clip_norm=1.0
    )
    print("Trainer intialized with gradient clipping")

    #creating synthetic training data
    train_data = [
        (Tensor(np.random.randn(4,3)),Tensor(np.random.randn(4,2))),
        (Tensor(np.random.randn(4,3)),Tensor(np.random.randn(4,2))),
        (Tensor(np.random.randn(4,3)),Tensor(np.random.randn(4,2)))

    ]
    print("Training data: 3 batches and 4 samples")

    #training for multiple epochs
    print("\nStarting Training...")
    print("-"*60)
    print(f"{'Epoch':<8}{'Train Loss':<12} {'Learning Rate':<15}")
    print("-"*60)

    for epoch in range(3):
        loss = trainer.get_epoch(train_data)
        lr = scheduler.get_lr(epoch)
        print(f"{epoch:<8} {loss:<12.6f} {lr:<15.6f}")

    #save checkpoint
    checkpoint_path = "/tmp/training_example_checkpoint.pkl"
    trainer.save_checkpoint(checkpoint_path)
    print(f"\nCheckpoint saved:{checkpoint_path}")

    #evaluating
    eval_loss, accuracy = trainer.evaluate(train_data)
    print(f"✓ Evaluation - Loss: {eval_loss:.6f}, Accuracy: {accuracy:.6f}")


    #cleaning up memory
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print("\n"+ "=" *60)