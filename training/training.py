import sched
import numpy as np
import time 
from typing import Dict,List,Optional,Tuple,Any,Callable
from pathlib import Path 
import sys
import os 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#importing dependecies from other modules
from Tensor import Tensor 
from layers.layers import Linear
from losses.losses import MSELoss,CrossEntropyLoss
from optimizers.optimizers import SGD,AdamW
from nanotorch.utils.checkpointing import load_checkpoint as load_training_checkpoint
from nanotorch.utils.checkpointing import save_checkpoint as save_training_checkpoint
from nanotorch.utils.validation import (
    assert_finite_parameters,
    assert_finite_tensor,
    collect_gradient_issues,
)

#Constant for learning rate scheduling defaults
DEFAULT_MAX_LR = 0.1 #default maximum learning rate for cosine schedule
DEFAULT_MIN_LR = 0.01 #default minimum learning rate for cosine schedule
DEFAULT_TOTAL_EPOCHS =100 #default total epochs for learning rate schedule

class CosineSchedule:
    """
    Cosine annealing learning rate schedule.

    Starts at max_lr then decreases following a cosine curve to mi min_lr over T epochs.
    This thereby provides aggressive learing intially then fine tuing at the end.
    """
    def __init__(self,max_lr:float=DEFAULT_MAX_LR,min_lr:float =DEFAULT_MIN_LR,total_epochs:int =DEFAULT_TOTAL_EPOCHS):
        self.max_lr = max_lr
        self.min_lr = min_lr 
        self.total_epochs = total_epochs
        
    def get_lr(self,epoch:int)-> float:
        """
        Get learning rate for current epoch.
        """
        if epoch >= self.total_epochs:
            return self.min_lr

        #cosine annealing formula
        cosine_factor = (1+np.cos(np.pi * epoch / self.total_epochs)) / 2
        return self.min_lr + (self.max_lr - self.min_lr) * cosine_factor

def clip_grad_norm(parameters:List,max_norm:float = 1.0)->float:
    """
    Clipping gradients by global norm to prevent exploding gradients.


    """
    if not parameters:
        return 0.0

    #collect all gradients and compute global norm
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            #handles both Tensor gradient and numpy array gradients
            if isinstance(param.grad,np.ndarray):
                grad_data = param.grad
            else:
                #trust that Tensor has .data attribute
                grad_data = param.grad.data
            total_norm += np.sum(grad_data**2)

    total_norm = np.sqrt(total_norm)

    #clip if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / total_norm
        for param in parameters:
            if param.grad is not None:
                #handle both Tensor gradients and numpy arrays gradients
                if isinstance(param.grad,np.ndarray):
                    param.grad = param.grad * clip_coef
                else:
                    #trusting that Tensor has .data attribute
                    param.grad.data = param.grad.data * clip_coef

    return float(total_norm)

class Trainer:
    """
    Complete trainer for neural networks.

    Handles the full training lifecycle: forward pass,
    loss computation,backward pass optimization,scheduling,checkpointing and evaluation

    This is the central class that brings together all the
    components we've built in other modules
    """
    def __init__(self,model,optimizer,loss_fn,scheduler=None,grad_clip_norm=None,raise_on_nonfinite=True):
        """
        Intialize trainer with model and training components

        Args:
            model: Neural network to train
            optimizer:Parameter update strategy (SGD,Adam,etc)
            loss_fn: Loss function (CrossEntropy,MSE)
            scheduler:Optional learning rate scheduler
            grad_clip_norm:optional gradient clipping threshold

        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn =loss_fn
        self.scheduler = scheduler
        self.grad_clip_norm = grad_clip_norm
        self.raise_on_nonfinite = raise_on_nonfinite

        #training state 
        self.epoch =0
        self.step =0
        self.training_mode = True 
        self.last_gradient_issues = {
            'missing_grad_indices': [],
            'nonfinite_grad_indices': [],
        }

        #history tracking
        self.history = {
            'train_loss':[],
            'eval_loss':[],
            'learning_rates':[]
        }

    def train_epoch(self, dataloader, accumulation_steps=1):
        """
        Train for one epoch through the dataset.

        Args:
            dataloader: iterable yielding (inputs, target) batches
            accumulation_steps: number of batches to accumulate before update

        Returns:
            Average loss for the epoch
        """
        self.model.training = True
        self.training_mode = True

        total_loss = 0.0
        num_batches = 0
        num_updates = 0
        accumulated_loss = 0.0

        self.optimizer.zero_grad()

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # forward pass
            outputs = self.model.forward(inputs)
            loss = self.loss_fn.forward(outputs, targets)
            if self.raise_on_nonfinite:
                assert_finite_tensor(loss, name="loss")

            # scale loss for accumulation
            batch_loss = loss.data if isinstance(loss.data, (float, np.float32, np.float64)) else loss.data.item()
            accumulated_loss += batch_loss
            
            # backward pass with scaled gradient
            # We scale the gradient by 1/accumulation_steps so that after summing 
            # accumulation_steps gradients, we have the average gradient.
            scaled_gradient = np.ones_like(loss.data) / accumulation_steps
            loss.backward(scaled_gradient)

            # updating parameters every accumulation steps
            if (batch_idx + 1) % accumulation_steps == 0:
                # gradient clipping
                if self.grad_clip_norm is not None:
                    params = self.model.parameters()
                    clip_grad_norm(params, self.grad_clip_norm)

                self.last_gradient_issues = collect_gradient_issues(self.model.parameters())
                if self.raise_on_nonfinite and self.last_gradient_issues['nonfinite_grad_indices']:
                    raise ValueError(
                        f"Encountered non-finite gradients: {self.last_gradient_issues['nonfinite_grad_indices']}"
                    )

                # optimizer step
                self.optimizer.step()
                if self.raise_on_nonfinite:
                    assert_finite_parameters(self.model)
                self.optimizer.zero_grad()

                total_loss += accumulated_loss / accumulation_steps
                accumulated_loss = 0.0
                num_updates += 1
                self.step += 1

            num_batches += 1

        # handling remaining accumulated gradients
        if (num_batches % accumulation_steps) != 0:
            remaining_steps = num_batches % accumulation_steps
            # Optional: Rescale gradients if we want the exact average over the smaller batch
            # But usually we just step with what we have.
            
            if self.grad_clip_norm is not None:
                params = self.model.parameters()
                clip_grad_norm(params, self.grad_clip_norm)

            self.last_gradient_issues = collect_gradient_issues(self.model.parameters())
            if self.raise_on_nonfinite and self.last_gradient_issues['nonfinite_grad_indices']:
                raise ValueError(
                    f"Encountered non-finite gradients: {self.last_gradient_issues['nonfinite_grad_indices']}"
                )

            self.optimizer.step()
            if self.raise_on_nonfinite:
                assert_finite_parameters(self.model)
            self.optimizer.zero_grad()
            total_loss += accumulated_loss / remaining_steps
            num_updates += 1
            self.step += 1

        avg_loss = total_loss / max(num_updates, 1)
        self.history['train_loss'].append(avg_loss)

        # update scheduler
        if self.scheduler is not None:
            current_lr = self.scheduler.get_lr(self.epoch)
            # update optimizer learning rate across all parameter groups
            for group in self.optimizer.param_groups:
                group['lr'] = current_lr
            self.history['learning_rates'].append(current_lr)

        self.epoch += 1
        return avg_loss

    def evaluate(self,dataloader):
        """
        Evaluate model on dataset without updating parameters

        Args:
            dataloader: iterable yielding (inputs,targets) batches

        Returns:
            Average loss and accuracy

        """
        self.model.training=False
        self.training_mode = False

        total_loss = 0.0
        correct= 0
        total = 0
        num_batches =0

        for inputs,targets in dataloader:
            #forward pass only
            outputs = self.model.forward(inputs)
            loss = self.loss_fn.forward(outputs,targets)
            if self.raise_on_nonfinite:
                assert_finite_tensor(loss, name="eval_loss")

            total_loss += loss.data
            num_batches += 1

            #calculates accuracy for classification
            if len(outputs.data.shape) >1: #multiclass
                predictions = np.argmax(outputs.data,axis=1)
                if len(targets.data.shape) == 1:#integer targets
                    total += len(targets.data)
                    correct += np.sum(predictions == targets.data)
                else: #one-hot targets
                    total += len(targets.data)
                    correct += np.sum(predictions == np.argmax(targets.data,axis=1))


        avg_loss = total_loss /num_batches if num_batches > 0 else 0.0
        accuracy = correct / total if total > 0 else 0.0
        
        self.history['eval_loss'].append(avg_loss)

        return avg_loss,accuracy

    def save_checkpoint(self,path:str):
        """
        Save complete training state for resumption

        Args:
           path:file path to save checkpoint

        """
        return save_training_checkpoint(
            path,
            self.model,
            optimizer=self.optimizer,
            epoch=self.epoch,
            metadata={
                'step': self.step,
                'scheduler_state': self._get_scheduler_state(),
                'history': self.history,
                'training_mode': self.training_mode,
            },
        )

    def load_checkpoint(self,path:str):
        """
        load training state from checkpoint

        Args:
            path: file path to load checkpoint from
        """
        checkpoint = load_training_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            strict=True,
        )

        self.epoch = checkpoint.get('epoch', self.epoch)
        metadata = checkpoint.get('metadata', {})
        self.step = metadata.get('step', self.step)
        self.history = metadata.get('history', self.history)
        self.training_mode = metadata.get('training_mode', self.training_mode)
        if 'scheduler_state' in metadata:
            self._set_scheduler_state(metadata['scheduler_state'])
        return checkpoint

    def _get_scheduler_state(self):
        """Extract scheduler state for checkpointing"""
        if self.scheduler is None:
            return None
        return {
            'max_lr':getattr(self.scheduler,'max_lr',None),
            'min_lr':getattr(self.scheduler,'min_lr',None),
            'total_epochs':getattr(self.scheduler,'total_epochs',None)
        }

    def _set_scheduler_state(self,state):
        """Restore scheduler state from checkpoint"""
        if state is None or self.scheduler is None:
            return 
        for key,value in state.items():
            setattr(self.scheduler,key,value)
            
