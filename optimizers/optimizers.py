from functools import lru_cache
import numpy as np
from typing import List,Union,Optional,Dict,Any 
import os
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

##import Tensor from Tensor now with gradient support from autograd
from Tensor import Tensor  

#enable autograd to add gradient tracking to Tensor
from autograd.autograd import enable_autograd 
enable_autograd()

#constants for optimzer defaults
DEFAULT_LEARNING_RATE_SGD = 0.01 #default learning rate for SGD(stochastic gradient descent)
DEFAULT_LEARNING_RATE_ADAM = 0.001 #default learning for Adam / AdamW
DEFAULT_MOMENTUM = 0.9 #default momentum for SGD 
DEFAULT_BETA1 = 0.9 # first moment decay rate for Adam
DEFAULT_BETA2 = 0.999 #second moment decay rate for Adam
DEFAULT_EPS = 1e-8 #small epilson for numerical stability in Adam 
DEFAULT_WEIGHT_DECAY_ADAMW = 0.01 #default weight decay for AdamW

class Optimizer:
    """
    Base class for all optimizers.

    This class defines the common interface that all optimizers must implement:
    -zero_grad(): clears gradients from parameters
    - step(): updates parameters based on gradients
    """
    def __init__(self,params:List[Tensor]):
        """
        Initializes optimizer with parameters to optimize.

        """
        #validating and storing parameters
        if not isinstance(params,list):
            params = list(params)

        #store parameters
        self.params = params 

        #ensuring parameters participate in autograd once it is enabled
        for param in self.params:
            if isinstance(params,Tensor):
                param.requires_grad =True 
                param.grad = None 
        self.step_count = 0 # for algorithms that need step counting

    def zero_grad(self):
        """
        Clears gradients from all parameters

        """
        ##iterating through all params
        for param in self.params:
            param.grad = None 

    def step(self):
        """
        Update parameters based on gradients

        This is abstract i.e each optimizer implements its own updatw rule 

        """
        raise NotImplementedError(
            f"Abstract method step() not implemented\n"
            f" Wrong {self.__class__.__name__} inherits from Optimizer but doesn't define step()\n"
            f"   Each optimizer must implement its won update rule (SGD,Adam,etc)\n"
            f"Override step() in your optimizer subclass:\n"
            f"    def step(self):\n"
            f"         for param in self.params:\n"
            f"                if param.grad is not None:\n"
            f"                          param.grad -= self.lr * param.grad.data"
        )

class SGD(Optimizer):
    """
    Stochastic Gradient Descent with momentum.

    SGD is the foundation optimization algortim that moves parameters
    in the direction opposite to gradient. With momentum, it remember
    previous updates to reduce oscillations and accelerate convergence.
    """

    def __init__(self,params:List[Tensor],lr:float = DEFAULT_LEARNING_RATE_SGD,momentum: float = 0.0,weight_decay:float=0.0):
        """
        Intialize SGD optimizer

        """
        ##calling parent constructor from Optimizer class
        super().__init__(params)

        self.lr =lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        #initializing momentum buffers
        self.momentum_buffers = [None for _ in self.params]

    def has_momentum(self) -> bool:
        """
        Check if this optimizer uses momentum.

        This explicit API method replaces the need for hasattr checks
        in checkpointing code

        Returns:
            bool: True if momentum is enabled i.e momentum >= 0, False otherwise
        """
        return self.momentum > 0

    def get_momentum(self) -> Optional[List]:
        """
        Get momentum bufffers for checkpointing

        This explicit API method provides safe access to momentum buffers
        without using hasattr, making the API contract clear.

        Return:
            Optional[List]:List of momentum buffers if momentum is enabled
                   None otherwise
        """
        if not self.has_momentum():
            return None
        return [buf.copy() if buf is not None else None for buf in self.momentum_buffers]

    def set_momentum_state(self,state:Optional[List]) -> None:
        """
        Restore momentum buffers from checkpointing.

        This explicit API method provides safe restoration of momentum state 
        without using hasattr

        Args:
            state: List of momentum buffers or None
        """

        if state is None or not self.has_momentum():
            return 
        if len(state) != len(self.momentum_buffers):
            raise ValueError(
                f"Momentum state length mismatch\n"
                f" Wrong!! State has {len(state)} buffers, but optimizers has {len(self.momentum_buffers)} parameters\n"
                f"  Checkpoint was saved with a different model architecture or parameter count\n"
                f"  Ensure you're loading state into an optimizwe with the same number of parameters:\n"
                f"    Check parameter counts mathc before restoring\n"
                f"   ssert len(saved_state) == len(optimizer.params)"

            )

            for i,buf in enumerate(state):
                if buf is not None:
                    self.momentum_buffers[i] = buf.copy()
    
    def step(self):
        """
        Perform SGD update step with momentum
        """
        for i,param in enumerate(self.params):
            if param.grad is None:
                continue

            #Get gradient data - grad can be Tensor or numpy array
            grad = param.grad 
            #handle both Tensor (with.data) and numpy array(from autograd) cases 
            if isinstance(grad,Tensor):
                grad_data = grad.data 
            else:
                #grad is already a numpy array from autograd 
                grad_data = grad

            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data

            #updating momentum buffer
            if self.momentum !=0:
                if self.momentum_buffers[i] is None:
                    #initialize momentum buffer
                    self.momentum_buffers[i] = np.zeros_like(param.data)
                
                # Update momentum: v = momentum * v_prev + grad
                self.momentum_buffers[i] = self.momentum * self.momentum_buffers[i] + grad_data
                grad_data = self.momentum_buffers[i]

            #update parameters: params = param- lr* grad 
            param.data = param.data - self.lr * grad_data 

        self.step_count += 1 

class Adam(Optimizer):
    """
    Adam Optimizer with adaptive learning rates.

    Adam computes individual adaptive learning rates for different parameters
    from estimates of first and second moments of the gradients.
    This makes it effective for problems with sparse gradients or noisy data.

    """ 

    def __init__(self,params:List[Tensor],lr:float=DEFAULT_LEARNING_RATE_ADAM,betas:tuple = (DEFAULT_BETA1,DEFAULT_BETA2),eps:float = DEFAULT_EPS,weight_decay:float =0.0):
          """
          Intialize Adam optimizer 

          """
          #calling parent constructor
          super().__init__(params)

          self.lr = lr 
          self.beta1, self.beta2 = betas 
          self.eps = eps 
          self.weight_decay = weight_decay 

          # intialize moment buffers
          self.m_buffers = [None for _ in self.params] # first moment i.e mean
          self.v_buffers = [None for _ in self.params] #second moment i.e variance


    def step(self):
        """
        Perform Adam Update step.
        """
        self.step_count += 1

        for i,param in enumerate(self.params):
            if param.grad is None:
                continue 


            #get gradient data i.egrad can be Tensor or numpy array
            grad = param.grad 
            #handles both Tensor (with .data) and numpy array(from autograd) cases
            if isinstance(grad,Tensor):
                grad_data = grad.data 
            else:
                #grad is already a numpy array from autograd 
                grad_data = grad 

            #apply weight decay 
            if self.weight_decay != 0:
                grad_data = grad_data + self.weight_decay * param.data 

            #intialize buffers if needed
            if self.m_buffers[i] is None:
                self.m_buffers[i] = np.zeros_like(param.data)
                self.v_buffers[i] = np.zeros_like(param.data)

            #update biased first moment estimate
            self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1- self.beta1)*grad_data 

            #update biased second moment estimatw
            self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1-self.beta2)* (grad_data **2 )

            #compute bias correction
            bias_correction1 = 1 -self.beta1 ** self.step_count
            bias_correction2 = 1 - self.beta2 ** self.step_count

            #compute bias-corrected moments
            m_hat = self.m_buffers[i] / bias_correction1
            v_hat = self.v_buffers[i] / bias_correction2

            param.data= param.data - self.lr * m_hat / (np.sqrt(v_hat)+ self.eps)


class AdamW(Optimizer):
    """
    AdamW optimizer with decoupled weight decay.

    Adam fixes a bug in Adam's weight decay implementation by decoupling
    weight decy from the gradient-based update. This leads to better regularization and
    is preferred version for most application.
    """
    def __init__(self,params:List[Tensor],lr:float=DEFAULT_LEARNING_RATE_ADAM,betas:tuple=(DEFAULT_BETA1,DEFAULT_BETA2),eps:float = DEFAULT_EPS,weight_decay:float = DEFAULT_WEIGHT_DECAY_ADAMW):
        """
        Intialize AdamW optimizer

        """
        super().__init__(params)

        self.lr =lr 
        self.beta1,self.beta2= betas
        self.eps =eps
        self.weight_decay = weight_decay

        #intialize moment buffers 
        self.m_buffers = [None for _ in self.params]
        self.v_buffers = [None for _ in self.params]

    def step(self):
        """
        Perform AdamW update step with decouled weight decay
        """
        ##increment step counter first 
        self.step_count += 1

        for i,param in enumerate(self.params):
            if param.grad is None:
                continue 

            # get gradient data i.e grad can be Tensor or Numpy array
            grad = param.grad
            #handles both Tensor (with .data) and numpy array from autograd cases
            if isinstance(grad,Tensor):
                grad_data = grad.data
            else:
                #grad is already a numpy array from autograd
                grad_data = grad

            # intialize buffers if needed
            if self.m_buffers[i] is None:
                self.m_buffers[i] = np.zeros_like(param.data)
                self.v_buffers[i] = np.zeros_like(param.data)

            #update moments using pure gradients
            self.m_buffers[i] = self.beta1 * self.m_buffers[i] + (1-self.beta1)* grad_data
            self.v_buffers[i] = self.beta2 * self.v_buffers[i] + (1-self.beta2) * (grad_data ** 2)

            #compute bias correction
            bias_correction1 = 1 -self.beta1 **self.step_count
            bias_correction2 = 1 - self.beta2 ** self.step_count 

            #compute bias-corrected moments
            m_hat = self.m_buffers[i] / bias_correction1
            v_hat = self.v_buffers[i] /bias_correction2

            #apply gradient-based update
            param.data = param.data - self.lr* m_hat / (np.sqrt(v_hat) + self.eps)

            #apply decoupled weight decay
            if self.weight_decay !=0:
                param.data = param.data * (1 - self.lr * self.weight_decay)
                