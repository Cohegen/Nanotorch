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
    """
    def __init__(self, params, defaults):
        self.defaults = defaults
        self.state = {}
        self.param_groups = []

        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]

        for param_group in param_groups:
            self.add_param_group(param_group)
        
        self.step_count = 0

    def add_param_group(self, param_group):
        assert isinstance(param_group, dict), "param_group must be a dict"
        params = param_group['params']
        if isinstance(params, Tensor):
            param_group['params'] = [params]
        elif not isinstance(params, list):
            param_group['params'] = list(params)
        
        for name, default in self.defaults.items():
            param_group.setdefault(name, default)
        
        self.param_groups.append(param_group)

    def zero_grad(self):
        for group in self.param_groups:
            for p in group['params']:
                p.grad = None

    def step(self):
        raise NotImplementedError()

class SGD(Optimizer):
    def __init__(self, params, lr=DEFAULT_LEARNING_RATE_SGD, momentum=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        self.step_count += 1
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            lr = group['lr']

            for i, p in enumerate(group['params']):
                if p.grad is None:
                    continue
                
                grad = p.grad.data if isinstance(p.grad, Tensor) else p.grad

                if weight_decay != 0:
                    grad = grad + weight_decay * p.data

                if momentum != 0:
                    state = self.state.get(id(p), {})
                    if 'momentum_buffer' not in state:
                        buf = state['momentum_buffer'] = np.zeros_like(p.data)
                    else:
                        buf = state['momentum_buffer']
                    
                    buf[:] = momentum * buf + grad
                    grad = buf
                    self.state[id(p)] = state

                p.data = p.data - lr * grad

class Adam(Optimizer):
    def __init__(self, params, lr=DEFAULT_LEARNING_RATE_ADAM, betas=(DEFAULT_BETA1, DEFAULT_BETA2), eps=DEFAULT_EPS, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        self.step_count += 1
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data if isinstance(p.grad, Tensor) else p.grad
                if group['weight_decay'] != 0:
                    grad = grad + group['weight_decay'] * p.data

                state = self.state.get(id(p), {})
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = np.zeros_like(p.data)
                    state['exp_avg_sq'] = np.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                exp_avg[:] = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * (grad * grad)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                p.data -= step_size * exp_avg / (np.sqrt(exp_avg_sq / bias_correction2) + group['eps'])
                self.state[id(p)] = state

class AdamW(Optimizer):
    def __init__(self, params, lr=DEFAULT_LEARNING_RATE_ADAM, betas=(DEFAULT_BETA1, DEFAULT_BETA2), eps=DEFAULT_EPS, weight_decay=DEFAULT_WEIGHT_DECAY_ADAMW):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self):
        self.step_count += 1
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            for p in group['params']:
                if p.grad is None:
                    continue
                
                # Perform step weight decay
                if group['weight_decay'] != 0:
                    p.data -= group['lr'] * group['weight_decay'] * p.data

                grad = p.grad.data if isinstance(p.grad, Tensor) else p.grad

                state = self.state.get(id(p), {})
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = np.zeros_like(p.data)
                    state['exp_avg_sq'] = np.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1

                exp_avg[:] = beta1 * exp_avg + (1 - beta1) * grad
                exp_avg_sq[:] = beta2 * exp_avg_sq + (1 - beta2) * (grad * grad)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                step_size = group['lr'] / bias_correction1
                p.data -= step_size * exp_avg / (np.sqrt(exp_avg_sq / bias_correction2) + group['eps'])
                self.state[id(p)] = state
                
