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
DEFAULT_MUON_MOMENTUM = 0.95
DEFAULT_MUON_NS_STEPS = 5

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

    def _all_params(self):
        params = []
        for group in self.param_groups:
            params.extend(group['params'])
        return params

    def state_dict(self):
        """Serializes optimizer hyperparameters and per-parameter state."""
        all_params = self._all_params()
        param_index = {id(param): index for index, param in enumerate(all_params)}

        serialized_groups = []
        for group in self.param_groups:
            serialized_group = {}
            for key, value in group.items():
                if key == 'params':
                    serialized_group['params'] = [param_index[id(param)] for param in value]
                else:
                    serialized_group[key] = value
            serialized_groups.append(serialized_group)

        serialized_state = {}
        for param in all_params:
            param_state = self.state.get(id(param), {})
            if not param_state:
                continue

            serialized_entries = {}
            for key, value in param_state.items():
                if isinstance(value, np.ndarray):
                    serialized_entries[key] = np.array(value, copy=True)
                else:
                    serialized_entries[key] = value
            serialized_state[param_index[id(param)]] = serialized_entries

        return {
            'defaults': dict(self.defaults),
            'state': serialized_state,
            'param_groups': serialized_groups,
            'step_count': self.step_count,
        }

    def load_state_dict(self, state_dict):
        """Loads optimizer state into the current optimizer instance."""
        saved_groups = state_dict['param_groups']
        if len(saved_groups) != len(self.param_groups):
            raise ValueError(
                f"Parameter group count mismatch: expected {len(self.param_groups)}, got {len(saved_groups)}"
            )

        all_params = self._all_params()
        current_group_sizes = [len(group['params']) for group in self.param_groups]
        saved_group_sizes = [len(group['params']) for group in saved_groups]
        if current_group_sizes != saved_group_sizes:
            raise ValueError(
                f"Parameter group sizes mismatch: expected {current_group_sizes}, got {saved_group_sizes}"
            )

        self.defaults = dict(state_dict.get('defaults', self.defaults))
        self.step_count = state_dict.get('step_count', 0)

        for current_group, saved_group in zip(self.param_groups, saved_groups):
            for key, value in saved_group.items():
                if key != 'params':
                    current_group[key] = value

        self.state = {}
        saved_state = state_dict.get('state', {})
        for param_index, entries in saved_state.items():
            param = all_params[param_index]
            restored_entries = {}
            for key, value in entries.items():
                if isinstance(value, np.ndarray):
                    restored_entries[key] = np.array(value, copy=True)
                else:
                    restored_entries[key] = value
            self.state[id(param)] = restored_entries

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

class Muon(Optimizer):
    """
    Momentum optimizer with matrix update orthogonalization.

    Muon is most useful for matrix-like parameters. For 1D parameters such as
    bias vectors, it falls back to momentum SGD so the optimizer can still be
    applied to a full parameter list.
    """
    def __init__(
        self,
        params,
        lr=DEFAULT_LEARNING_RATE_SGD,
        momentum=DEFAULT_MUON_MOMENTUM,
        nesterov=True,
        ns_steps=DEFAULT_MUON_NS_STEPS,
        eps=DEFAULT_EPS,
        weight_decay=0,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            eps=eps,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @staticmethod
    def _as_matrix(array):
        if array.ndim == 0:
            return array.reshape(1, 1), array.shape
        if array.ndim == 1:
            return array.reshape(1, -1), array.shape
        if array.ndim == 2:
            return array, array.shape
        return array.reshape(array.shape[0], -1), array.shape

    @staticmethod
    def _matrix_inverse_sqrt_orthogonalize(matrix, steps, eps):
        rows, cols = matrix.shape
        if rows == 0 or cols == 0:
            return matrix

        gram = matrix.T @ matrix if rows >= cols else matrix @ matrix.T
        dim = gram.shape[0]
        gram = gram + eps * np.eye(dim, dtype=matrix.dtype)

        trace = np.trace(gram)
        if not np.isfinite(trace) or trace <= 0:
            return matrix

        normed = gram / trace
        estimate = np.eye(dim, dtype=matrix.dtype)

        for _ in range(steps):
            estimate_sq = estimate @ estimate
            estimate = 0.5 * estimate @ (3.0 * np.eye(dim, dtype=matrix.dtype) - normed @ estimate_sq)

        scale = np.sqrt(trace)
        if rows >= cols:
            return matrix @ (estimate / scale)
        return (estimate @ matrix) / scale

    def step(self):
        self.step_count += 1
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data if isinstance(p.grad, Tensor) else p.grad
                grad = np.array(grad, dtype=p.data.dtype, copy=False)

                state = self.state.get(id(p), {})
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = np.zeros_like(p.data)

                buf = state['momentum_buffer']
                buf[:] = momentum * buf + grad
                update = grad + momentum * buf if nesterov else buf

                if weight_decay != 0:
                    p.data -= lr * weight_decay * p.data

                if p.data.ndim >= 2:
                    matrix_update, original_shape = self._as_matrix(update)
                    orthogonal_update = self._matrix_inverse_sqrt_orthogonalize(
                        matrix_update, steps=ns_steps, eps=eps
                    )
                    update = orthogonal_update.reshape(original_shape)

                p.data = p.data - lr * update
                self.state[id(p)] = state
