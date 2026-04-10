
import numpy as np
import os
import sys
import importlib.util

_script_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_script_dir)
sys.path.insert(0, _parent_dir)

from Tensor import Tensor

# Load autograd.py from same directory (folder is not a package)
_spec = importlib.util.spec_from_file_location("autograd", os.path.join(_script_dir, "autograd.py"))
_autograd_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_autograd_module)
_autograd_module.enable_autograd(quiet=True)


def testing_tensor_autograd():
    print("Tensor Autograd Enhancement...")

    def _grad_array(t):
        g = t.grad
        if g is None:
            return None
        if hasattr(g, 'data'):
            return np.asarray(g.data, dtype=np.float32)
        return np.asarray(g, dtype=np.float32)

    #test simple gradient computation
    x = Tensor([2.0],requires_grad=True)
    y = x*3
    z = y+1 # z= 3x+1, sodz/dx= 3

    z.backward()
    assert np.allclose(_grad_array(x), [3.0]), f"Expected [3.0], got {x.grad}"

    #test matrix multiplication gradients
    a = Tensor([[1.0,2.0]],requires_grad=True)
    b = Tensor([[3.0],[4.0]],requires_grad=True)
    c = a.matmul(b)

    c.backward()
    assert np.allclose(_grad_array(a), [[3.0, 4.0]]), f"Expected [[3.0, 4.0]], got {a.grad}"
    assert np.allclose(_grad_array(b), [[1.0], [2.0]]), f"Expected [[1.0], [2.0]], got {b.grad}"

    #test computation graph
    x = Tensor([1.0,2.0],requires_grad=True)
    y = x*2
    z = y.sum()

    z.backward()
    assert np.allclose(_grad_array(x), [2.0, 2.0]), f"Expected [2.0, 2.0], got {x.grad}"

    print(" Tensor autograd enhancement works correctly!")

if __name__ == "__main__":
    testing_tensor_autograd()
