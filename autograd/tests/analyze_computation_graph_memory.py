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

def analyze_computation_graph_memory():
    """
    This function demonstrates memory overhead of computation graphs.
    """
    print("Analyzing Computational Graph Memory")
    print("="*60)

    import sys 

    #creating tensors with different sizes
    sizes =[(100,100),(500,500),(1000,1000)]

    print("\nMemory comparison: With vs Without gradient tracking")
    print("-"*60)

    for shape in sizes:
        #without gradient tracking
        x_no_grad = Tensor(np.random.randn(*shape))
        base_memory= x_no_grad.data.nbytes

        #with gradient tracking
        x_with_grad = Tensor(np.random.randn(*shape),requires_grad=True)
        y = x_with_grad * 2
        z = y + 1

        #estimate graph overhead: saved tensors in grad_fn
        graph_overhead =0
        if hasattr(z,'_grad_fn') and z._grad_fn is not None:
            for tensor in z._grad_fn.saved_tensors:
                if isinstance(tensor,Tensor):
                    graph_overhead += tensor.data.nbytes

        print(f"\nShape {shape}:")
        print(f"   Base tensor:{base_memory / 1024:.1f} kb")
        print(f" Graph overhead:  {graph_overhead/ 1024:.1f} KB")
        print(f"  Overhead ratio : {(graph_overhead /base_memory):.1f}x")

        print("\n"+ "="* 60)

if __name__ =='__main__':
    analyze_computation_graph_memory()