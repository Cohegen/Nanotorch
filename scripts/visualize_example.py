import sys
import os
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Tensor import Tensor
from autograd.autograd import enable_autograd
from nanotorch.nn.modules.linear import Linear
from nanotorch.nn.modules.activation import ReLU
from nanotorch.utils.visualization import save_graph

# Enable autograd
enable_autograd()

def main():
    # 1. Create a simple network
    model = Linear(10, 5)
    relu = ReLU()
    
    # 2. Forward pass
    x = Tensor(np.random.randn(1, 10), requires_grad=True)
    out = relu(model(x))
    
    # 3. Create a scalar loss
    target = Tensor(np.random.randn(1, 5))
    loss = ((out - target) * (out - target)).sum()
    
    print("Forward pass complete.")
    
    # 4. Save the graph
    save_graph(loss, "example_graph.dot")
    
    # 5. Run backward (optional, helps show grad status)
    loss.backward()
    print("Backward pass complete.")
    
    # Save again with gradient info
    save_graph(loss, "example_graph_after_backward.dot")

if __name__ == "__main__":
    main()
