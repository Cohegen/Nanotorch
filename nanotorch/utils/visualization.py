"""Utilities for visualizing the NanoTorch computational graph."""

import os
from Tensor import Tensor

def make_dot(root):
    """
    Generates a Graphviz DOT representation of the computational graph.
    Args:
        root: The Tensor to start the traversal from (usually the loss).
    Returns:
        A string containing the DOT source code.
    """
    dot_lines = [
        "digraph G {",
        "  rankdir=LR;",
        "  node [fontname=\"Arial\"];"
    ]
    
    seen_nodes = set()
    
    def get_id(obj):
        return str(id(obj))
    
    def add_nodes(tensor):
        if tensor in seen_nodes:
            return
        seen_nodes.add(tensor)
        
        # Tensor node
        label = f"Tensor\nshape={tensor.shape}"
        if hasattr(tensor, 'requires_grad') and tensor.requires_grad:
            color = "lightblue"
            if tensor.grad is not None:
                label += "\nhas_grad=True"
        else:
            color = "lightgrey"
            
        dot_lines.append(f"  {get_id(tensor)} [label=\"{label}\", style=filled, fillcolor={color}, shape=ellipse];")
        
        # Follow the gradient function
        grad_fn = getattr(tensor, '_grad_fn', None)
        if grad_fn is not None:
            fn_id = get_id(grad_fn)
            if fn_id not in seen_nodes:
                seen_nodes.add(fn_id)
                fn_name = grad_fn.__class__.__name__
                dot_lines.append(f"  {fn_id} [label=\"{fn_name}\", shape=box, style=filled, fillcolor=orange];")
            
            # Edge from function to output tensor
            dot_lines.append(f"  {fn_id} -> {get_id(tensor)};")
            
            # Follow saved tensors (inputs to the function)
            for input_tensor in grad_fn.saved_tensors:
                if isinstance(input_tensor, Tensor):
                    dot_lines.append(f"  {get_id(input_tensor)} -> {fn_id};")
                    add_nodes(input_tensor)

    add_nodes(root)
    dot_lines.append("}")
    return "\n".join(dot_lines)

def save_graph(root, filename="computational_graph.dot"):
    """Saves the graph as a .dot file."""
    dot_code = make_dot(root)
    with open(filename, "w") as f:
        f.write(dot_code)
    print(f"Graph saved to {filename}. You can visualize it at https://dreampuf.github.io/GraphvizOnline/")
