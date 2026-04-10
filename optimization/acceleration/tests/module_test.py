import os 

from re import A
import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from acceleration import fused_gelu,vectorized_matmul,DEFAULT_TILING_ITERATIONS, DEFAULT_WARMUP_ITERATIONS, tiled_matmul
import numpy as np

def module_test():
    """
    This function intends to test whether the acceleration module
    functionalities work correctly.
    """

    #creating a realistic model scenario
    batch_size,seq_len,hidden_dim = 16,64,256
    print(f"   Model config: batch={batch_size}, seq_len={seq_len}, hidden={hidden_dim}")

    # Testing  data
    x = Tensor(np.random.randn(batch_size, seq_len, hidden_dim).astype(np.float32))
    weight = Tensor(np.random.randn(hidden_dim, hidden_dim).astype(np.float32))
    print(f"   Input tensor: {x.shape}, Weight tensor: {weight.shape}")

    # Testing complete pipeline: reshape → matmul → activation
    print("   Testing vectorized operations...")

    # Reshaping for matrix multiplication (flatten batch and sequence)
    x_reshaped = Tensor(x.data.reshape(-1, hidden_dim))
    assert x_reshaped.shape == (batch_size * seq_len, hidden_dim)
   
   #vectorized matrix multiplication
    linear_output = vectorized_matmul(x_reshaped,weight)
    assert linear_output.shape == (batch_size*seq_len,hidden_dim)
    print(f"    Matrix multiplication: {x_reshaped.shape} @ {weight.shape} → {linear_output.shape}")

    #fused activation
    activated = fused_gelu(linear_output)
    assert activated.shape == linear_output.shape 
    print(f"   Fused GELU activation: {linear_output.shape}-> {activated.shape}")

      # Reshape back to original structure
    final_output = Tensor(activated.data.reshape(batch_size, seq_len, hidden_dim))
    assert final_output.shape == x.shape
    print(f"   Output reshape: {activated.shape} → {final_output.shape}")

    class TransformerBlock:
        def __init__(self,hidden_dim):
            self.hidden_dim = hidden_dim
            self.weight1 = Tensor(np.random.randn(hidden_dim, hidden_dim).astype(np.float32))
            self.weight2 = Tensor(np.random.randn(hidden_dim, hidden_dim).astype(np.float32))
            self.weight1.grad = None
            self.weight2.grad = None

        def __call__(self, x):
            # Simulating transformer block: linear → activation → linear
            batch_size, seq_len, hidden_dim = x.shape
            x_flat = Tensor(x.data.reshape(-1, hidden_dim))

             # First linear layer
            h1 = vectorized_matmul(x_flat, self.weight1)
            h1_activated = fused_gelu(h1)

            # Second linear layer
            h2 = vectorized_matmul(h1_activated, self.weight2)

            # Reshape back
            output = Tensor(h2.data.reshape(batch_size, seq_len, hidden_dim))
            return output

        def parameters(self):
            return [self.weight1,self.weight2]

    #intializing model and testing forward pass
    model =TransformerBlock(hidden_dim)
   
    print(f"   Model parameters: {len(model.parameters())}")

    # Testing  model forward pass with accelerated operations
    print("   Testing model forward pass with accelerated operations...")
    output = model(x)
    assert output.shape == x.shape
    print(f"    Model forward pass: {x.shape} → {output.shape}")

    # Verifying accelerated operations provide correct results
    print("   Validating numerical correctness...")
    # Checking  output is finite and has reasonable values
    assert np.all(np.isfinite(output.data)), "Model output contains NaN or Inf"
    output_mean = np.mean(np.abs(output.data))
    # Random initialization can produce larger values  verify reasonable range
    assert output_mean < 1000.0, f"Output values unreasonably large: {output_mean}"
    print(f"    Numerical validation passed (mean magnitude: {output_mean:.4f})")

    print("   Testing performance characteristics...")

    # Verify acceleration provides measurable benefits
    import time
    test_sizes = [128, 256]
    for size in test_sizes:
        test_x = Tensor(np.random.randn(size, size).astype(np.float32))
        test_y = Tensor(np.random.randn(size, size).astype(np.float32))

        # Time operations and verify reasonable performance
        start = time.time()
        _ = vectorized_matmul(test_x, test_y)
        matmul_time = time.time() - start

        start = time.time()
        _ = fused_gelu(test_x)
        gelu_time = time.time() - start

        # Verify operations complete in reasonable time
        assert matmul_time < 1.0, f"Matrix multiplication too slow: {matmul_time:.3f}s"
        assert gelu_time < 0.1, f"GELU activation too slow: {gelu_time:.3f}s"

        print(f"    Size {size}: matmul={matmul_time*1000:.1f}ms, gelu={gelu_time*1000:.1f}ms")

if __name__ == "__main__":
    module_test()



