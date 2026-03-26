import os 

import sys

##a command to access all directories within nanotorch
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing dependencies from other modules
from Tensor import Tensor
from layers.layers import Linear,Sequential
from activations.activations import ReLU
from compression import measure_sparsity,magnitude_prune,structured_prune,low_rank_approximate,KnowledgeDistillation,compress_model
from optimization.profiling.profiling import Profiler
import numpy as np

def module_testing():

    #creating a realistic model
    input_layer = Linear(784,512)
    hidden1 = Linear(512,256)
    hidden2 = Linear(256,128)
    output_layer= Linear(128,10)
    model =Sequential(input_layer,hidden1,hidden2,output_layer)

    original_params =sum(p.size for  p in model.parameters())
    print(f"Original model: {original_params:,} parameters")

    
    # Apply comprehensive compression - students see each technique
    compression_config = {
        'magnitude_prune': 0.8,    # Remove 80% of smallest weights
        'structured_prune': 0.3     # Remove 30% of channels
    }

    stats = compress_model(model, compression_config)
    final_sparsity = measure_sparsity(model)

    # Validate compression results
    assert final_sparsity > 70, f"Expected >70% sparsity, got {final_sparsity:.1f}%"
    assert stats['sparsity_increase'] > 70, "Should achieve significant compression"
    assert len(stats['applied_techniques']) == 2, "Should apply both techniques"

    print(f"✅ Achieved {final_sparsity:.1f}% sparsity with {len(stats['applied_techniques'])} techniques")

    # Test 2: Knowledge distillation setup
    print("🧪 Integration Test: Knowledge distillation...")

    # Create teacher with more capacity - explicit layers show architecture
    teacher_l1 = Linear(100, 200)
    teacher_l2 = Linear(200, 50)
    teacher = Sequential(teacher_l1, teacher_l2)

    # Create smaller student - explicit shows size difference
    student_l1 = Linear(100, 50)
    student = Sequential(student_l1)  # 3x fewer parameters

    kd = KnowledgeDistillation(teacher, student, temperature=4.0, alpha=0.8)

    # Verify setup
    teacher_params = sum(p.size for p in teacher.parameters())
    student_params = sum(p.size for p in student.parameters())
    compression_ratio = student_params / teacher_params

    assert compression_ratio < 0.5, f"Student should be <50% of teacher size, got {compression_ratio:.2f}"
    assert kd.temperature == 4.0, "Temperature should be set correctly"
    assert kd.alpha == 0.8, "Alpha should be set correctly"

    print(f"✅ Knowledge distillation: {compression_ratio:.2f}x size reduction")

    # Testing Low-rank approximation
    print("🧪 Integration Test: Low-rank approximation...")

    large_matrix = np.random.randn(200, 150)
    U, S, V = low_rank_approximate(large_matrix, rank_ratio=0.3)

    original_size = large_matrix.size
    compressed_size = U.size + S.size + V.size
    compression_ratio = compressed_size / original_size

    assert compression_ratio < 0.7, f"Should achieve compression, got ratio {compression_ratio:.2f}"

    # Testing reconstruction
    reconstructed = U @ np.diag(S) @ V
    error = np.linalg.norm(large_matrix - reconstructed) / np.linalg.norm(large_matrix)
    # Low-rank approximation trades accuracy for compression - some error is expected
    assert error < 0.7, f"Reconstruction error too high: {error:.3f}"

    print(f" Low-rank: {compression_ratio:.2f}x compression, {error:.3f} error")
