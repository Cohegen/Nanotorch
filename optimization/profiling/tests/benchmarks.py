from multiprocessing import reduction
import os
import sys
from turtle import backward 
import numpy as np


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear


def benchmark_operation_efficiency():
    """
    This function compares efficiency of different operations for optimization guidance
    """

    profiler = Profiler()
    operations = []

    #test different  operation types
    size = 256
    input_tensor = Tensor(np.random.randn(32,size))

    #elementwise operations (memory-bound)
    #creating a simple model wrapper for elementwise operations
    class ElementwiseModel():
        def forward(self,x):
            return x+ x # simple elementwise operation

    elementwise_model =ElementwiseModel()
    elementwise_latency = profiler.measure_latency(elementwise_model,input_tensor,iterations=20)
    elementwise_flops = size * 32 #one operation per element

    operations.append({
        'operation':'Elementwise',
        'latency_ms':elementwise_latency,
        'flops':elementwise_flops,
        'gflops_per_second':(elementwise_flops /1e9) / (elementwise_latency / 1000),
        'efficiency_class':'memory-bound',
        'optimization_focus':'data_locality'
    })

    #matrix operations (compute-bound)
    matrix_model = Linear(size,size)
    matrix_latency = profiler.measure_latency(matrix_model,input_tensor,iterations=10)
    matrix_flops = size*size*2 #matrix multiplication

    operations.append({
        'operation':'Matrix Multiply',
        'latency_ms':matrix_latency,
        'flops':matrix_flops,
        'gflops_per_second':(matrix_flops / 1e9) /(matrix_latency / 1000),
        'efficiency_class':'compute-bound',
        'optimization_focus':'algorithms'
    })

    #Reduction operations (memory-bound)
    class ReductionModel:
        def forward(self,x):
            return x.sum() #sum reduction operation

    reduction_model = ReductionModel()
    reduction_latency = profiler.measure_latency(reduction_model,input_tensor,iterations=20)
    reduction_flops = size * 32 #sum reduction

    operations.append({
        'operation': 'Reduction',
        'latency_ms': reduction_latency,
        'flops': reduction_flops,
        'gflops_per_second': (reduction_flops / 1e9) / (reduction_latency / 1000),
        'efficiency_class': 'memory-bound',
        'optimization_focus': 'parallelization'
    })

    print("\nOperation Efficiency Comparison:")
    print("Operation\t\tLatency(ms)\tGFLOP/s\t\tEfficiency Class\tOptimization Focus")
    print("-" * 95)

    
    for op in operations:
        print(f"{op['operation']:<15}\t{op['latency_ms']:.3f}\t\t"
              f"{op['gflops_per_second']:.2f}\t\t{op['efficiency_class']:<15}\t{op['optimization_focus']}")


    print("\n Operation Optimization Insights:")

    #finding most and least efficient
    best_op = max(operations,key=lambda x:x['gflops_per_second'])
    worst_op = min(operations,key=lambda x:x['gflops_per_second'])

    print(f"Most efficient: {best_op['operation']} ({best_op['gflops_per_second']:.2f} GFLOP/s)")
    print(f"Least efficient: {worst_op['operation']} ({worst_op['gflops_per_second']:.2f} GFLOP/s)")

    #count operation types
    memory_bound_ops = [op for op in operations if op['efficiency_class'] == 'memory-bound']
    compute_bound_ops = [op for op in operations if op['efficiency_class'] == 'compute-bound']

    print(f"\n Optimization Priority: ")
    if len(memory_bound_ops) > len(compute_bound_ops):
        print("Focus on memory optimization: data locality, bandwidth, caching")
    else:
        print("Focus on compute optimization: better algorithms, vectorization")

def profiling_analysis_overhead():
    """
    Measuring the overheard of profiling itself
    """

    #testing with and without profiling
    test_tensor = Tensor(np.random.randn(100,100))
    iterations = 50

    import time
    #without profiling -baseline measurement
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = test_tensor.data.copy() #simple operation
    end_time = time.perf_counter()
    baseline_ms = (end_time - start_time)*1000

    #with profiling which includes measurement overhead 
    profiler = Profiler()
    #create a simple model for profiling overhead measurement
    class TestModel:
        def forward(self,x):
            return x+1.0

    test_model = TestModel()
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = profiler.measure_latency(test_model,test_tensor,warmup=1,iterations=1)
    end_time =time.perf_counter()
    profiled_ms = (end_time - start_time)*1000

    overhead_factor = profiled_ms / max(baseline_ms,0.001)

    
    print(f"\nProfiling Overhead Analysis:")
    print(f"Baseline execution: {baseline_ms:.2f} ms")
    print(f"With profiling: {profiled_ms:.2f} ms")
    print(f"Profiling overhead: {overhead_factor:.1f}× slower")

    print(f"\n Profiling Overhead Insights: ")
    if overhead_factor < 2:
        print("Low overhead - suitable for frequent profiling")
    elif overhead_factor < 10:
        print("Moderate overhead - use for development and debugging")
    else:
        print("High overhead - use sparingly in production")


#run optimization analysis
if __name__ == "__main__":
    benchmark_operation_efficiency()
    profiling_analysis_overhead()
