import os
import sys 
import numpy as np


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiling import Profiler, analyze_weight_distribution,quick_profile
from Tensor import Tensor 
from layers.layers import Linear

def testing_latency_measurement():

    profiler = Profiler()

    #testing basic latency measurement
    test_model = Linear(8,4)
    test_input = Tensor(np.random.randn(4,8))
    latency = profiler.measure_latency(test_model,test_input,warmup=2,iterations=5)

    assert latency >= 0, f"Latency should be non-negative, got{latency}"
    assert latency < 1000, f"Latency seems too high from simple operation: {latency} ms"
    print(f"Basic Latency: {latency:.3f} ms")

    #testing measurement consistency 
    latencies = []
    for _ in range(3):
        lat = profiler.measure_latency(test_model,test_input,warmup=1,iterations=3)
        latencies.append(lat)

    #measurements should be in reasonable range
    avg_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    assert std_latency < avg_latency, "Standard deviation shouldn't exceed mean for simple operations"
    print(f"Consistency: {avg_latency:.3f}  ± {std_latency:.3f} ms ")

    #testing size scaling
    small_model = Linear(2,2)
    large_model = Linear(20,20)
    small_input = Tensor(np.random.randn(2,2))
    large_input = Tensor(np.random.randn(20,20))

    small_latency = profiler.measure_latency(small_model,small_input,warmup=1,iterations=3)
    large_latency = profiler.measure_latency(large_model,large_input,warmup=1,iterations=3)

    #larger operations might take longer 
    print(f"Scaling: Small {small_latency:.3f} ms, Large {large_latency:.3f} ms")

    print("Latency measurement works correctly")

if __name__ == "__main__":
    testing_latency_measurement()