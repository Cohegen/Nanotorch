from re import M
import sys 
import os 
import time 
import numpy as np
import tracemalloc
from typing import Dict,List,Any,Optional,Tuple 
from collections import defaultdict 
import gc



sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

#importing from previous modules 
from Tensor import Tensor
from layers.layers import Linear 
from convolution.convolutions import Conv2d

#constants for memory and performance measurement
BYTES_PER_FLOAT32 = 4 #standard float32 size in bytees 
KB_TO_BYTES = 1024 #kilobytes to bytes conversion
MB_TO_BYTES = 1024 * 1024 # megabytes to the bytes conversion

class Profiler:
    """
    A profiler that meaures parameters,FLOPS, memory usage and latency with statistica
    rigor.
    """
    def __init__(self):
        """
        Intializes profiler with measurement state
        """
        self.measurement = {}
        self.operation_counts = defaultdict(int)
        self.memory_tracker = None

    def _count_layer_parameters(self,layer)-> int:
        """
        Counts parameters in a single layer by inspecting
        weight and bias attributes

        
        Args:
           layer : A layer object with .weights (and optionally .bias)

        Returns:
           int:Total parameter count for this layer.

        """
        params = 0
        if hasattr(layer,'weight'):
            params += layer.weight.data.size
            if hasattr(layer,'bias') and layer.bias is not None:
                params += layer.bias.data.size
        return params 

    def count_parameters(self,model)->int:
        """
        Count total trainable parameter in a model
        """
        if hasattr(model,'layers'):
            return sum(p.data.size for layer in model.layers for p in layer.parameters())

        elif hasattr(model,'parameters'):
            return sum(p.data.size for p in model.parameters())
        
        elif hasattr(model,'weight'):
            return self._count_layer_parameters(model)
        return 0

    def _count_linear_flops(self,model,input_shape:Tuple[int,...])->int:
        """
        Counts FLOPs for a Linear layer forward pass

          Linear FLOP Formula:
        FLOPs = in_features × out_features × 2
                     ↑              ↑          ↑
              Input dimension  Output dimension  Multiply + Add

        Args: 
           model:A linear layer with .weight attribute
           input_shape:Input tensor shape (batch,in_features)

        Returns:
             int:FLOP count for one forward pass (batch-independent)

            
        """

        in_features = input_shape[-1]
        out_features = model.weight.shape[1] if hasattr(model,'weight') else 1 
        return in_features *out_features * 2

    def _count_conv_flops(self,model,input_shape:Tuple[int,...])->int:
        """
        Count FLOps for a Conv2d layer forward pass 

        Conv2d FLOP Formula:
        FLOPs = out_H × out_W × kernel_H × kernel_W × in_C × out_C × 2
                  ↑       ↑        ↑          ↑         ↑       ↑      ↑
              Output spatial    Kernel spatial     Channel dims   Mul+Add

        Args:
            model:A Conv2d layer with kernel_szie,in_channels,out_channels
            input_shape:Input tensor shape (batch,channels,height,width)

        Returns:
            int:FLOP count for one forward pass
        """

        if not(hasattr(model,'kernel_size')and hasattr(model,'in_channels')):
            return 0

        in_channels = model.in_channels
        out_channels = model.out_channels
        kernel_h = kernel_w = model.kernel_size

        input_h,input_w = input_shape[-2],input_shape[-1]
        stride = model.stride if hasattr(model,'stride') else 1
        output_h = input_h // stride 
        output_w = input_w //stride 

        return output_h*output_w*kernel_w*kernel_h*in_channels*out_channels* 2

    def _count_sequential_flops(self,model,input_shape:Tuple[int,...])->int:
        """
        Counts FLOPs for a Sequential model by summing per-layer FLOPs.


         ```
        Sequential FLOP Accumulation:
        Layer 1 FLOPs + Layer 2 FLOPs + ... + Layer N FLOPs = Total FLOPs
             ↓               ↓                    ↓
          Shape propagated through each layer
        ```

        Args:
           model:A model with .layers attribute (list of layer)
           input_shape:Input tensor shape for first layer

        Returns:
            int: Total FLOP count across all layers.

        """
        total_flops = 0
        current_shape = input_shape 
        for layer in model.layers:
            total_flops += self.count_flops(layer,current_shape)
            if hasattr(layer,'weight'):
                current_shape =current_shape[:-1] + (layer.weight.shape[1],)

        return total_flops


    def count_flops(self,model,input_shape:Tuple[int,...])-> int:
        """
        Counts FLOPS for oe forward pass

        """
        model_name = model.__class__.__name__

        if model_name == 'Linear':
            return self._count_linear_flops(model,input_shape)
        elif  model_name == 'Conv2d':
            return self._count_conv_flops(model,input_shape)
        elif model_name == 'Sequential' or hasattr(model,'layers'):
            return self._count_sequential_flops(model,input_shape)
        else:
            return int(np.prod(input_shape))

    def _calculate_parameter_memory(self,model)->float:
        """
        Calculate memory used by model parmeters in megabytes

        ```
        Parameter Memory Formula:
        Memory (MB) = parameter_count × 4 bytes / (1024 × 1024)
                           ↑              ↑
                     From count_parameters  FP32 size
        ```

        Args:
           model:Model to analyze

        Returns:
           float:Paraeter memory in megabytes
        """
        param_count = self.count_parameters(model)
        return (param_count*BYTES_PER_FLOAT32) / MB_TO_BYTES

    def _calculate_memory_efficiency(self,useful_memory_mb:float,peak_memory_mb:float)->float:
        """
        Calculates memory efficiency as ration of useful to total memory.


        ```
        Efficiency = useful_memory / peak_memory
                         ↑               ↑
              Parameters + Activations   tracemalloc peak

        Ideal: 1.0 (all memory is useful)
        Typical: 0.3-0.8 (overhead from allocator, fragmentation)
        ```

        Args: 
            useful_memory_mb : sum of parameter + activation memory
            peak_memory_mb : peak memory_observed by tracemalloc

        Returns:
            float: Effiencency ratio clampled to [0,1]

        """
        ratio = useful_memory_mb /max(peak_memory_mb,0.001)
        return min(ratio,1.0)

    def measure_memory(self,model,input_shape:Tuple[int,...])->Dict[str,float]:
        """
        Measures memory usage during forward pass.

        """
        tracemalloc.start()
        _baseline_memory = tracemalloc.get_traced_memory()[0]

        parameter_memory_mb = self._calculate_parameter_memory(model)

        dummy_input = Tensor(np.random.randn(*input_shape))
        activation_memory_mb = (dummy_input.data.nbytes*2) / MB_TO_BYTES

        _ = model.forward(dummy_input)

        _current_memory,peak_memory = tracemalloc.get_traced_memory()
        peak_memory_mb = (peak_memory- _baseline_memory) /MB_TO_BYTES

        useful_memory = parameter_memory_mb + activation_memory_mb
        return {
            'parameter_memory_mb':parameter_memory_mb,
            'activation_memory_mb':activation_memory_mb,
            'peak_memory_mb':max(peak_memory_mb,useful_memory),
            'memory_efficiency':self._calculate_memory_efficiency(useful_memory,peak_memory_mb)

        }

    def measure_latency(self,model,input_tensor,warmup:int=10,iterations:int=100)-> float:
        """
        Measures  model inference latency with statistical rigor


        Params:
           warmup:number of warmup runs (default 10)
           iterations:Number of measrurement runs(default 100)

        """ 

        #warmup runs to stabilize performance
        for _ in range(warmup):
            _ = model.forward(input_tensor)

        #measurement runs 
        times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            _ = model.forward(input_tensor)
            end_time = time.perf_counter()
            times.append((end_time-start_time)*1000)#converts to miliseconds

        #calculates statistics - uses media for robustness
        times = np.array(times)
        median_latency = np.median(times)

        return float(median_latency)

    def profile_layer(self,layer,input_shape:Tuple[int,...])->Dict[str,Any]:
        """
        Profiles a single layer comprehensively
        """

        #creating a dummy input for latency measurement
        dummy_input = Tensor(np.random.randn(*input_shape))

        #gathering all measurements
        params = self.count_parameters(layer)
        flops = self.count_flops(layer,input_shape)
        memory = self.measure_memory(layer,input_shape)
        latency = self.measure_latency(layer,dummy_input,warmup=3,iterations=10)

        #computes derived metrics 
        gflops_per_second = (flops / 1e9) / max(latency/1000,1e-6)


        return {
            'layer_type': layer.__class__.__name__,
            'parameters': params,
            'flops': flops,
            'latency_ms': latency,
            'gflops_per_second': gflops_per_second,
            **memory
        }

    def _compute_derived_metrics(self,flops:int,latency_ms:float,peak_memory_mb:float)->Dict[str,float]:
        """
        Computes throughput and efficiency metrics from raw measurements.

        ```
        Derived Metrics Pipeline:
        FLOPs + Latency → GFLOP/s (throughput)
        Memory + Latency → MB/s (bandwidth)
        GFLOP/s / Peak → Efficiency (utilization)
        ```

        Args:
            flops:total floating point operations
            latency_ms:measured latency in milliseconds
            peak_memory_mb :Peak memory usage in megabytes

        Returns:
            dict with gflops_per_second,memory_bandwidth_mbs, computational_efficiency
        """
        latency_seconds = latency_ms/1000.0
        gflops_per_second = (flops /1e9) / max(latency_seconds,1e-6)
        memory_bandwidth = peak_memory_mb / max(latency_seconds,1e-6)
        theoretical_peak_gflops = 100.0
        computational_efficiency = min(gflops_per_second/theoretical_peak_gflops,1.0)

        return {
            'gflops_per_second': gflops_per_second,
            'memory_bandwidth_mbs': memory_bandwidth,
            'computational_efficiency': computational_efficiency
        }

    def _analyze_bottleneck(self,gflops_per_second:float,memory_bandwidth_mbs:float)->Dict[str,Any]:
        """
        Identity whether workload is memory-bound or compute-bound 

         ```
        Bottleneck Decision:
        If bandwidth >> GFLOP/s × 100 → Memory-bound (data movement dominates)
        Otherwise                      → Compute-bound (arithmetic dominates)
        ```

        Args:
            gflops_per_second: Compute throughput
            memory_bandwidth_mbs: Memory bandwidth in MB/s

        Returns:
            dict with is_memory_bound, is_compute_bound, bottleneck label
        """
        is_memory_bound = memory_bandwidth_mbs > gflops_per_second * 100
        return {
             'is_memory_bound': is_memory_bound,
            'is_compute_bound': not is_memory_bound,
            'bottleneck': 'memory' if is_memory_bound else 'compute'
        }

    def profile_forward_pass(self,model,input_tensor)->Dict[str,Any]:
        """
        Comprehensive profiling of a model's forward pass
        """
        param_count = self.count_parameters(model)
        flops= self.count_flops(model,input_tensor.shape)
        memory_stats = self.measure_memory(model,input_tensor.shape)
        latency_ms = self.measure_latency(model,input_tensor,warmup=5,iterations=20)

        derived = self._compute_derived_metrics(flops,latency_ms,memory_stats['peak_memory_mb'])
        bottleneck = self._analyze_bottleneck(
            derived['gflops_per_second'],
            derived['memory_bandwidth_mbs']
        )

        return {
            'parameters': param_count, 'flops': flops, 'latency_ms': latency_ms,
            **memory_stats, **derived, **bottleneck
        }

    def _estimate_backward_costs(self,forward_flops:int,forward_latency_ms:float)-> Dict[str,float]:
        """
        Estimate backward pass compute costs from forward pass measurements.

        ```
        Backward Pass Cost Estimation:
        Backward FLOPs   = Forward FLOPs × 2   (gradient computation)
        Backward Latency = Forward Latency × 2 (more complex operations)

        Why 2×? Each operation needs:
        1. Gradient w.r.t. weights (same cost as forward)
        2. Gradient w.r.t. inputs (same cost as forward)
        ```

        Args: 
             forward_flops:FLOP count from forward pass
             forward_latency_ms:Latency from forward pass

        Returns: 
             dict with backward_flops and backward_latency_ms
        """

        return {
            'backward_flops': forward_flops * 2,
            'backward_latency_ms': forward_latency_ms * 2
        }

    def _estimate_optimizer_memory(self,gradient_memory_mb:float)->Dict[str,float]:
        """
        Estimate additional memory required by different optimizers

        ```
        Optimizer Memory Requirements:
        ┌───────────┬────────────────────────────────────┐
        │ Optimizer │ Extra Memory                       │
        ├───────────┼────────────────────────────────────┤
        │ SGD       │ 0× (no state)                      │
        │ Adam      │ 2× gradient memory (m + v)         │
        │ AdamW     │ 2× gradient memory (m + v)         │
        └───────────┴────────────────────────────────────┘
        ```

        Args:
            gradient_memory_mb: memory for gradient in MB

        Returns:
            dict mapping Optimizer name to extra memory in  MB
        """

        return {
            'sgd':0,
            'adam':gradient_memory_mb*2,
            'adamw':gradient_memory_mb*2,
        }

    def profile_backward_pass(self,model,input_tensor,_loss_fn=None)->Dict[str,Any]:
        """
        Profiles both forward and backward passes for training analysis
        """
        fwd = self.profile_forward_pass(model,input_tensor)
        bwd = self._estimate_backward_costs(fwd['flops'],fwd['latency_ms'])

        gradient_memory_mb = fwd['parameter_memory_mb']
        total_flops = fwd['flops'] + bwd['backward_flops']
        total_latency_ms = fwd['latency_ms'] + bwd['backward_latency_ms']
        total_memory_mb = fwd['parameter_memory_mb'] + fwd['activation_memory_mb']
        
        return {
            'forward_flops': fwd['flops'],
            'forward_latency_ms': fwd['latency_ms'],
            'forward_memory_mb': fwd['peak_memory_mb'],
            **bwd,
            'gradient_memory_mb': gradient_memory_mb,
            'total_flops': total_flops,
            'total_latency_ms': total_latency_ms,
            'total_memory_mb': total_memory_mb,
            'total_gflops_per_second': (total_flops / 1e9) / (total_latency_ms / 1000.0),
            'optimizer_memory_estimates': self._estimate_optimizer_memory(gradient_memory_mb),
            'memory_efficiency': fwd['memory_efficiency'],
            'bottleneck': fwd['bottleneck']
        }


"""
Helper functions that provide simplified interfaces for common
profiling task as the ones below.
They make it easy to quickly profile models and analyze characteristics without manually
calling multiple profiler methods.
"""
    
def quick_profile(model,input_tensor,profiler=None):
    """
    Quick profiling function for immediate insights.

    Args:
        model:Model to profile
        input_tensor:Input data for profiling
        profiler:Optional Profiler instance (creates new one if None)

    Returns:
         dict:Profile results with key metrics

    """
    if profiler is None:
        profiler = Profiler()

    profile = profiler.profile_forward_pass(model,input_tensor)

    #displays format results 
    print(" Quick Profile Results:")
    print(f"    Parameters:{profile['parameters']:,}")
    print(f"    FLOPs: {profile['flops']:,}")
    print(f"    Latency:{profile['latency_ms']:.2f} ms")
    print(f"    Memory: {profile['peak_memory_mb']:.2f} MB")
    print(f"    Bottleneck: {profile['bottleneck']}")
    print(f"    Efficiency:  {profile['computational_efficiency']*100:.1f}%")

    return profile 

def analyze_weight_distribution(model,percentiles=[10,25,50,75,90]):
    """
    Analyzes weight distribution across layers

    This helper function helps in understanding how weights are distributed across layers.
    It is useful for identifying patterns in parameter magnitudes.

    Args:
        model:model to analyze
        percentiles:List of percentiles to compute

    Returns:
        dict:Weight distribution statistics

    """
    weights = []
    if hasattr(model,'parameters'):
        for param in model.parameters():
            weights.extend(param.data.flatten().tolist())
    elif hasattr(model,'weight'):
        weights.extend(model.weight.data.flatten().tolist())
    else:
        return {'error':'No weights found'}

    weights = np.array(weights)
    abs_weights = np.abs(weights)

    #calculating stats 
    stats = {
        'total_weights': len(weights),
        'mean': float(np.mean(abs_weights)),
        'std': float(np.std(abs_weights)),
        'min': float(np.min(abs_weights)),
        'max': float(np.max(abs_weights)),
    }

    #percentile analysis
    for p in percentiles:
        stats[f'percentile_{p}'] = float(np.percentile(abs_weights,p))

    #threshold analysis 
    for threshold in [0.001,0.01,0.1]:
        below = np.sum(abs_weights < threshold) /len(weights)*100
        stats[f'below_threshold_{str(threshold).replace(".","")}'] = below

    return stats 
