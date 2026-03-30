import numpy as np
import time 
from typing import Tuple,Optional,Dict,List
import os 
import sys 
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from Tensor import Tensor 
#internal constants for memory calculations
_BYTES_PER_FLOAT32 =4 #standard float32 size in bytes
_MB_TO_BYTES = 1024*1024 #megabytes to bytes conversion


class KVCache:
    """
    Efficient key-value cache for autoregressive generation.

    Stores K,V matrices for each transformer layer to avoid recomputation
    during sequential token generation.
    This is the critical optimization that makes production LLMs serving economically viable.

    KV Caching is designed ONLY for inference (generation), NOT training.
    - During generation:No gradients computed (model.eval() mode)
    - Cache operations use .data (no gradient tracking)

    Architecture:
         - Pre-allocates cache tensors with maximum sequence length
         - Tracks current sequence position for efficient O(1) updates
         - Provides update() method to append new K,V pairs without copying
         - Provides get() method to retrieve cached values for attention
         -  Handles multiple layers and attention heads properly

    Memory Layout:
    ```
    Layer 0: [Key_cache, Value_cache]  # Shape: (batch, num_heads, max_seq, head_dim)
    Layer 1: [Key_cache, Value_cache]
    ...
    Layer N: [Key_cache, Value_cache]
    ```
    """
    def __init__(self,batch_size:int,max_seq_len:int,num_layers:int,num_heads:int,head_dim:int):
        """
        Intialize KV cache for efficient generation

        Args:
            batch_size: Number of sequences to generate simultaneously
            max_seq_len: Maximum sequence length to support
            num_layers: Number of transformer layers
            num_heads: Number of attention heads per layer
            head_dim: Dimension of each attention head

           EXAMPLE:
        >>> cache = KVCache(batch_size=2, max_seq_len=128, num_layers=4,
        ...                 num_heads=8, head_dim=64)
        >>> cache.seq_pos  # 0 (no tokens cached yet)
        >>> len(cache.caches)  # 4 (one per layer)
        >>> cache.caches[0][0].shape  # (2, 8, 128, 64) - key cache for layer 0

        """
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim


        #current sequence position (how many tokens are cached)
        self.seq_pos = 0

        #cache storage: list of (key_cache,value_cache) tuples per layer
        self.caches = []

        for layer_idx in range(num_layers):
            #pre-allocating cache tensors with maximum size
            #shape : (batch_size,num_heads,max_seq_len,head_dim)
            key_cache = Tensor(np.zeros((batch_size,num_heads,max_seq_len,head_dim)))
            value_cache = Tensor(np.zeros((batch_size,num_heads,max_seq_len,head_dim)))

            self.caches.append((key_cache,value_cache))

    def update(self,layer_idx:int,key:Tensor,value:Tensor)->None:
        """
        Updating cache with new key-value pairs for a given layer.

        Args:
             layer_idx:Which transformer layer (0 to num_layers-1)
             key:New key tensor, shape (batch_size,num_heads,1,head_dim)
             value: New value tensor, shape (batch_size,num_heads,1,head_dim)

        Raises:
            ValueError: If layer_idx is out of range or sequence is full

        EXAMPLE:
        >>> cache = KVCache(batch_size=1, max_seq_len=10, num_layers=2,
        ...                 num_heads=4, head_dim=64)
        >>> new_k = Tensor(np.random.randn(1, 4, 1, 64))
        >>> new_v = Tensor(np.random.randn(1, 4, 1, 64))
        >>> cache.update(layer_idx=0, key=new_k, value=new_v)
        >>> cache.seq_pos  # Still 0 (update doesn't advance position)
        >>> cache.advance()
        >>> cache.seq_pos  # Now 1
        """
        if layer_idx >= self.num_layers:
            raise ValueError(
                f"Invalid layer index for cache update\n"
                f"   layer_idx={layer_idx} is out of range [0, {self.num_layers - 1}]\n"
                f"   KVCache was initialized with num_layers={self.num_layers}, so valid indices are 0 to {self.num_layers - 1}\n"
                f"   Check your transformer block loop: for layer_idx in range({self.num_layers})"
            )

        if self.seq_pos>= self.max_seq_len:
             raise ValueError(
                f"KV cache is full - cannot add more tokens\n"
                f"   Current position {self.seq_pos} has reached max_seq_len={self.max_seq_len}\n"
                f"   The cache was pre-allocated for {self.max_seq_len} tokens maximum. Autoregressive generation cannot exceed this limit.\n"
                f"   Either: (1) call cache.reset() to start a new sequence, or (2) create a larger cache with max_seq_len > {self.max_seq_len}"
            )

        #Get cache for this layer
        key_cache,value_cache = self.caches[layer_idx]

        #updating cache at current position (efficient O(1) write)
        # we use .data here because caching is inference-only 
        # This avoids gradient tracking overhead during generation
        key_cache.data[:,:,self.seq_pos:self.seq_pos+1,:] = key.data 
        value_cache.data[:,:,self.seq_pos:self.seq_pos+1,:] = value.data 

        # seq_los is advance externally via advance() after all layers process


    def get(self,layer_idx:int)->Tuple[Tensor,Tensor]:
        """
        Retrieves cached key-value pairs for attention computation


        Args:
            layer_idx :Which transformer layer to get cache for

        Returns:
            (cached_keys,cached_values): Tensors shaped for attention
            Keys:(batch_size,num_heads,seq_pos,head_dim)
            Value:(batch_size,num_heads,seq_pos,head_dim)


        EXAMPLE:
        >>> cache = KVCache(batch_size=1, max_seq_len=100, num_layers=2,
        ...                 num_heads=4, head_dim=64)
        >>> # After processing 3 tokens
        >>> cache.seq_pos = 3
        >>> cached_k, cached_v = cache.get(layer_idx=0)
        >>> cached_k.shape  # (1, 4, 3, 64) - only first 3 positions
        >>> cached_v.shape  # (1, 4, 3, 64)



        """
        #validating whether layer_idx is in range
        if layer_idx >= self.num_layers:
            raise ValueError(
                f"Invalid layer index for cache retrieval\n"
                f"   layer_idx={layer_idx} is out of range [0, {self.num_layers - 1}]\n"
                f"   KVCache was initialized with num_layers={self.num_layers}, so valid indices are 0 to {self.num_layers - 1}\n"
                f"   Check your transformer block loop: for layer_idx in range({self.num_layers})"
            )

        #get cache for this layer 
        key_cache,value_cache = self.caches[layer_idx]

        #Returns only the valid portion (up to current sequence position)
        #seq_pos tracks where to write next, so we have seq_pos valid tokens
        valid_len = self.seq_pos
        
        #Creating new Tensors from .data (no gradient tracking)
        cached_keys = Tensor(key_cache.data[:,:,:valid_len,:])
        cached_values = Tensor(value_cache.data[:,:,:valid_len:,:])

        return cached_keys ,cached_values

    def  advance(self) ->None:
        """
        Advancing sequence position after processing current token

        Call this after all layers have processed the curent token and updated their caches.
        This moves the write pointer forward
        """
        self.seq_pos += 1

    def reset(self)->None:
        """
        Resets cache for new generation sequence

        Call this when starting a new generation (new prompt)
        Resets the sequence position counter and optionally zero cache data
        """
        self.seq_pos = 0

        #zeroing out caches for clean state 
        for layer_idx in range(self.num_layers):
            key_cache,value_cache = self.caches[layer_idx]
            key_cache.data.fill(0.0)
            value_cache.data.fill(0.0)

    def get_memory_usage(self) ->Dict[str,float]:
        """
        Calculate memory usage of the cache system

        Returns:
             Dictionary with memory statistic in MB



        """
        #calculates size of one cache tensor
        cache_size = self.batch_size * self.num_heads* self.max_seq_len * self.head_dim

        #each layer has key_cache + value_cache
        total_cache_tensors = self.num_layers * 2
        total_elements = cache_size * total_cache_tensors
        total_bytes = total_elements * _BYTES_PER_FLOAT32
        total_mb = total_bytes / _MB_TO_BYTES

        return {
             'total_mb': total_mb,
            'per_layer_mb': total_mb / self.num_layers,
            'cache_tensors': total_cache_tensors,
            'total_elements': total_elements
        }
        



