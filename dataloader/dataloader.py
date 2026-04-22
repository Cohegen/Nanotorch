from multiprocessing import Value
import numpy as np
import time
import sys
from typing import Iterator, Tuple, List,Optional,Union
from abc import ABC, abstractmethod
import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
class Dataset(ABC):
    """
    Abstract base class for all datasets
    """
    @abstractmethod
    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset

        This method must be implemented by all subclasses to enable
        len(dataset) calls and batch size calculations.
        """
        pass

    @abstractmethod
    def __getitem__(self,idx:int):
        """
        Returns the sample at the given index

        Args:
            idx: the index of the sample to retrieve (0<=idx<len(dataset))

        Returns:
           The sample at index idx. 
           Could be (data,label) tuple, single tensor etc.

        """
        pass
    
class TensorDataset(Dataset):
    """
    Dataset wrapping tensors for supervised learning

    Each sample is a tuple of tensors from the same index across all input tensors
    All tensors must have the same size in their first dimension.
    """

    def __init__(self,*tensors):
        """
        Create dataset from multiple tensors.

        Args:
            *tensors: variable number of tensor objects

         All tensors must have the same size in their first dimension   
        """

        assert len(tensors) > 0, "Must provide at leaast one tensor"

        #store all tensors
        self.tensors = tensors

        #validating all tensors have the same first dimension
        first_size = len(tensors[0].data) #size of first dimension
        for i,tensor in enumerate(tensors):
            if len(tensor.data) != first_size:
                raise ValueError(
                    f"All tensors must have same size in first dimension."
                    f"Tensor 0: {first_size},Tensor{i}:{len(tensor.data)}"
                )

    def __len__(self) -> int:
        """
        Return number of samples (size of first dimension)

        """
        return len(self.tensors[0].data)

      
    def __getitem__(self,idx:int) ->Tuple[Tensor, ...]:
        
        """Returns tuple of tensor slices at given index"""

        if idx >= len(self) or idx < 0:
            raise IndexError(f"Index {idx} out of range for the dataset of size {len(self)}")

        #returns tuple of slices from all tensors
        return tuple(Tensor(tensor.data[idx]) for tensor in self.tensors)

class Dataloader:
    """
    Dataloader with batching and shuffling support

    Wraps a dataset to provide batched iteration with optional shuffling
    Essential for efficient training with mini-batch gradient descent.
    """

    def __init__(self,dataset:Dataset,batch_size,shuffle:bool = False):
        """
        Creating dataloader for batched iteration

        Args:
             dataset: Dataset to load from
             batch_size : Number of samples per batch
             shuffle: whether to shuffle data each epoch

        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle


    def __len__(self) -> int:
        """
        Return number of batches per epoch
        """
        #calculates number of complete batches
        return (len(self.dataset) + self.batch_size -1) // self.batch_size

    def __iter__(self) -> Iterator:
        """
        Returns iterator over batches.
        """
        #creating list of indices
        indices = np.arange(len(self.dataset))

        #shuffle if requested
        if self.shuffle:
            # Use NumPy RNG so `np.random.seed(...)` controls determinism.
            np.random.shuffle(indices)

        if isinstance(self.dataset, TensorDataset):
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                yield tuple(Tensor(tensor.data[batch_indices]) for tensor in self.dataset.tensors)
            return

        #yield batches
        for i in range(0,len(indices),self.batch_size):
            batch_indices = indices[i:i+self.batch_size]
            batch = [self.dataset[idx] for idx in batch_indices]

            #collate batch i.e converting list of tuples to tuple of tensors
            yield self._collate_batch(batch)

    def _collate_batch(self,batch:List[Tuple[Tensor, ...]]) -> Tuple[Tensor,...]:
        """
        Collate individual samples into batch tensors
        """

        if len(batch) == 0:
            return ()

        #determining number of tensors per sample
        num_tensors = len(batch[0])

        #group tensors by position
        batched_tensors = []
        for tensor_idx in range(num_tensors):
            #extracting all tensors at this position
            tensor_list = [sample[tensor_idx].data for sample in batch]

            #stacking into batch tensor
            batched_data = np.stack(tensor_list,axis=0)
            batched_tensors.append(Tensor(batched_data))

        return tuple(batched_tensors)
            
            
        
class RandomHorizontalFlip:
    """
    Randomly flips images horizontally with given probability

    A simple but effective augmentation for the msot image datasets.
    Flipping is appropriate when horizontal orientation doesn't change class

    Args:
        p:probability of flipping
    """    

    def __init__(self,p=0.5):
        """
        Intialize RandomHorizontalFlip

        """
        #validating is p is the range[0,1]
        if not 0.0 <=p <=1.0:
            raise ValueError(f"Probability must be between 0 and 1, got{p}")
        self.p = p

    def __call__(self,x):   
        """Apply random horizontal flip to input
        
        x:
          input array with shape (....,H,W) or (...,H,W,C)
          flips along the last-1 axis 

        Returns:
            Flipped ot unchanged array 
        """

        if np.random.random() < self.p:
            is_tensor = isinstance(x,Tensor)
            data = x.data if is_tensor else x

            #determining width axis for HW/HWC/CHW (and batched variants)
            if data.ndim == 2:
                #(H,W)
                axis = -1
            elif data.ndim >= 3:
                     if data.shape[-1] <=4:
                        #channel-last: (...,H,W,C)
                        axis = -2
                     elif data.shape[-3] <= 4:
                        #channels-first: (...,C,H,W)
                        axis = -1
                     else:
                        #fallback to width as last axis
                        axis = -1
            else:
               raise ValueError(f"Expected 2D+ input, got shape {data.shape}") 

            flipped = np.flip(data,axis=axis).copy()
            return Tensor(flipped) if is_tensor else flipped
        return x   

class RandomCrop:
    """
    Randomly crop image after padding

    This is the standard augmentation for CIFAR100:
    1. Pad the image by `padding` pixels on each size
    2. Randomly cropback to original size

    This simulates small translations in the image, forcing the model to
    recognize objects regardless of their exact position.

    Args:
        size: output crop size (int for square, or tuple(H,w))
        padding: pixles to pad on each side before cropping (default:4)
    """

    def __init__(self,size,padding=4):
        """
        Initialize RandomCrop.
        """

        if isinstance(size,int):
            self.size = (size,size)
        else:
            self.size = size
        self.padding = padding


    def __call__(self,x):
        """
        Applying random crop after padding

        Args:
            x: Input image with the shape (C,H,W) or (H,W)or (H,W,C)
            assume spatial dimensions are H,W
        """

        is_tensor = isinstance(x,Tensor)
        data = x.data if is_tensor else x

        target_h,target_w = self.size

        #determine image format and dimensions
        if len(data.shape) == 2:
            # (H,W) format
            h,w = data.shape
            padded = np.pad(data,self.padding,mode='constant',constant_values=0)

            #Random crop position
            top = np.random.randint(0,2*self.padding + h - target_h+1)
            left = np.random.randint(0,2*self.padding+ w - target_w +1)

            cropped = padded[top:top + target_h, left:left + target_w]

        elif len(data.shape) == 3:
            if data.shape[0] <=4: #likely (C,H,W) format
                c,h,w = data.shape
                #pad only spatial dimensions
                padded = np.pad(data, ((0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant', constant_values=0)

                #random crop position 
                top = np.random.randint(0,2*self.padding + 1)
                left = np.random.randint(0,2*self.padding +1)

                cropped = padded[:,top:top + target_h, left:left + target_w]

            else: #likely (H,W,C)format
                h,w,c = data.shape
                padded = np.pad(
                    data,
                    ((self.padding,self.padding),(self.padding,self.padding),(0,0)),
                    mode='constant',
                    constant_values=0
                )

                top = np.random.randint(0,2*self.padding + 1)
                left = np.random.randint(0,2*self.padding +1)

                cropped = padded[top:top +target_h, left:left + target_w,: ]

        else:
            raise ValueError(f"Expected 2D or 3D input, got shape {data.shape}")

        return Tensor(cropped) if is_tensor else cropped

class Compose:
    """
    Compose multiple transforms into a pipeline

    Applies transforms in sequence, passing output of each 
    as input to the next

    Args:
        transforms: List of transforms callables
    """

    def __init__(self,transforms):
        """Initialize Compose with list of transforms."""
        self.transforms = transforms

    def __call__(self,x):
        """Apply all transforms in sequence"""
        for transform in self.transforms:
            x = transform(x)
        return x


        

