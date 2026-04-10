import os
import sys
import numpy as np
from dataloader import TensorDataset
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import Dataset
from Tensor import Tensor


def testing_tensordataset():
    """Tests TensorDataset"""
    print("Unit test: TensorDataset....")

    #tests basic functionality
    features = Tensor([[1,2],[3,4],[5,6]]) #3 samples, 2 features
    labels = Tensor([0,1,0])

    dataset = TensorDataset(features,labels)

    #test length
    assert len(dataset) == 3,f"Expected length 3,got {len(dataset)}"

    #test indexing
    sample = dataset[0]
    assert len(sample) == 2, "Should return tuple with tensors"
    assert np.array_equal(sample[0].data,[1,2]), f"Wrong features: {sample[0].data}"

    sample = dataset[1]
    assert np.array_equal(sample[1].data,1),f"Wrong label at index 1: {sample[1].data}"

    #test error handling
    try:
        dataset[10]
        assert False, "Should raise IndexError for out of bounds access"
    except IndexError:
        pass

    #testing mismatched tensor sizes
    try:
        bad_features = Tensor([[1,2],[3,4]])
        bad_labels = Tensor([0,1,0]) # 3 labels
        TensorDataset(bad_features,bad_labels)
        assert False, "Should raise error for mismatched tensor sizes"
    except ValueError:
        pass

    print("TensorDataset works correctly!")

if __name__ == "__main__":
    testing_tensordataset()