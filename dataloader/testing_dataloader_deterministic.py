import os
import sys


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from dataloader.dataloader import TensorDataset, Dataloader, Dataset
from Tensor import Tensor


def testing_dataloader_deterministic():
    print("Testing DataLoader Deterministic Shuffling")

    #create test dataset
    features = Tensor([[1,2],[3,4],[5,6],[7,8]])
    labels = Tensor([0,1,0,1])
    dataset = TensorDataset(features,labels)

    np.random.seed(42)
    loader1 = Dataloader(dataset,batch_size=2,shuffle=True)
    batches1 = list(loader1)

    np.random.seed(42)
    loader2 = Dataloader(dataset,batch_size=2,shuffle=True)
    batches2 = list(loader2)

    #should produce identical batches with same seed
    for i, (batch1,batch2) in enumerate(zip(batches1,batches2)):
        assert np.array_equal(batch1[0].data,batch2[0].data),f"Batch{i} features should be identical with the same seed"
        assert np.array_equal(batch1[1].data,batch2[1].data), f"Batch{i} labels should be identical with same seed"


    #testing that different seeds produce different shuffles
    np.random.seed(42)
    loader3 = Dataloader(dataset,batch_size=2,shuffle=True)
    batches3 = list(loader3)

    np.random.seed(123)
    loader4 = Dataloader(dataset,batch_size=2,shuffle=True)
    batches4 = list(loader4)

    #should produce different batches with different seeds
    different = False
    for batch3,batch4 in zip(batches3,batches4):
        if not np.array_equal(batch3[0].data,batch4[0].data):
            different = True 
        
    assert different, "Different seeds should produce different shuffles"

    print("Determiniistic shuffling works")

if __name__ == "__main__":
    testing_dataloader_deterministic()