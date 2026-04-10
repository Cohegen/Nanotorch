import os
import sys
import numpy as np
from dataloader import TensorDataset
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import Dataset,Dataloader,TensorDataset
from Tensor import Tensor

def testing_dataloader():
    print("Testing Dataloader.")

    #create test dataset
    features = Tensor([[1,2],[3,4],[5,6],[7,8],[9,10]]) #sampples
    labels = Tensor([0,1,0,1,0])
    dataset = TensorDataset(features,labels)

    #test basic batching (no shuffling)
    loader = Dataloader(dataset,batch_size=2,shuffle=False)

    #test length calculation
    assert len(loader) == 3,f"Expected 3 batches, got{len(loader)}"

    batches = list(loader)
    assert len(batches)== 3, f"Expected 3 batches, got {len(batches)}"

    #Testing first batch
    batch_features,batch_labels = batches[0]
    assert batch_features.data.shape == (2,2),f"Wrong batch features shape: {batch_features.data.shape} "
    assert batch_labels.data.shape == (2,),f"Wrong batch labels shape: {batch_labels.data.shape}"

    #testing last batch
    batch_features,batch_labels = batches[2]
    assert batch_features.data.shape == (1,2),f"Wrong last batch features shape: {batch_features.data.shape}"
    assert batch_labels.data.shape == (1,),f"Wrong last batch labels shape: {batch_labels.data.shape}"

    #testing that data is preserved
    assert np.array_equal(batches[0][0].data[0],[1,2]),"First sample should be [1,2]"
    assert batches[0][1].data[0] == 0, "First label should be 0"

    #testing shuffling produces different order
    loader_shuffle = Dataloader(dataset,batch_size=5,shuffle=True)
    loader_no_shuffle = Dataloader(dataset,batch_size=5,shuffle=False)

    batch_shuffle = list(loader_shuffle)[0]
    batch_no_shuffle = list(loader_no_shuffle)[0]

    shuffle_features = set(tuple(row) for row in batch_shuffle[0].data)
    no_shuffle_features = set(tuple(row) for row in batch_no_shuffle[0].data)
    expected_features = {(1,2),(3,4),(5,6),(7,8),(9,10)}

    assert shuffle_features == expected_features,"Shuffled should preserve all data"
    no_shuffle_features == expected_features, "No shuffle should preserve all data"

    print("Dataloader works correctly...")
if __name__ == "__main__":
    testing_dataloader()