import os
import sys
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import Dataset


def testing_dataset():
    """Testing Dataset abstract base class"""

    #test that Dataset is properly abstract
    try:
        dataset = Dataset()
        assert False, "Should not be able to instantiate abstract Dataset"
    except TypeError:
        print("Dataset is properly abstract")

    #testing concrete implementation
    class TestDataset(Dataset):
        def __init__(self,size):
            self.size = size

        def __len__(self):
            return self.size 

        def __getitem__(self,idx):
            return f"item_{idx}"

    dataset = TestDataset(10)
    assert len(dataset) == 10
    assert dataset[9] == "item_9"

    print("Dataset interface works correctly")

if __name__ == "__main__":
    testing_dataset()