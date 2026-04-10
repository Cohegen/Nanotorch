import os
import sys
import numpy as np
from dataloader import TensorDataset
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataloader import Dataset,RandomHorizontalFlip,RandomCrop,Compose
from Tensor import Tensor

def testing_augmentation():
    print("Testing whether data augmentation works")

    #test1-RandomHorizontalFlip
    print("RandomHorizontalFlip....")
    flip = RandomHorizontalFlip(p=1.0)

    img = np.array([[1,2,3],[4,5,6]]) #2x3 image
    flipped = flip(img)
    expected = np.array([[3,2,1],[6,5,4]])
    assert np.array_equal(flipped,expected), f"Flip failed: {flipped} vs {expected}"

    #test never flip
    no_flip = RandomHorizontalFlip(p=0.0)
    unchanged = no_flip(img)
    assert np.array_equal(unchanged,img), "p=0 should never change"

    #test2 randomCrop shape preservation
    print("Testing RandomCrop....")
    crop = RandomCrop(32,padding=4)

    #testing with (c,h,w) format 
    img_chw = np.random.randn(3,32,32)
    cropped = crop(img_chw)
    assert cropped.shape == (3,32,32),f"CHW crop shape wrong: {cropped.shape}"

    #testing with (h,w) format
    img_hw = np.random.randn(28,28)
    crop_hw = RandomCrop(28,padding=4)
    cropped_hw = crop_hw(img_hw)
    assert cropped_hw.shape == (28,28),f"HW crop shape wrong: {cropped_hw.shape}"

    #testing 3: Compose pipeline
    print("Testing Compose....")
    transforms = Compose([
        RandomHorizontalFlip(p=0.5),
        RandomCrop(32,padding=4)

    ])

    img = np.random.randn(3,32,32)
    augmented = transforms(img)
    assert augmented.shape == (3,32,32),f"Compose output shape wrong: {augmented.shape}"

    #test4- testing transforms work with Tensor
    print("Testing Tensor Compatibility")
    tensor_img = Tensor(np.random.randn(3,32,32))

    flip_result = RandomHorizontalFlip(p=1.0)(tensor_img)
    assert isinstance(flip_result,Tensor), "Flip should return Tensor when given Tensor"

    crop_result = RandomCrop(32,padding=4)(tensor_img)
    assert isinstance(crop_result,Tensor),"Crop should return Tensor when Tensor is given"

    #Test 5: Random verification
    print("  Testing randomness")
    flip_random =RandomHorizontalFlip(p=0.5)

    #run many times and check we get both outcomes
    flips = 0
    no_flips = 0
    test_img = np.array([[1,2]])

    for _ in range(1000):
        result = flip_random(test_img)
        if np.array_equal(result,np.array([[2,1]])):
            flips += 1
        else:
            no_flips += 1

    # with p=0.5 we should get roughly 50/50 
    assert flips > 20 and no_flips > 20, f"Flip randomness seems broken: {flips} flips, {no_flips} no flips"

if __name__ == "__main__":
    testing_augmentation()   