import os
from re import T
import sys


sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from autograd.autograd import ReLUBackward
from Tensor import Tensor
from convolutions import Conv2d,MaxPool2d,AvgPool2d,BatchNorm2d
from layers.layers import Linear
import numpy as np
"""
This script tries to implement a Simple convolution network

Here we will build a complete CNN that demonstrates how convolution and pooling work together.

#### The CNN Architecture Pattern

```
SimpleCNN Architecture Visualization:

Input: (batch, 3, 32, 32)     ← RGB images (CIFAR-10 size)
         ↓
┌─────────────────────────┐
│ Conv2d(3→16, 3×3, p=1)  │    ← Detect edges, textures
│ ReLU()                  │    ← Remove negative values
│ MaxPool(2×2)            │    ← Reduce to (batch, 16, 16, 16)
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│ Conv2d(16→32, 3×3, p=1) │   ← Detect shapes, patterns
│ ReLU()                  │   ← Remove negative values
│ MaxPool(2×2)            │   ← Reduce to (batch, 32, 8, 8)
└─────────────────────────┘
         ↓
┌─────────────────────────┐
│ Flatten()               │   ← Reshape to (batch, 2048)
│ Linear(2048→10)         │   ← Final classification
└─────────────────────────┘
         ↓
Output: (batch, 10)           ← Class probabilities
```

"""

class SimpleCNN:
    """
    Simple CNN demonstrating spatial operations integration.

    Architecture:
        Conv2d(3->16,3x3) + ReLU + MaxPool(2x2)
        Conv2d(16->32,3x3) + ReLU + MaxPool(2x2)
        Flatten + Linear(features->num_classes)

    """
    def __init__(self,num_classes=10):
        """
        Intializing SimpleCNN
        """
        super().__init__()

        #convolutional layers
        self.conv1 = Conv2d(in_channels=3,out_channels=16,kernel_size=3,padding=1)
        self.pool1 = MaxPool2d(kernel_size=2,stride=2)

        self.conv2 = Conv2d(in_channels=16,out_channels=32,kernel_size=3,padding=1)
        self.pool2 = MaxPool2d(kernel_size=2,stride=2)

        ##calculating flattened size
        #input = 32x32 -> Conv1 + Pool1: 16x16 -> Conv2+Pool2 : 8x8
        #Final: 32 channels x 8 x 8 = 2048 features
        self.flattened_size = 32 * 8 * 8
        self.num_classes = num_classes
        self.flattened_size = 32 * 8 * 8

    def forward(self,x):
        """

        Forward pass through SimpleCNN

        
        """
        #applying conv1 -> ReLU -> pool1
        #first conv block
        x = self.conv1(x)
        x = self.relu(x) #activation
        x = self.pool1(x)

        #seconnd conv block (conv2->ReLU-> pool2d)
        x = self.conv2(x)
        x = self.relu(x) # ReLU activation
        x = self.pool2(x)

        #flatten for classification (reshapes to 2D)
        batch_size = x.shape[0]
        x = x.reshape(batch_size,-1)

        #Returns Flattened features
        ##in a complete implementation, this would go through a Linear Layer
        return x

    def relu(self,x):
        """ReLU activation with gradient tracking for CNN"""
        result_data = np.maximum(0,x.data)
        result = Tensor(result_data)
        if x.requires_grad:
            result.requires_grad =True 
            result._grad_fn = ReLUBackward(x)
        return result 

    def parameters(self):
        """Returns all trainable parameters"""
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())

        #Linear layer parameters would be added here
        return params

    def __call__(self,x):
        """Enables model(x) syntax"""
        return self.forward(x)

def testing_simplecnn():
    """
    This function is intended to test whether the SimpleCNN
    works correctly

    """
    #testing forward pass with CIFAR-10 sized input
    print("  Testing forward pass...")
    model =SimpleCNN(num_classes=10)
    x = Tensor(np.random.rand(2,3,32,32)) #batch of 2,RGB , 32x32

    features = model(x)

    #expected shape : 2 samples, 32 channels x 8x8 spatial = 2048 features
    expected_shape = (2,2048)
    assert features.shape == expected_shape,f"Expected {expected_shape},got {features.shape}"

    #testing parameter counting
    print("Testing parameter counting")
    params = model.parameters()

    #conv1:(16,3,3,3) + bias (16,) =432 + 16 = 448
    conv1_params = 16 * 3 * 3 * 3 + 16 #weights+ bias 

    #conv2:(32,16,3,3) + bias (32,)=4608 + 32 = 4640
    conv2_params = 32 * 16 * 3 * 3 + 32 #weight + bias
    expected_total = conv1_params + conv2_params 

    #total: 448 + 4640 = 5088 parameters
    actual_total = sum(np.prod(p.shape)for p in params)
    assert actual_total == expected_total,f"Expected {expected_total} parameters, got {actual_total}"

    #Testing Different input sizes
    print("   Testing different input sizes....")

    #testing with different spatial dimensions
    x_small = Tensor(np.random.randn(1,3,16,16))
    features_small = model(x_small)

    #16x16 -> 8x8-> 4x4, so 32 x 4x4 = 512 features
    expected_small = (1,512)
    assert features_small.shape == expected_small,f"Expected {expected_small}, got {features_small.shape}"

    #testing batch processing
    print(" Testing batch processing...")
    x_batch = Tensor(np.random.randn(8,3,32,32))
    features_batch = model(x_batch)

    expected_batch= (8,2048)
    assert features_batch.shape ==expected_batch, f"Expected {expected_batch}, got {features_batch.shape}"

    print("SimpleCNN integration works goooood!!!")

if __name__ == "__main__":
    testing_simplecnn()



