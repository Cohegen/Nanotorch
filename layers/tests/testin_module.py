from multiprocessing import Value
from re import T
import time
import os
import sys
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Tensor import Tensor
from activations.activations import ReLU, Softmax
from layers.layers import XAVIER_SCALE_FACTOR, Layer,Linear,Dropout
from layers.testing_dropout_layer import testing_dropout_layer
from layers.testing_linear_layer import testing_linear_layer
from layers.testing_edge_cases import test_edge_cases
from layers.testing_parameter import testing_parameter_collection_linear
from layers.testing_dropout_layer import testing_dropout_layer


def testing_module():
    print("Running Module Intergration Test")
    print("="*50)

    # Run all unit tests
    print("Running unit tests...")
    testing_linear_layer()
    test_edge_cases()
    testing_parameter_collection_linear()
    testing_dropout_layer()

    #using ReLU imported from package at module level
    ReLU_class = ReLU

    #Building individual layers for manual compostion
    layer1 = Linear(784,256)
    activation1 = ReLU_class()
    dropout1 = Dropout(0.5)
    layer2 = Linear(256,64)
    activation2 = ReLU_class()
    dropout2 = Dropout(0.3)
    layer3 = Linear(64,10)

    #testing end-to-end forward pass 
    batch_size = 16
    x = Tensor(np.random.randn(batch_size,784))

    #manual forward pass
    x = layer1.forward(x)
    x = activation1.forward(x)
    x = dropout1.forward(x)
    x = layer2.forward(x)
    x = activation2.forward(x)
    x = dropout2.forward(x)
    output = layer3.forward(x)

    assert output.shape == (batch_size,10), f"Expected output shape ({batch_size,10}, got {output.shape})"
    
    #testing parameter counting from individual layers
    all_params = layer1.parameters() + layer2.parameters() + layer3.parameters()
    expected_params = 6 # 3 weights + 3 biases from 3 linear layers
    assert len(all_params) == expected_params, f"Expected {expected_params} parameters,got {len(all_params)}"

    #testing individual layer functionality
    test_x = Tensor(np.random.randn(4,784))
    #testing dropout in training vs inference
    dropout_test = Dropout(0.5)
    train_output = dropout_test.forward(test_x,training=True)
    infer_output = dropout_test.forward(test_x,training=False)
    assert np.array_equal(test_x.data,infer_output.data),"Inference mode should pass through unchanged"

    print("Multilayer network intergration done")
    print("\n" + "="*50)
    print("All tests passed")

if __name__ == "__main__":
    testing_module()