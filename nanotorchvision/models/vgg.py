import numpy as np

from Tensor import Tensor
from activations.activations import ReLU
from convolution.convolutions import Conv2d, MaxPool2d
from layers.layers import Linear

from .common import flatten_spatial, mark_parameters_trainable


class VGGNetTinyDigits:
    """Compact VGG-style CNN for 8x8 grayscale digit images."""

    def __init__(self, num_classes=10):
        self.conv1 = Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.conv2 = Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1)
        self.pool1 = MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv4 = Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1)
        self.pool2 = MaxPool2d(kernel_size=2, stride=2)

        self.relu = ReLU()
        self.fc1 = Linear(16 * 2 * 2, 32)
        self.fc2 = Linear(32, num_classes)

        mark_parameters_trainable(self.fc1.parameters() + self.fc2.parameters())

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool1(x)

        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.pool2(x)

        x = flatten_spatial(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def predict(self, images):
        logits = self.forward(Tensor(images[:, None, :, :]))
        return np.argmax(logits.data, axis=1)

    def parameters(self):
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        params.extend(self.conv3.parameters())
        params.extend(self.conv4.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params

    def __call__(self, x):
        return self.forward(x)
