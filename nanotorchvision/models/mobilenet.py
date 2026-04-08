import numpy as np

from Tensor import Tensor
from activations.activations import ReLU
from convolution.convolutions import Conv2d, MaxPool2d
from layers.layers import Linear

from .common import flatten_spatial, mark_parameters_trainable


class SeparableStyleBlock:
    """
    MobileNet-inspired block adapted to current NanoTorch constraints.

    True depthwise grouped convolution is not implemented yet in the core
    convolution module, so this block uses a lightweight spatial conv followed
    by a pointwise 1x1 conv.
    """

    def __init__(self, in_channels, out_channels):
        self.spatial = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1)
        self.pointwise = Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.relu = ReLU()

    def forward(self, x):
        x = self.relu(self.spatial(x))
        x = self.relu(self.pointwise(x))
        return x

    def parameters(self):
        params = []
        params.extend(self.spatial.parameters())
        params.extend(self.pointwise.parameters())
        return params

    def __call__(self, x):
        return self.forward(x)


class MobileNetStyleTinyDigits:
    """CPU-friendly MobileNet-inspired model for NanoDigits."""

    def __init__(self, num_classes=10):
        self.stem = Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.block1 = SeparableStyleBlock(8, 12)
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.block2 = SeparableStyleBlock(12, 16)
        self.relu = ReLU()
        self.fc1 = Linear(16 * 4 * 4, 24)
        self.fc2 = Linear(24, num_classes)

        mark_parameters_trainable(self.fc1.parameters() + self.fc2.parameters())

    def forward(self, x):
        x = self.relu(self.stem(x))
        x = self.block1(x)
        x = self.pool(x)
        x = self.block2(x)
        x = flatten_spatial(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

    def predict(self, images):
        logits = self.forward(Tensor(images[:, None, :, :]))
        return np.argmax(logits.data, axis=1)

    def parameters(self):
        params = []
        params.extend(self.stem.parameters())
        params.extend(self.block1.parameters())
        params.extend(self.block2.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params

    def __call__(self, x):
        return self.forward(x)
