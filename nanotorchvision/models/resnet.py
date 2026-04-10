import numpy as np

from Tensor import Tensor
from activations.activations import ReLU
from convolution.convolutions import Conv2d, MaxPool2d
from layers.layers import Linear

from .common import flatten_spatial, mark_parameters_trainable


class ResidualBlock:
    """A small basic residual block without batch normalization."""

    def __init__(self, channels, method='im2col'):
        self.conv1 = Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, method=method)
        self.conv2 = Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1, method=method)
        self.relu = ReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        return out

    def parameters(self):
        params = []
        params.extend(self.conv1.parameters())
        params.extend(self.conv2.parameters())
        return params

    def __call__(self, x):
        return self.forward(x)


class MiniResNetDigits:
    """CPU-friendly mini ResNet for 8x8 grayscale digit images."""

    def __init__(self, num_classes=10, method='im2col'):
        self.stem = Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1, method=method)
        self.block1 = ResidualBlock(channels=8, method=method)
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.transition = Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, method=method)
        self.block2 = ResidualBlock(channels=16, method=method)

        self.relu = ReLU()
        self.fc1 = Linear(16 * 4 * 4, 32)
        self.fc2 = Linear(32, num_classes)

        mark_parameters_trainable(self.fc1.parameters() + self.fc2.parameters())

    def forward(self, x):
        x = self.stem(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.pool(x)
        x = self.transition(x)
        x = self.relu(x)
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
        params.extend(self.transition.parameters())
        params.extend(self.block2.parameters())
        params.extend(self.fc1.parameters())
        params.extend(self.fc2.parameters())
        return params

    def __call__(self, x):
        return self.forward(x)
