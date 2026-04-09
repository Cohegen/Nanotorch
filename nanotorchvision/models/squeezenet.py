import numpy as np

from Tensor import Tensor
from activations.activations import ReLU
from convolution.convolutions import Conv2d, MaxPool2d
from layers.layers import Linear

from .common import flatten_spatial, mark_parameters_trainable


class FireModule:
    """Small SqueezeNet-style fire module for NanoDigits."""

    def __init__(self, in_channels, squeeze_channels, expand_channels):
        self.squeeze = Conv2d(
            in_channels=in_channels,
            out_channels=squeeze_channels,
            kernel_size=1,
        )
        self.expand1x1 = Conv2d(
            in_channels=squeeze_channels,
            out_channels=expand_channels,
            kernel_size=1,
        )
        self.expand3x3 = Conv2d(
            in_channels=squeeze_channels,
            out_channels=expand_channels,
            kernel_size=3,
            padding=1,
        )
        self.relu = ReLU()

    def forward(self, x):
        x = self.relu(self.squeeze(x))
        expand_1x1 = self.relu(self.expand1x1(x))
        expand_3x3 = self.relu(self.expand3x3(x))
        return expand_1x1 + expand_3x3

    def parameters(self):
        params = []
        params.extend(self.squeeze.parameters())
        params.extend(self.expand1x1.parameters())
        params.extend(self.expand3x3.parameters())
        return params

    def __call__(self, x):
        return self.forward(x)


class SqueezeNetTinyDigits:
    """Compact SqueezeNet-style classifier sized for 8x8 grayscale digits."""

    def __init__(self, num_classes=10):
        self.stem = Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.relu = ReLU()
        self.pool = MaxPool2d(kernel_size=2, stride=2)

        self.fire1 = FireModule(in_channels=8, squeeze_channels=4, expand_channels=8)
        self.fire2 = FireModule(in_channels=8, squeeze_channels=4, expand_channels=8)

        self.classifier = Linear(8 * 4 * 4, num_classes)
        mark_parameters_trainable(self.classifier.parameters())

    def forward(self, x):
        x = self.relu(self.stem(x))
        x = self.fire1(x)
        x = self.pool(x)
        x = self.fire2(x)
        x = flatten_spatial(x)
        return self.classifier(x)

    def predict(self, images):
        logits = self.forward(Tensor(images[:, None, :, :]))
        return np.argmax(logits.data, axis=1)

    def parameters(self):
        params = []
        params.extend(self.stem.parameters())
        params.extend(self.fire1.parameters())
        params.extend(self.fire2.parameters())
        params.extend(self.classifier.parameters())
        return params

    def __call__(self, x):
        return self.forward(x)
