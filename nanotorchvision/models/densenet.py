import numpy as np

from Tensor import Tensor
from activations.activations import ReLU
from convolution.convolutions import Conv2d, MaxPool2d
from layers.layers import Linear

from .common import concatenate_tensors, flatten_spatial, mark_parameters_trainable


class DenseLayer:
    """A tiny DenseNet layer that appends new features to the input stack."""

    def __init__(self, in_channels, growth_rate):
        self.in_channels = in_channels
        self.growth_rate = growth_rate
        self.relu = ReLU()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=growth_rate,
            kernel_size=3,
            padding=1,
        )

    def forward(self, x):
        new_features = self.conv(self.relu(x))
        return concatenate_tensors([x, new_features], axis=1)

    def parameters(self):
        return self.conv.parameters()

    def __call__(self, x):
        return self.forward(x)


class DenseBlock:
    """Stack of dense layers with feature reuse through concatenation."""

    def __init__(self, in_channels, growth_rate, num_layers):
        self.layers = []
        channels = in_channels
        for _ in range(num_layers):
            layer = DenseLayer(channels, growth_rate)
            self.layers.append(layer)
            channels += growth_rate
        self.out_channels = channels

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def __call__(self, x):
        return self.forward(x)


class TransitionBlock:
    """Channel compression followed by spatial downsampling."""

    def __init__(self, in_channels, out_channels):
        self.relu = ReLU()
        self.compress = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
        )
        self.pool = MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.compress(self.relu(x))
        return self.pool(x)

    def parameters(self):
        return self.compress.parameters()

    def __call__(self, x):
        return self.forward(x)


class DenseNetTinyDigits:
    """A DenseNet-style classifier sized for 8x8 grayscale NanoDigits."""

    def __init__(self, num_classes=10, growth_rate=4):
        self.stem = Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        self.relu = ReLU()

        self.block1 = DenseBlock(in_channels=8, growth_rate=growth_rate, num_layers=2)
        self.transition = TransitionBlock(
            in_channels=self.block1.out_channels,
            out_channels=12,
        )
        self.block2 = DenseBlock(in_channels=12, growth_rate=growth_rate, num_layers=2)

        self.classifier = Linear(self.block2.out_channels * 4 * 4, num_classes)
        mark_parameters_trainable(self.classifier.parameters())

    def forward(self, x):
        x = self.relu(self.stem(x))
        x = self.block1(x)
        x = self.transition(x)
        x = self.block2(x)
        x = flatten_spatial(x)
        return self.classifier(x)

    def predict(self, images):
        logits = self.forward(Tensor(images[:, None, :, :]))
        return np.argmax(logits.data, axis=1)

    def parameters(self):
        params = []
        params.extend(self.stem.parameters())
        params.extend(self.block1.parameters())
        params.extend(self.transition.parameters())
        params.extend(self.block2.parameters())
        params.extend(self.classifier.parameters())
        return params

    def __call__(self, x):
        return self.forward(x)
