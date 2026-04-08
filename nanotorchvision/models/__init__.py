"""Vision model exports."""

from .alexnet import AlexNetTinyDigits
from .common import count_parameters
from .densenet import DenseNetTinyDigits
from .mobilenet import MobileNetStyleTinyDigits
from .resnet import MiniResNetDigits
from .vit import ViTTinyDigits

MODEL_REGISTRY = {
    "alexnet_tiny_digits": AlexNetTinyDigits,
    "densenet_tiny_digits": DenseNetTinyDigits,
    "mobilenet_style_tiny_digits": MobileNetStyleTinyDigits,
    "mini_resnet_digits": MiniResNetDigits,
    "vit_tiny_digits": ViTTinyDigits,
}

__all__ = [
    "AlexNetTinyDigits",
    "count_parameters",
    "DenseNetTinyDigits",
    "MiniResNetDigits",
    "MobileNetStyleTinyDigits",
    "MODEL_REGISTRY",
    "ViTTinyDigits",
]
