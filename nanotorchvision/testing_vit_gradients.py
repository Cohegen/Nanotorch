import numpy as np

from Tensor import Tensor
from losses.losses import CrossEntropyLoss

from nanotorchvision.models import ViTTinyDigits


def test_vit_tiny_digits_backward_reaches_all_parameters():
    model = ViTTinyDigits(num_classes=10, patch_size=2, embed_dim=16, num_heads=4)
    x = Tensor(np.random.rand(4, 1, 8, 8).astype(np.float32))
    y = Tensor(np.array([0, 1, 2, 3], dtype=np.int64))

    logits = model(x)
    loss = CrossEntropyLoss()(logits, y)
    loss.backward()

    missing = [
        (index, param.shape)
        for index, param in enumerate(model.parameters())
        if param.grad is None
    ]

    assert not missing, f"Missing gradients for parameters: {missing}"
