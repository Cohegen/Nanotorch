import numpy as np

from Tensor import Tensor
from layers.layers import Linear
from transformers.transformers import TransformerBlock

from .common import mark_parameters_trainable, patchify_8x8_grayscale


class ViTTinyDigits:
    """
    Experimental ViT-style classifier for 8x8 grayscale images.

    This uses 2x2 patches, a small token projection, a single transformer block,
    sinusoidal-style fixed positional offsets, mean pooling, and a linear head.
    It is intentionally tiny so it remains plausible on CPU in this repository.
    """

    def __init__(self, num_classes=10, patch_size=2, embed_dim=16, num_heads=4):
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_dim = patch_size * patch_size
        self.num_patches = (8 // patch_size) ** 2

        self.patch_projection = Linear(self.patch_dim, embed_dim)
        self.block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, mlp_ratio=2)
        self.head = Linear(embed_dim, num_classes)

        self.position_encoding = self._build_position_encoding(self.num_patches, embed_dim)

        mark_parameters_trainable(self.parameters())

    def _build_position_encoding(self, seq_len, embed_dim):
        positions = np.arange(seq_len, dtype=np.float32)[:, None]
        dims = np.arange(embed_dim, dtype=np.float32)[None, :]
        scale = np.maximum(1.0, embed_dim - 1)
        encoding = np.sin((positions + 1.0) * (dims + 1.0) / scale)
        return Tensor(encoding[None, :, :])

    def forward(self, x):
        patches = patchify_8x8_grayscale(x, patch_size=self.patch_size)
        tokens = self.patch_projection(patches)
        tokens = tokens + self.position_encoding
        tokens = self.block(tokens)
        # Keep pooling in tensor space so gradients can flow into the encoder.
        pooled = tokens.mean(axis=1)
        return self.head(pooled)

    def predict(self, images):
        logits = self.forward(Tensor(images[:, None, :, :]))
        return np.argmax(logits.data, axis=1)

    def parameters(self):
        params = []
        params.extend(self.patch_projection.parameters())
        params.extend(self.block.parameters())
        params.extend(self.head.parameters())
        return params

    def __call__(self, x):
        return self.forward(x)
