import numpy as np

from Tensor import Tensor
from autograd.autograd import Function


class _ConcatBackward(Function):
    """Route gradients from a concatenated tensor back to each input slice."""

    def __init__(self, tensors, axis):
        super().__init__(*tensors)
        self.axis = axis
        self.sizes = [tensor.shape[axis] for tensor in tensors]

    def apply(self, grad_output):
        grads = []
        start = 0

        for tensor, size in zip(self.saved_tensors, self.sizes):
            grad = None
            if getattr(tensor, "requires_grad", False):
                slices = [slice(None)] * grad_output.ndim
                slices[self.axis] = slice(start, start + size)
                grad = grad_output[tuple(slices)]
            grads.append(grad)
            start += size

        return tuple(grads)


def mark_parameters_trainable(parameters):
    """Ensure parameters participate in optimization in this codebase."""
    for param in parameters:
        param.requires_grad = True
        param.grad = None


def count_parameters(model):
    """Return the total number of scalar trainable parameters."""
    total = 0
    for param in model.parameters():
        total += int(np.prod(param.data.shape))
    return total


def flatten_spatial(x):
    """Flatten NCHW feature maps into (batch, features)."""
    batch_size = x.shape[0]
    return x.reshape(batch_size, -1)


def concatenate_tensors(tensors, axis=0):
    """Concatenate tensors while preserving autograd links for each input."""
    if not tensors:
        raise ValueError("Expected at least one tensor to concatenate")

    data = np.concatenate([tensor.data for tensor in tensors], axis=axis)
    requires_grad = any(getattr(tensor, "requires_grad", False) for tensor in tensors)
    result = Tensor(data, requires_grad=requires_grad)

    if requires_grad:
        result._grad_fn = _ConcatBackward(tensors, axis)

    return result


def patchify_8x8_grayscale(images, patch_size=2):
    """
    Convert (batch, 1, 8, 8) images into flattened patches.

    Returns a Tensor shaped (batch, num_patches, patch_dim).
    """
    batch_size, channels, height, width = images.shape
    if channels != 1:
        raise ValueError(f"Expected grayscale images with 1 channel, got {channels}")
    if height != 8 or width != 8:
        raise ValueError(f"Expected 8x8 images, got {(height, width)}")
    if 8 % patch_size != 0:
        raise ValueError(f"Patch size {patch_size} must divide 8")

    patches_per_side = 8 // patch_size
    patch_dim = channels * patch_size * patch_size
    patch_count = patches_per_side * patches_per_side
    patch_data = np.zeros((batch_size, patch_count, patch_dim), dtype=np.float32)

    patch_idx = 0
    for row in range(0, 8, patch_size):
        for col in range(0, 8, patch_size):
            patch = images.data[:, :, row:row + patch_size, col:col + patch_size]
            patch_data[:, patch_idx, :] = patch.reshape(batch_size, -1)
            patch_idx += 1

    return Tensor(patch_data)
