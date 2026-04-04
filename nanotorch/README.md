# NanoTorch Package Guide

`nanotorch/` is a compatibility package that gives this project a more PyTorch-like import structure without changing the existing implementation modules.

It does not replace the original source folders such as `Tensor/`, `layers/`, `autograd/`, `optimizers/`, `attention/`, or `transformers/`.
Instead, it re-exports them through a cleaner package layout.

## Purpose

Use `nanotorch` when you want imports that look closer to PyTorch:

```python
import nanotorch as nt
import nanotorch.nn as nn
import nanotorch.optim as optim
import nanotorch.utils.data as data
```

This package is a thin wrapper layer:

- Existing implementation code stays in the original project modules
- `nanotorch` provides aliases and namespace organization
- Most files inside `nanotorch/` only import and re-export symbols

## Package Layout

### Top level

```python
import nanotorch as nt

nt.Tensor
nt.tensor(...)
nt.nn
nt.optim
nt.autograd
nt.utils
nt.tokenization
nt.training
nt.optimization
```

### `nanotorch.nn`

`nanotorch.nn` groups neural-network building blocks.

Available categories include:

- Core layers: `Layer`, `Module`, `Linear`, `Dropout`, `Sequential`
- Activations: `ReLU`, `Sigmoid`, `Tanh`, `GELU`, `Softmax`
- Convolutional layers: `Conv2d`, `MaxPool2d`, `AvgPool2d`, `BatchNorm2d`
- Embeddings: `Embedding`, `PositionalEncoding`, `EmbeddingLayer`
- Attention and transformers: `MultiHeadAttention`, `LayerNorm`, `MLP`, `TransformerBlock`, `GPT`
- Losses: `MSELoss`, `CrossEntropyLoss`, `BinaryCrossEntropyLoss`

Examples:

```python
import nanotorch.nn as nn

linear = nn.Linear(4, 8)
relu = nn.ReLU()
conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
loss_fn = nn.CrossEntropyLoss()
```

### `nanotorch.nn.modules`

This namespace provides a more PyTorch-like module split:

- `nanotorch.nn.modules.linear`
- `nanotorch.nn.modules.activation`
- `nanotorch.nn.modules.conv`
- `nanotorch.nn.modules.pooling`
- `nanotorch.nn.modules.normalization`
- `nanotorch.nn.modules.attention`
- `nanotorch.nn.modules.sparse`
- `nanotorch.nn.modules.transformer`
- `nanotorch.nn.modules.loss`
- `nanotorch.nn.modules.container`

Example:

```python
from nanotorch.nn.modules.linear import Linear
from nanotorch.nn.modules.conv import Conv2d
from nanotorch.nn.modules.transformer import TransformerBlock
```

### `nanotorch.optim`

Optimizers are exposed here:

- `Optimizer`
- `SGD`
- `Adam`
- `AdamW`

Example:

```python
import nanotorch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=1e-3)
```

### `nanotorch.autograd`

Autograd exports are available here:

- `enable_autograd`
- `Function`

Example:

```python
from nanotorch.autograd import enable_autograd

enable_autograd()
```

### `nanotorch.utils.data`

Dataset and dataloader utilities live here:

- `Dataset`
- `TensorDataset`
- `Dataloader`
- `DataLoader` alias
- `Compose`
- `RandomHorizontalFlip`
- `RandomCrop`

Example:

```python
from nanotorch.utils.data import DataLoader, TensorDataset

dataset = TensorDataset(inputs, targets)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

### `nanotorch.tokenization`

Tokenization utilities are grouped here:

- `Tokenizer`
- `CharTokenizer`
- `BPETokenizer`
- `create_tokenizer`
- `tokenize_dataset`
- `analyze_tokenization`

Example:

```python
from nanotorch.tokenization import create_tokenizer

tokenizer = create_tokenizer("char", corpus=["hello", "world"])
tokens = tokenizer.encode("hello")
```

### `nanotorch.training`

Training helpers are exposed here:

- `Trainer`
- `CosineSchedule`
- `clip_grad_norm`

Example:

```python
from nanotorch.training import Trainer, CosineSchedule
```

### `nanotorch.optimization`

Advanced optimization utilities are grouped by topic:

- `nanotorch.optimization.acceleration`
- `nanotorch.optimization.compression`
- `nanotorch.optimization.memoization`
- `nanotorch.optimization.profiling`
- `nanotorch.optimization.quantization`

These are thin wrappers over the existing `optimization/` project modules.

## Basic Usage

### Create tensors

```python
import nanotorch as nt

x = nt.tensor([[1, 2], [3, 4]])
print(x.shape)
```

### Build a small model

```python
import nanotorch as nt
import nanotorch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.Dropout(0.1),
    nn.Linear(8, 2),
)

x = nt.tensor([[1.0, 2.0, 3.0, 4.0]])
y = model(x)
```

### Use optimizer and loss

```python
import nanotorch.nn as nn
import nanotorch.optim as optim

loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

## Important Notes

- `nanotorch` reuses the current codebase. It is not a second implementation.
- If you need to fix actual math, autograd, tensor, or layer behavior, update the original implementation modules rather than the wrapper files in `nanotorch/`.
- Wrapper files inside `nanotorch/` should stay lightweight and focused on imports, aliases, and namespace organization.
- Some existing project APIs use names that differ from PyTorch. The compatibility package normalizes some of those through aliases, such as `Module = Layer` and `DataLoader = Dataloader`.

## Where To Edit

If you want to change behavior, edit the original implementation source:

- Tensor logic: `Tensor/`
- Autograd: `autograd/`
- Layers: `layers/`
- Optimizers: `optimizers/`
- Data utilities: `dataloader/`
- Activations: `activations/`
- Losses: `losses/`
- Attention: `attention/`
- Convolution: `convolution/`
- Embeddings: `embeddings/`
- Transformers: `transformers/`
- Training: `training/`
- Tokenization: `tokenization/`
- Optimization tooling: `optimization/`

If you want to change import structure or aliases, edit `nanotorch/`.
