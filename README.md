![Output examples](azula.gif)

![Alt text](https://github.com/Cohegen/Nanotorch/blob/main/assets/logo1.png)

# NanoTorch: Deep Learning from First Principles

**Ever wondered what actually happens when you call `.backward()` in PyTorch?**

NanoTorch is an educational, from-scratch implementation of a modern deep learning framework in pure Python. While production frameworks like PyTorch and TensorFlow are optimized for performance, they often hide the elegant mathematical machinery behind layers of C++ and CUDA code. 

NanoTorch pulls back the curtain. By stripping away the complexity and focusing on the core logic, this project provides a transparent "glass box" view into:
- **Autograd Engine:** Building a dynamic computational graph and implementing backpropagation manually.
- **Tensor Foundations:** Understanding how multidimensional arrays and broadcast operations form the bedrock of AI.
- **Architectural Anatomy:** Constructing everything from basic Linear layers to complex Multi-Head Attention mechanisms from the ground up.
- **Optimization Intuition:** Implementing algorithms like Adam and SGD to see exactly how they steer weights through the loss landscape.

Whether you're a student trying to bridge the gap between theory and code, or an engineer looking to deepen your architectural understanding, NanoTorch is built to be read, tinkered with, and understood.

## Why NanoTorch
- Pure Python implementation of tensors, autograd, layers, optimizers, and training loops.
- Built to expose the mechanics behind backpropagation instead of hiding them behind optimized kernels.
- Covers both classical vision models and transformer-era components in one educational codebase.
- Useful as a readable sandbox for debugging ideas, testing concepts, and learning system internals.

## Mission
This repository intends to showcase the implementation of PyTorch, one of the most popular ML libraries, from scratch in pure Python.

## Who It Is For
- Students learning how gradients, layers, and optimizers work under the hood.
- Engineers who want a compact reference implementation of core deep learning components.
- Builders who want to experiment with model ideas in a transparent, hackable codebase.

## Quick Start

```powershell
git clone https://github.com/Cohegen/NanoTorch.git
cd NanoTorch
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install numpy matplotlib
```

Run a simple benchmark:

```powershell
python projects/CNNS/alexnet_digits.py
```

Generate the benchmark leaderboard:

```powershell
python projects/CNNS/generate_leaderboard.py
```

Run a focused gradient regression test:

```powershell
python -m pytest nanotorchvision/testing_densenet_gradients.py
```

## Table of Contents

| Module | Description |
| :--- | :--- |
| [Tensor](./Tensor) | Core multidimensional array structure with basic operations. |
| [activations](./activations) | Common activation functions (ReLU, Sigmoid, Softmax, Tanh, GELU). |
| [layers](./layers) | Fundamental building blocks for neural networks (Linear, Dropout, etc.). |
| [losses](./losses) | Standard loss functions for optimization. |
| [dataloader](./dataloader) | Utilities for batching, shuffling, and processing datasets. |
| [autograd](./autograd) | Automatic differentiation engine for gradient computation. |
| [optimizers](./optimizers) | Optimization algorithms (SGD, Adam, etc.) to update parameters. |
| [training](./training) | Scripts and utilities for managing the training lifecycle. |
| [convolution](./convolution) | Implementation of convolutional layers and pooling operations. |
| [tokenization](./tokenization) | Text processing and tokenization tools for NLP. |
| [embeddings](./embeddings) | Vector representation for discrete tokens and positional encoding. |
| [attention](./attention) | Scaled Dot-Product and Multi-Head Attention mechanisms. |
| [transformers](./transformers) | Transformer architecture implementation. |
| [optimization](./optimization) | Specialized optimization techniques and performance analyses. |
| [nanotorch](./nanotorch) | Core library integration. |
| [nanotorchvision](./nanotorchvision) | Vision model zoo, NanoDigits dataset helpers, and benchmark leaderboard tooling. |
| [experiments](./experiments) | Pipeline tests and architectural experiments. |
| [projects](./projects) | End-to-end applications and project examples. |

## Disclaimer
The implementation is still ongoing, so the code in this repo is not fully complete.

## Implementation Status

| Status | Coverage |
| :--- | :--- |
| Implemented | Tensor operations, autograd, activations, losses, optimizers, dataloading, convolution, attention, embeddings, transformers, and benchmark tooling. |
| Experimental | NanoTorchVision model zoo, ViT benchmark path, and several optimization analysis modules. |
| Still Improving | Runtime performance, API consistency, vectorized kernels, and broader dataset coverage. |

## How It Fits Together

```text
Tensor -> Autograd -> Layers -> Loss -> Optimizer -> Training Loop -> Benchmarks
                     |                |
                     +-> CNNs         +-> Metrics / Plots / Summaries
                     +-> Attention
                     +-> Transformers
```

## NanoTorchVision
`nanotorchvision/` is a lightweight vision companion package for this repository. It centralizes NanoDigits dataset loading, CPU-friendly image model definitions, and leaderboard generation from saved benchmark summaries.

The first pass includes:
- `MiniResNetDigits`
- `AlexNetTinyDigits`
- `MobileNetStyleTinyDigits`
- `DenseNetTinyDigits`
- `ViTTinyDigits` as an experimental tiny patch model
- leaderboard helpers in `nanotorchvision.benchmarks`

DenseNet support is now available through a small differentiable channel-concatenation path used by `DenseNetTinyDigits`.

## Computer Vision Analysis
The NanoDigits computer vision runs now have enough saved benchmark data to compare optimization behavior directly instead of relying on rough visual inspection alone. The summary below uses the saved CSV and JSON artifacts in `projects/CNNS/plots/`.

### NanoDigits Model Comparison

| Model | Parameters | Runtime | Final Train Acc | Final Test Acc | Best Test Acc | Final Test Loss |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| ViTTinyDigits | 2,474 | 15.08 s | 91.0% | 92.0% | 92.0% | 0.3021 |
| AlexNetTinyDigits | 4,026 | 1189.86 s | 95.1% | 95.0% | 95.0% | 0.1511 |
| DenseNetTinyDigits | 5,238 | 9215.34 s | 99.6% | 98.0% | 99.0% | 0.0491 |
| MobileNetStyleTinyDigits | 8,706 | 3111.75 s | 90.9% | 91.5% | 97.5% | 0.3176 |

### What The Numbers Suggest

| Observation | Evidence | Interpretation |
| :--- | :--- | :--- |
| DenseNet gives the strongest peak accuracy | Best test accuracy reaches 99.0% with a 0.0356 lowest test loss | Dense feature reuse works well on NanoDigits and the model is expressive enough to fit the task very cleanly. |
| ViT is the fastest model by a wide margin | 15.08 seconds total runtime with only 2,474 parameters | The tiny patch model is computationally cheap in this codebase and gives a strong speed-to-accuracy tradeoff even though its final accuracy is lower than the best CNNs. |
| AlexNet is the most stable conventional CNN run in the saved artifacts | Test accuracy climbs to 95.0% and finishes at its peak with matching train/test accuracy | The optimization path is smoother than MobileNet and less aggressive than DenseNet, which makes it a good baseline for this dataset. |
| MobileNet shows the largest late-training instability | Best test accuracy is 97.5% at epoch 8, but final test accuracy falls to 91.5% by epoch 10 | The current learning-rate and architecture combination is too volatile late in training, so the last checkpoint is materially worse than the best checkpoint. |
| Parameter count is not the main driver of benchmark quality here | MobileNet has the most parameters yet underperforms DenseNet and AlexNet in final accuracy | In this educational implementation, training dynamics and architecture fit matter more than raw size on NanoDigits. |

### Model-by-Model Reading

| Model | Analysis |
| :--- | :--- |
| ViTTinyDigits | The model improves steadily every epoch without a collapse phase. It appears underpowered relative to the stronger CNNs, but it is dramatically faster and still reaches 92.0% test accuracy, which is a credible result for such a small transformer. |
| AlexNetTinyDigits | This run looks balanced: the model starts slowly, then accelerates through the middle epochs and finishes with train and test accuracy aligned at about 95%. That small gap suggests good generalization and a clean optimization path. |
| DenseNetTinyDigits | This is the strongest run in absolute terms. Test accuracy reaches 98.5% by epoch 3, peaks at 99.0% by epoch 7, and stays high through the remainder of training. The runtime cost is substantial because dense reuse increases convolution work in an implementation that still uses explicit Python loops. |
| MobileNetStyleTinyDigits | The model learns quickly and reaches 97.5% test accuracy, but the run is not robust through the last two epochs. The jump in test loss to 0.7239 at epoch 9 and the drop in final train accuracy indicate optimizer instability rather than a simple capacity limit. |

### Practical Takeaways

| Goal | Best Current Choice | Reason |
| :--- | :--- | :--- |
| Best NanoDigits accuracy | DenseNetTinyDigits | Highest measured best test accuracy at 99.0%. |
| Best speed | ViTTinyDigits | Fastest end-to-end runtime by a very large margin. |
| Best balanced baseline | AlexNetTinyDigits | Strong final accuracy with a smooth and stable training curve. |
| Most in need of tuning | MobileNetStyleTinyDigits | High mid-run quality, but weak final stability. |

## Benchmarks
Benchmarks are meant to show how the framework behaves during training, not to act as a full leaderboard dump inside the top-level README.

### Vision Benchmarks

| Model | Dataset | Best Test Acc | Notes |
| :--- | :--- | ---: | :--- |
| DenseNetTinyDigits | NanoDigits | 99.0% | Best accuracy in the current saved runs. |
| AlexNetTinyDigits | NanoDigits | 95.0% | Strong and stable baseline CNN. |
| ViTTinyDigits | NanoDigits | 92.0% | Fastest model in the current implementation. |
| MobileNetStyleTinyDigits | NanoDigits | 97.5% | High peak accuracy, but unstable late in training. |
| LeNetCIFAR | CIFAR-10 subset | 31.5% | Educational baseline; clearly underfits the harder dataset. |

### Key Takeaways

| Signal | Brief Reading |
| :--- | :--- |
| NanoDigits difficulty | Small CNNs already perform well because the dataset is compact and visually simple. |
| Best current NanoDigits model | DenseNetTinyDigits gives the strongest measured held-out accuracy. |
| Best speed/efficiency tradeoff | ViTTinyDigits is much faster than the CNN baselines, but it gives up some accuracy. |
| Most obvious tuning target | MobileNetStyleTinyDigits needs more stable late-epoch optimization. |
| Harder benchmark | CIFAR-10 is still much tougher for the current pure-Python convolution stack. |

### Artifacts
- Vision plots and metrics: `projects/CNNS/plots/`
- Vision package notes: `nanotorchvision/README.md`
- Benchmark entry scripts: `projects/CNNS/`

Use the CSV and JSON files in `projects/CNNS/plots/` as the authoritative source for exact benchmark values.

## Collaboration
I'm open for collaboration on this project and also in the future when I implement this project in either pure C or C++.

## Issues
If you face any issues, kindly notify me by opening an issue on this repository.
