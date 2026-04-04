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

## Mission
This repository intends to showcase the implementation of PyTorch, one of the most popular ML libraries, from scratch in pure Python.

## Table of Contents

| Module | Date Created | Description |
| :--- | :--- | :--- |
| [activations](./activations) | Jan 10, 2026 | Common activation functions (ReLU, Sigmoid, Softmax, Tanh, GELU). |
| [Tensor](./Tensor) | Jan 10, 2026 | Core multidimensional array structure with basic operations. |
| [layers](./layers) | Jan 14, 2026 | Fundamental building blocks for neural networks (Linear, Dropout, etc.). |
| [losses](./losses) | Jan 17, 2026 | Standard loss functions for optimization. |
| [dataloader](./dataloader) | Jan 24, 2026 | Utilities for batching, shuffling, and processing datasets. |
| [autograd](./autograd) | Jan 31, 2026 | Automatic differentiation engine for gradient computation. |
| [optimizers](./optimizers) | Feb 8, 2026 | Optimization algorithms (SGD, Adam, etc.) to update parameters. |
| [training](./training) | Feb 13, 2026 | Scripts and utilities for managing the training lifecycle. |
| [convolution](./convolution) | Feb 21, 2026 | Implementation of convolutional layers and pooling operations. |
| [tokenization](./tokenization) | Feb 24, 2026 | Text processing and tokenization tools for NLP. |
| [embeddings](./embeddings) | Mar 3, 2026 | Vector representation for discrete tokens and positional encoding. |
| [attention](./attention) | Mar 6, 2026 | Scaled Dot-Product and Multi-Head Attention mechanisms. |
| [transformers](./transformers) | Mar 13, 2026 | Transformer architecture implementation. |
| [optimization](./optimization) | Mar 18, 2026 | Specialized optimization techniques and performance analyses. |
| [nanotorch](./nanotorch) | Apr 4, 2026 | Core library integration. |
| [experiments](./experiments) | Apr 4, 2026 | Pipeline tests and architectural experiments. |
| [projects](./projects) | Apr 4, 2026 | End-to-end applications and project examples. |

## Disclaimer
The implementation is still ongoing, so the code in this repo is not fully complete.

## Collaboration
I'm open for collaboration on this project and also in the future when I implement this project in either pure C or C++.

## Issues
If you face any issues, kindly notify me by opening an issue on this repository.


