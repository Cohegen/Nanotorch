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
| [experiments](./experiments) | Pipeline tests and architectural experiments. |
| [projects](./projects) | End-to-end applications and project examples. |


This benchmark guide documents representative training behavior for the educational models in NanoTorch. The goal is not to present production-grade leaderboard numbers, but to communicate how the framework behaves during optimization, how quickly models converge, and how training and evaluation metrics evolve across epochs.

For the current CNN benchmark, `projects/CNNS/lenet_digits.py` trains a LeNet-style classifier on the local NanoDigits dataset for 10 epochs and writes two benchmark artifacts:
- `projects/CNNS/plots/lenet_digits_loss.png`
- `projects/CNNS/plots/lenet_digits_accuracy.png`

These plots track four core statistics after every epoch: training loss, test loss, training accuracy, and test accuracy. They are intended to help readers judge convergence quality, generalization behavior, and the stability of the underlying training implementation.

### LeNetDigits Benchmark Summary

| Benchmark | Dataset | Epochs | Metrics Tracked | Observed Result |
| :--- | :--- | :---: | :--- | :--- |
| LeNetDigits | NanoDigits | 10 | Train loss, test loss, train accuracy, test accuracy | Loss curves decrease over training, while accuracy curves increase and then stabilize toward the later epochs. |

### LeNetDigits Benchmark Interpretation

| Signal | What the plots show | Why it matters |
| :--- | :--- | :--- |
| Training loss | Falls steadily across epochs | The optimizer is successfully fitting the training split. |
| Test loss | Declines alongside training loss | The model is improving on held-out data rather than only memorizing the train set. |
| Training accuracy | Rises quickly in early epochs and then tapers | The network learns useful digit features early and approaches convergence later in training. |
| Test accuracy | Tracks the training trend with a small gap | Generalization remains reasonably aligned with training performance. |

### Benchmark Artifacts

| Plot | Purpose |
| :--- | :--- |
| `lenet_digits_loss.png` | Visualizes how train loss and test loss change after each epoch. |
| `lenet_digits_accuracy.png` | Visualizes how train accuracy and test accuracy change after each epoch. |


![Alt text](https://github.com/Cohegen/Nanotorch/blob/main/projects/CNNS/plots/lenet_digits_accuracy.png)

![Alt text](https://github.com/Cohegen/Nanotorch/blob/main/projects/CNNS/plots/lenet_digits_loss.png)


## Disclaimer
The implementation is still ongoing, so the code in this repo is not fully complete.

## Collaboration
I'm open for collaboration on this project and also in the future when I implement this project in either pure C or C++.

## Issues
If you face any issues, kindly notify me by opening an issue on this repository.
