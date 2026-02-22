# Convolution

Convolution is a mathematical operation used extensively in signal processing and image analysis. The process involves two functions (or signals): the input function and the kernel (or filter).

## General Process
1. **Input Function**: This is the original signal or image.
2. **Kernel**: A small matrix used to apply effects like blurring, sharpening, or edge detection.
3. **Output Function**: The result of applying the kernel to the input function.

### Mathematical Definition
The convolution of two functions f and g is defined as:

$$ (f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau $$

### Applications
- Image Processing
- Signal Filtering
- Feature Extraction in Machine Learning

## Example
In the context of image processing, applying a Gaussian blur can be done using convolution with a Gaussian kernel.