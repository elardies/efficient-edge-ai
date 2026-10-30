# Efficient Edge AI

A machine learning engineering project exploring the trade-offs
between neural network accuracy, model size, and inference speed.

## Project Goal

Investigate how neural networks can be optimized for efficient
inference on resource-constrained devices.

## Planned Experiments

1. Train a baseline CNN using PyTorch.
2. Compare lightweight neural network architectures.
3. Apply model compression techniques.
4. Benchmark accuracy, model size, and inference latency.
5. Analyze the trade-offs between efficiency and performance.

## Technologies

- Python
- PyTorch
- NumPy
- Matplotlib

## Current Status

### Development Progress

### Phase 1: PyTorch Foundations

**Status: In progress**

Completed:
- Set up the Python development environment with PyTorch.
- Learned tensor operations and automatic differentiation (autograd).
- Implemented gradient descent manually using PyTorch.
- Built and trained a multilayer perceptron (MLP) using `torch.nn`.
- Used Mean Squared Error (MSE) and the Adam optimizer.
- Trained the network for 500 epochs to approximate the function \(y = 3x + 2\).
- Achieved a final training MSE of approximately **0.0163**.

The introductory exercises are available in the `learning/` directory.

### Next Steps

- Understand training, validation, and test datasets.
- Implement a baseline convolutional neural network (CNN) on CIFAR-10.
- Compare the baseline against MobileNetV3.
- Evaluate accuracy, model size, and inference latency.
- Investigate model compression techniques for efficient edge deployment.

