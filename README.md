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

### Phase 1: PyTorch Foundations

**Status: Completed — November 2, 2026**

Completed:
- Set up the Python development environment with PyTorch.
- Practiced tensor operations and automatic differentiation (autograd).
- Implemented gradient descent manually using PyTorch.
- Built and trained a multilayer perceptron (MLP) using `torch.nn`.
- Used Mean Squared Error (MSE) and the Adam optimizer.
- Implemented reproducible training, validation, and test splits.
- Evaluated the trained network on held-out observations.
- Practiced inference using a trained model.

### Training and Evaluation

As an introductory exercise, a 1–8–1 multilayer perceptron was trained to approximate the function \(y = 3x + 2\).

**Experimental setup:**
- Dataset: 100 synthetic observations.
- Split: 70 training, 15 validation, and 15 test samples.
- Optimizer: Adam (learning rate = 0.01).
- Loss function: Mean Squared Error (MSE).
- Training duration: 500 epochs.

**Results:**

| Dataset | MSE |
|---|---:|
| Training | 0.01986 |
| Validation | 0.02577 |
| Test | 0.01769 |

The model achieved low prediction errors on both training and held-out data, suggesting that it learned to approximate the underlying relationship.

These experiments are introductory PyTorch exercises and are separate from the main Efficient Edge AI benchmarks.

The code is available in the `learning/` directory.

### Phase 2: CNN Baseline on CIFAR-10

**Status: Not started**

Next steps:
- Load and explore the CIFAR-10 dataset using PyTorch.
- Understand `Dataset` and `DataLoader`.
- Implement and train a baseline convolutional neural network (CNN).
- Evaluate training and validation performance.
- Record accuracy and loss curves.
- Prepare for comparison against MobileNetV3.

### Future Experiments

- Compare the baseline CNN against MobileNetV3.
- Investigate model compression techniques.
- Benchmark model size and inference latency.
- Analyze the trade-offs between accuracy and computational efficiency.