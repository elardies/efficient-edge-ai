# Efficient Edge AI

A machine learning engineering project exploring the trade-offs between **neural network accuracy, model size, and inference efficiency** on resource-constrained devices.

This project uses PyTorch to train and compare image classification models, investigate optimization strategies, and evaluate their suitability for edge deployment.

## Project Goal

Investigate how neural network architecture and optimization techniques affect classification accuracy, model size, and inference performance.

The project focuses on three questions:

1. How does a simple convolutional neural network compare with a mobile-oriented architecture?
2. How do training strategies affect model convergence and validation accuracy?
3. What trade-offs arise when optimizing neural networks for resource-constrained hardware?

## Project Progress

**November 2026 — Model Development and Evaluation**

The first stage of the project focused on learning PyTorch, implementing a baseline CNN, and comparing it with MobileNetV3-Small on CIFAR-10. Both models have been trained and evaluated, including additional experiments with learning rates and scheduling strategies.

The results and learning curves are documented below.

**Upcoming Work**

The next stage will focus on model compression and inference benchmarking, followed by an analysis of the trade-offs between accuracy, storage requirements, and inference speed.

## Technologies

- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib

## Phase 1: PyTorch Foundations

**Status: Completed — November 2, 2026**

Completed exercises included tensor operations, automatic differentiation, manual gradient descent, multilayer perceptrons, optimization, and model evaluation.

### Introductory MLP Experiment

A 1–8–1 multilayer perceptron was trained to approximate:

[
y = 3x + 2
]

**Experimental setup:**

- Dataset: 100 synthetic observations
- Split: 70 training, 15 validation, and 15 test observations
- Optimizer: Adam, learning rate 0.01
- Loss function: Mean Squared Error
- Training: 500 epochs

| Dataset    |     MSE |
| ---------- | ------: |
| Training   | 0.01986 |
| Validation | 0.02577 |
| Test       | 0.01769 |

The model achieved low prediction errors on both training and held-out observations.

These introductory exercises are separate from the main image classification experiments. The code is available in `learning/`.

## Phase 2: CIFAR-10 Image Classification

**Status: Training experiments completed — November 2026**

Two convolutional neural network architectures were trained on CIFAR-10:

- **SimpleCNN:** A small custom CNN used as the baseline.
- **MobileNetV3-Small:** A mobile-oriented convolutional architecture implemented using Torchvision.

Both architectures were trained from randomly initialized weights, without pretrained ImageNet parameters.

### Dataset and Experimental Setup

CIFAR-10 contains 60,000 color images across 10 classes.

The experiments use 40,000 training images and 10,000 validation images selected from the official 50,000-image training partition.

The official 10,000-image CIFAR-10 test partition was not used for the reported validation results.

| Parameter                   | Value                   |
| --------------------------- | ----------------------- |
| Dataset                     | CIFAR-10                |
| Training images             | 40,000                  |
| Validation images           | 10,000                  |
| Random seed                 | 42                      |
| Batch size                  | 64                      |
| Optimizer                   | Adam                    |
| Loss function               | CrossEntropyLoss        |
| Maximum epochs              | 30                      |
| Checkpoint selection        | Lowest validation loss  |
| Data preprocessing          | `transforms.ToTensor()` |
| SimpleCNN training device   | CPU                     |
| MobileNetV3 training device | Apple MPS               |

A fixed random seed and the same 40,000/10,000 split were used across experiments.

Training was stopped early when validation loss failed to improve for a specified number of epochs.

### Model Architectures

**SimpleCNN**

The baseline model contains:

- Convolutional layer: 3 → 16 channels
- ReLU activation and 2×2 max pooling
- Convolutional layer: 16 → 32 channels
- ReLU activation and 2×2 max pooling
- Flatten layer
- Fully connected classifier with 10 outputs

Total trainable parameters: **25,578**

**MobileNetV3-Small**

MobileNetV3-Small was initialized without pretrained weights, and its final classifier layer was replaced to predict 10 CIFAR-10 classes.

Total trainable parameters: **1,528,106**

### Model Comparison

The following table reports the accuracy of the saved checkpoint selected by minimum validation loss.

| Model             | Parameters | Checkpoint size | Validation accuracy |
| ----------------- | ---------: | --------------: | ------------------: |
| SimpleCNN         |     25,578 |        0.10 MiB |          **68.28%** |
| MobileNetV3-Small |  1,528,106 |        5.96 MiB |              63.11% |

The SimpleCNN achieved 5.17 percentage points higher validation accuracy while using approximately 59.7 times fewer parameters.

These results describe the specific architectures and training configurations evaluated in this project. They do not establish that SimpleCNN is generally more efficient than MobileNetV3.

Inference latency and runtime memory usage have not yet been benchmarked.

### MobileNetV3 Learning-Rate Experiments

Three training strategies were investigated:

1. Constant learning rate of 0.001.
2. Constant learning rate of 0.0001.
3. Initial learning rate of 0.001 with `ReduceLROnPlateau`.

| Experiment   | Best observed accuracy | Saved checkpoint accuracy | Epochs completed |
| ------------ | ---------------------: | ------------------------: | ---------------: |
| LR = 0.001   |                 63.11% |                **63.11%** |               28 |
| LR = 0.0001  |                 48.55% |                    47.82% |               14 |
| LR scheduler |                 62.73% |                    62.29% |               21 |

The original learning rate achieved the highest saved validation accuracy but produced substantial fluctuations during training.

Reducing the learning rate to 0.0001 produced a smoother learning trajectory but substantially lower validation accuracy within the training budget.

The scheduler experiment used `ReduceLROnPlateau` with a reduction factor of 0.5 and patience of 2 epochs. Early stopping patience was increased to 7 epochs, compared with 5 in the constant-learning-rate experiments.

The scheduler reduced learning-rate values during training and produced more stable later validation accuracy, but it did not outperform the original experiment.

The original MobileNetV3 checkpoint was selected for the primary model comparison.

### Learning Curves

#### SimpleCNN vs. MobileNetV3 — Validation Accuracy



#### Training Loss



#### Validation Loss



### MobileNetV3 Hyperparameter Analysis

#### Validation Accuracy Comparison



#### Training Loss Comparison



#### Validation Loss Comparison



#### Learning-Rate Schedule



### Checkpoint Verification

The selected SimpleCNN and MobileNetV3 checkpoints were reloaded and independently evaluated on the same 10,000-image validation split.

The verified accuracies were:

| Model             | Reproduced validation accuracy |
| ----------------- | -----------------------------: |
| SimpleCNN         |                         68.28% |
| MobileNetV3-Small |                         63.11% |

This confirms that the saved model weights reproduce the validation accuracies recorded during training.

## Reproducing the Experiments

### Requirements

- Python
- PyTorch
- Torchvision
- Matplotlib
- NumPy

### Dataset

Download the CIFAR-10 Python dataset from the official source:

[https://www.cs.toronto.edu/\~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

Extract the dataset so that the directory structure contains:

```text
experiments/
└── data/
    └── cifar-10-batches-py/
```

The dataset is excluded from Git version control.

### Training

The training scripts are located in `experiments/`.

Run the scripts with the working directory set to `experiments/`, because the current training implementations load CIFAR-10 from `./data`.

```bash
cd experiments
python baseline_cnn.py
python mobilenet_v3.py
```

**Important:** Running these scripts starts a new training experiment and may overwrite checkpoints or result files if the output filenames are unchanged. Preserve existing results before retraining.

The MobileNetV3 script can be configured with different learning rates and scheduler settings to reproduce the experimental variants.

### Analysis and Verification

The repository includes scripts for:

- Plotting learning curves from saved JSON results
- Comparing model parameter counts and checkpoint sizes
- Reloading and verifying saved checkpoints

Saved model checkpoints are stored locally and excluded from Git version control. JSON experiment results and generated figures document the reported measurements.

## Limitations

- Results are based on a single seeded training run per configuration.
- All reported classification accuracies are validation accuracies, not official CIFAR-10 test accuracies.
- Models were trained from random initialization rather than pretrained weights.
- The architectures differ substantially in parameter count and design.
- SimpleCNN and MobileNetV3 were trained on different devices, so their training times are not directly comparable.
- Inference latency, memory consumption, and energy efficiency have not yet been evaluated.
- Hyperparameter tuning was exploratory rather than exhaustive.

## Next Steps

The next phase will investigate model compression, including techniques such as quantization, followed by inference benchmarking on consistent hardware.

The final analysis will examine the trade-offs among classification accuracy, model storage size, inference latency, and practical edge deployment constraints.this&#x20;
