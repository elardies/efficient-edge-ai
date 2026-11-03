
from pathlib import Path
import matplotlib.pyplot as plt

# Results from our completed 10-epoch experiment
train_losses = [
    1.6160, 1.3217, 1.1925, 1.0931, 1.0256,
    0.9699, 0.9328, 0.8960, 0.8715, 0.8464
]

val_losses = [
    1.3947, 1.2996, 1.1756, 1.1148, 1.0554,
    1.0385, 1.0083, 1.0102, 0.9736, 0.9894
]

val_accuracies = [
    51.02, 54.46, 58.79, 60.93, 63.32,
    63.72, 65.10, 65.10, 66.23, 66.01
]

epochs = range(1, 11)

# Create figures folder at project root
figures_dir = Path(__file__).resolve().parent.parent / "figures"
figures_dir.mkdir(exist_ok=True)

# Plot training and validation loss
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_losses, marker="o", label="Training loss")
plt.plot(epochs, val_losses, marker="o", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Cross-entropy loss")
plt.title("Baseline CNN - CIFAR-10 Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(figures_dir / "baseline_cnn_loss.png", dpi=150)
plt.close()

# Plot validation accuracy
plt.figure(figsize=(8, 5))
plt.plot(epochs, val_accuracies, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Validation accuracy (%)")
plt.title("Baseline CNN - CIFAR-10 Accuracy")
plt.grid(True)
plt.tight_layout()
plt.savefig(figures_dir / "baseline_cnn_accuracy.png", dpi=150)
plt.close()

print("Both graphs saved successfully!")
print("Best validation accuracy: 66.23% (epoch 9)")
