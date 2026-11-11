
from pathlib import Path
import json
import torch
import torch.nn as nn
from torchvision import models


# PROJECT FOLDERS
project_dir = Path(__file__).resolve().parent.parent
checkpoint_dir = project_dir / "checkpoints"
results_dir = project_dir / "results"


# SIMPLE CNN ARCHITECTURE
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        return self.fc(x)


# MOBILENETV3 ARCHITECTURE
def create_mobilenet():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, 10
    )
    return model


# EXPERIMENTS
experiments = {
    "SimpleCNN": (
        SimpleCNN(),
        "baseline_cnn_best.pth",
        "baseline_cnn_results.json"
    ),
    "MobileNetV3 original": (
        create_mobilenet(),
        "mobilenet_v3_lr001_best.pth",
        "mobilenet_v3_lr001_results.json"
    ),
    "MobileNetV3 lower LR": (
        create_mobilenet(),
        "mobilenet_v3_lr0001_best.pth",
        "mobilenet_v3_lr0001_results.json"
    ),
    "MobileNetV3 scheduler": (
        create_mobilenet(),
        "mobilenet_v3_scheduler_best.pth",
        "mobilenet_v3_scheduler_results.json"
    )
}


# COMPARISON
print("\nMODEL COMPARISON\n")

for name, (model, checkpoint_name, results_name) in experiments.items():

    parameters = sum(p.numel() for p in model.parameters())

    checkpoint_path = checkpoint_dir / checkpoint_name

    if checkpoint_path.exists():
        checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 ** 2)
    else:
        checkpoint_size_mb = None

    with open(results_dir / results_name, "r") as f:
        results = json.load(f)

    accuracy = results["saved_val_accuracy"]

    print(name)
    print(f"  Parameters: {parameters:,}")
    print(f"  Validation accuracy: {accuracy:.2f}%")

    if checkpoint_size_mb is not None:
        print(f"  Checkpoint size: {checkpoint_size_mb:.2f} MiB")
    else:
        print(f"  Checkpoint not found: {checkpoint_name}")

    print()
