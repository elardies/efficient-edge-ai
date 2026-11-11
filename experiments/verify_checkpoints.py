
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split


# PROJECT PATHS
project_dir = Path(__file__).resolve().parent.parent
checkpoint_dir = project_dir / "checkpoints"

torch.manual_seed(42)


# SIMPLE CNN
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        return self.fc(x)


# MOBILENETV3
def create_mobilenet():
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, 10
    )
    return model


# VALIDATION DATASET
dataset = datasets.CIFAR10(
    root=str(project_dir / "experiments" / "data"),
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

_, val_data = random_split(
    dataset,
    [40000, 10000],
    generator=torch.Generator().manual_seed(42)
)

val_loader = DataLoader(
    val_data,
    batch_size=64,
    shuffle=False
)


# VERIFICATION FUNCTION
def verify_model(model, checkpoint_name):
    checkpoint_path = checkpoint_dir / checkpoint_name

    model.load_state_dict(
        torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True
        )
    )

    model.eval()
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()

    return 100 * correct / len(val_data)


# VERIFY
print("Verifying saved checkpoints...\n")

baseline_accuracy = verify_model(
    SimpleCNN(),
    "baseline_cnn_best.pth"
)

mobilenet_accuracy = verify_model(
    create_mobilenet(),
    "mobilenet_v3_lr001_best.pth"
)

print(f"SimpleCNN checkpoint accuracy: {baseline_accuracy:.2f}%")
print(f"MobileNetV3 checkpoint accuracy: {mobilenet_accuracy:.2f}%")
