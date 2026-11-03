from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

torch.manual_seed(42)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc(x)
        return x


model = SimpleCNN()


dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transforms.ToTensor()
)


train_data, val_data = random_split(
    dataset,
    [40000, 10000])

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True)

val_loader = DataLoader(
    val_data,
    batch_size=64,
    shuffle=False)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


checkpoint_dir = Path(__file__).resolve().parent.parent / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

best_val_accuracy = 0
best_epoch = 0

epochs = 10

train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(epochs):

    # TRAINING
    model.train()
    total_train_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * images.size(0)

    avg_train_loss = total_train_loss / len(train_data)
    train_losses.append(avg_train_loss)

    # VALIDATION
    model.eval()
    total_val_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()

    avg_val_loss = total_val_loss / len(val_data)
    val_accuracy = 100 * correct / len(val_data)

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            checkpoint_dir / "baseline_cnn_best.pth"
        )

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train loss: {avg_train_loss:.4f} | "
        f"Val loss: {avg_val_loss:.4f} | "
        f"Val accuracy: {val_accuracy:.2f}%")

print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print(f"Best epoch: {best_epoch}")