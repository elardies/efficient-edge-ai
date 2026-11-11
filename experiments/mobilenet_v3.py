
from pathlib import Path
import time
import json

import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader, random_split


# REPRODUCIBILITY
torch.manual_seed(42)

# DEVICE
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)
print("Using device:", device, flush=True)


# MODEL
model = models.mobilenet_v3_small(weights=None)

# Adapt MobileNetV3 to CIFAR-10 (10 classes)
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features,
    10
)

model = model.to(device)

parameters = sum(p.numel() for p in model.parameters())
print("Total parameters:", parameters)


# DATASET
dataset = datasets.CIFAR10(
    root=str(Path(__file__).resolve().parent / "data"),
    train=True,
    download=False,
    transform=transforms.ToTensor()
)

train_data, val_data = random_split(
    dataset,
    [40000, 10000],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(
    train_data,
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=64,
    shuffle=False
)

print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))


# TRAINING SETUP
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


epochs = 30
patience = 7
epochs_without_improvement = 0

best_val_loss = float("inf")
best_val_accuracy = 0
best_epoch = 0
saved_val_accuracy = 0
saved_epoch = 0

train_losses = []
val_losses = []
val_accuracies = []
learning_rates = []

project_dir = Path(__file__).resolve().parent.parent

checkpoint_dir = project_dir / "checkpoints"
checkpoint_dir.mkdir(exist_ok=True)

results_dir = project_dir / "results"
results_dir.mkdir(exist_ok=True)


# TRAINING
start_time = time.perf_counter()

for epoch in range(epochs):

    model.train()
    total_train_loss = 0

    for batch_idx, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * images.size(0)

        if (batch_idx + 1) % 100 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Batch {batch_idx + 1}/{len(train_loader)}",
                flush=True
            )

    avg_train_loss = total_train_loss / len(train_data)
    train_losses.append(avg_train_loss)

    # VALIDATION
    model.eval()
    total_val_loss = 0
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_val_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()

    avg_val_loss = total_val_loss / len(val_data)
    val_accuracy = 100 * correct / len(val_data)
    learning_rates.append(optimizer.param_groups[0]["lr"])
    scheduler.step(avg_val_loss)

    # TRACK BEST ACCURACY
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

    # SAVE BEST MODEL BY VALIDATION LOSS
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_without_improvement = 0

        saved_val_accuracy = val_accuracy
        saved_epoch = epoch + 1

        torch.save(
            model.state_dict(),
            checkpoint_dir / "mobilenet_v3_scheduler_best.pth"
        )
    else:
        epochs_without_improvement += 1

    val_losses.append(avg_val_loss)
    val_accuracies.append(val_accuracy)

    print(
        f"Epoch {epoch + 1}/{epochs} | "
        f"Train loss: {avg_train_loss:.4f} | "
        f"Val loss: {avg_val_loss:.4f} | "
        f"Val accuracy: {val_accuracy:.2f}%",
        flush=True
    )
    print(f"Learning rate for next epoch: {optimizer.param_groups[0]['lr']:.6f}")

    # SAVE RESULTS AFTER EVERY EPOCH
    training_time = time.perf_counter() - start_time

    results = {
        "device": str(device),
        "parameters": parameters,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "best_val_accuracy": best_val_accuracy,
        "best_epoch": best_epoch,
        "saved_val_accuracy": saved_val_accuracy,
        "saved_epoch": saved_epoch,
        "best_val_loss": best_val_loss,
        "training_time_seconds": training_time,
        "epochs_completed": len(train_losses),
        "learning_rates": learning_rates,
        "initial_learning_rate": 0.001,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_factor": 0.5,
        "scheduler_patience": 2,
        "early_stopping_patience": 7,
    }

    with open(results_dir / "mobilenet_v3_scheduler_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # EARLY STOPPING
    if epochs_without_improvement >= patience:
        print("Early stopping triggered.")
        break


# FINAL RESULTS
training_time = time.perf_counter() - start_time

print("\nTraining finished!")
print(f"Training time: {training_time:.2f} seconds")
print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
print(f"Best epoch: {best_epoch}")
print(f"Saved checkpoint epoch: {saved_epoch}")
print(f"Saved checkpoint accuracy: {saved_val_accuracy:.2f}%")

