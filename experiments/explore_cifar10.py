import torch
from torchvision import datasets, transforms

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=False,
    transform=transforms.ToTensor())

print("Number of training images: ", len(train_dataset))

image, label = train_dataset[0]

print("Image shape:", image.shape)
print("Label:", label)
print("Class:", train_dataset.classes[label])

import matplotlib.pyplot as plt

plt.imshow(image.permute(1, 2, 0))
plt.title(train_dataset.classes[label])
plt.axis("off")
plt.show()

from torch.utils.data import DataLoader

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

images, labels = next(iter(train_loader))

print("Batch image shape:", images.shape)
print("Batch labels shape:", labels.shape)

from torch.utils.data import DataLoader, random_split

torch.manual_seed(42)

train_data, val_data = random_split(
    train_dataset,
    [40000, 10000]
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

print("Training images:", len(train_data))
print("Validation images:", len(val_data))

images, labels = next(iter(train_loader))
print("Training batch shape:", images.shape)