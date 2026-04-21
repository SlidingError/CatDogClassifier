# backend/training/resnet18_cat_dog.py

from pathlib import Path
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent              # backend/training
BACKEND_DIR = BASE_DIR.parent                          # backend
DATA_DIR = BACKEND_DIR / "data"
MODEL_DIR = BACKEND_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "resnet18_catdog.pth"


# --------------------------------------------------
# Training / Evaluation
# --------------------------------------------------
def train_epoch(model, device, loader, optimizer, criterion):
    model.train()

    running_loss = 0.0

    for data, target in loader:
        data = data.to(device)
        target = target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)

    return running_loss / len(loader.dataset)


def evaluate(model, device, loader, criterion):
    model.eval()

    running_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data, target in loader:
            data = data.to(device)
            target = target.to(device)

            output = model(data)
            loss = criterion(output, target)

            running_loss += loss.item() * data.size(0)

            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()

    loss = running_loss / len(loader.dataset)
    accuracy = 100.0 * correct / len(loader.dataset)

    return loss, accuracy


# --------------------------------------------------
# Model
# --------------------------------------------------
class ResNet18CatDog(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        self.resnet.fc = nn.Linear(
            self.resnet.fc.in_features,
            num_classes
        )

    def forward(self, x):
        return self.resnet(x)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Using device: {device}")

    batch_size = 32
    epochs = 1
    lr = 1e-4

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="trainval",
        target_types="binary-category",
        transform=train_transform,
        download=True
    )

    test_dataset = datasets.OxfordIIITPet(
        root=str(DATA_DIR),
        split="test",
        target_types="binary-category",
        transform=test_transform,
        download=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = ResNet18CatDog().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr
    )

    criterion = nn.CrossEntropyLoss()

    start = time.time()

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model,
            device,
            train_loader,
            optimizer,
            criterion
        )

        test_loss, test_acc = evaluate(
            model,
            device,
            test_loader,
            criterion
        )

        print(
            f"Epoch {epoch}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Accuracy: {test_acc:.2f}%"
        )

    elapsed = time.time() - start

    print("Training complete.")
    print(f"Training time: {elapsed:.2f}s")

    torch.save(model.state_dict(), MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()