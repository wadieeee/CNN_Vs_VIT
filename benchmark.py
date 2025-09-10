import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from models.resnet_from_scratch import ResNet18
from models.vit_from_scratch import DeiTTiny

# -----------------------------
# Device
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# CIFAR-10 dataset
# -----------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# -----------------------------
# Training + Evaluation
# -----------------------------
def train(model, optimizer, criterion, train_loader, device):
    model.train()
    running_loss, correct, total = 0, 0, 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)[0] if isinstance(model, DeiTTiny) else model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100. * correct / total

def evaluate(model, criterion, test_loader, device):
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)[0] if isinstance(model, DeiTTiny) else model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return running_loss / total, 100. * correct / total

# -----------------------------
# Benchmark loop
# -----------------------------
models = {
    "ResNet18": ResNet18(num_classes=10).to(device),
    "DeiT-Tiny": DeiTTiny(num_classes=10).to(device)
}

criterion = nn.CrossEntropyLoss()
num_epochs = 5
results = {}
history = {name: {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []} for name in models}

for name, model in models.items():
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(f"\n===== Training {name} =====")
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, optimizer, criterion, train_loader, device)
        test_loss, test_acc = evaluate(model, criterion, test_loader, device)
        
        history[name]["train_loss"].append(train_loss)
        history[name]["train_acc"].append(train_acc)
        history[name]["test_loss"].append(test_loss)
        history[name]["test_acc"].append(test_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% || "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
    results[name] = test_acc

# -----------------------------
# Final results
# -----------------------------
print("\n===== Benchmark Results =====")
for name, acc in results.items():
    print(f"{name}: {acc:.2f}% Test Accuracy")

# -----------------------------
# Plot training curves
# -----------------------------
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))

# Loss
plt.subplot(1, 2, 1)
for name in models:
    plt.plot(epochs, history[name]["train_loss"], label=f"{name} Train")
    plt.plot(epochs, history[name]["test_loss"], linestyle="--", label=f"{name} Test")
plt.title("Loss vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# Accuracy
plt.subplot(1, 2, 2)
for name in models:
    plt.plot(epochs, history[name]["train_acc"], label=f"{name} Train")
    plt.plot(epochs, history[name]["test_acc"], linestyle="--", label=f"{name} Test")
plt.title("Accuracy vs Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.legend()

plt.tight_layout()
plt.show()
