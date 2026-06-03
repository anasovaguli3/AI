"""
train.py — Modelni o'qitish (Early Stopping bilan)
- CastCNN modelini yaratadi
- Train va validation DataLoader lar bilan o'qitadi
- Eng yaxshi modelni saqlaydi
- Loss/Accuracy grafiklarini chizadi
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_config, set_seed, get_device, ensure_dirs
from model import CastCNN
from dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Bitta epoch o'qitish."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    """Validation bo'yicha baholash."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Val  ", leave=False):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_history(history: dict, output_dir: str):
    """Loss va Accuracy grafiklarini chizadi va saqlaydi."""
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss grafik
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], "b-", label="Train Loss")
    plt.plot(epochs, history["val_loss"], "r-", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss grafik")
    plt.legend()
    plt.grid(True)

    # Accuracy grafik
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], "b-", label="Train Acc")
    plt.plot(epochs, history["val_acc"], "r-", label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy grafik")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "loss_accuracy.png"), dpi=150)
    plt.close()
    print(f"[INFO] Grafiklar saqlandi: {output_dir}/loss_accuracy.png")


def train(config: dict):
    """Asosiy o'qitish funksiyasi."""

    # Sozlamalar
    seed = config["seed"]
    epochs = config["training"]["epochs"]
    lr = config["training"]["learning_rate"]
    patience = config["training"]["early_stop_patience"]
    dropout = config["model"]["dropout"]

    set_seed(seed)
    device = get_device()
    ensure_dirs(config)

    # DataLoader lar
    print("\n[1/4] Ma'lumotlar yuklanmoqda...")
    loaders = get_dataloaders(config)

    if "train" not in loaders or "val" not in loaders:
        raise RuntimeError("[XATO] Train yoki Val papkasi topilmadi. Avval split_data.py ni ishga tushiring!")

    # Model
    print("\n[2/4] Model yaratilmoqda...")
    num_classes = len(config["data"]["classes"])
    model = CastCNN(num_classes=num_classes, dropout=dropout).to(device)
    print(f"  Parametrlar: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Loss va Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # O'qitish
    print(f"\n[3/4] O'qitish boshlanmoqda (epochs={epochs}, patience={patience})...\n")
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, loaders["train"], criterion, optimizer, device
        )
        val_loss, val_acc = validate(
            model, loaders["val"], criterion, device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"  Epoch [{epoch:02d}/{epochs}] | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Train Acc: {train_acc*100:.2f}% | Val Acc: {val_acc*100:.2f}%"
        )

        # Eng yaxshi modelni saqlash
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), config["output"]["best_model"])
            print(f"    ✓ Eng yaxshi model saqlandi (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  ⚠ Early Stopping! {patience} epoch davomida yaxshilanish bo'lmadi.")
                break

    # Oxirgi modelni saqlash
    torch.save(model.state_dict(), config["output"]["last_model"])
    print(f"\n[INFO] Oxirgi model saqlandi: {config['output']['last_model']}")

    # Grafiklar
    print("\n[4/4] Grafiklar chizilmoqda...")
    plot_history(history, config["output"]["plots_dir"])

    # History ni saqlash
    history_path = os.path.join(config["output"]["metrics_dir"], "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[INFO] Trening tarixi saqlandi: {history_path}")

    print("\n" + "=" * 50)
    print("O'QITISH YAKUNLANDI!")
    print(f"  Eng yaxshi Val Loss: {best_val_loss:.4f}")
    print(f"  Jami epoch: {len(history['train_loss'])}")
    print("=" * 50)


if __name__ == "__main__":
    config = load_config()
    train(config)
