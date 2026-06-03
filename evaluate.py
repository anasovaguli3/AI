"""
evaluate.py — Test to'plamida modelni baholash
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix grafigi
- Natijalarni JSON va TXT ga saqlash
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from tqdm import tqdm

from utils import load_config, set_seed, get_device, ensure_dirs
from model import CastCNN
from dataset import get_dataloaders


def evaluate(config: dict):
    """Test to'plamida modelni baholaydi."""

    set_seed(config["seed"])
    device = get_device()
    ensure_dirs(config)

    # DataLoader
    print("\n[1/3] Test ma'lumotlari yuklanmoqda...")
    loaders = get_dataloaders(config)

    if "test" not in loaders:
        raise RuntimeError("[XATO] Test papkasi topilmadi. Avval split_data.py ni ishga tushiring!")

    test_loader = loaders["test"]

    # Model yuklash
    print("\n[2/3] Model yuklanmoqda...")
    num_classes = len(config["data"]["classes"])
    model = CastCNN(num_classes=num_classes, dropout=config["model"]["dropout"]).to(device)

    model_path = config["output"]["best_model"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[XATO] Model topilmadi: {model_path}. Avval train.py ni ishga tushiring!")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"  Model yuklandi: {model_path}")

    # Prediksiya
    print("\n[3/3] Baholash boshlandi...")
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="  Test"):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Metrikalar
    classes = config["data"]["classes"]
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average="weighted")
    rec = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    print("\n" + "=" * 50)
    print("TEST NATIJALARI")
    print("=" * 50)
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  F1-Score:  {f1*100:.2f}%")
    print(f"  Jami test rasmlari: {len(all_labels)}")
    print("=" * 50)

    # Classification Report
    report = classification_report(all_labels, all_preds, target_names=classes)
    print(f"\nBatafsil hisobot:\n{report}")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Bashorat qilingan")
    plt.ylabel("Haqiqiy")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    cm_path = os.path.join(config["output"]["plots_dir"], "confusion_matrix.png")
    plt.savefig(cm_path, dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saqlandi: {cm_path}")

    # Natijalarni saqlash
    metrics = {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "total_test_samples": int(len(all_labels)),
        "confusion_matrix": cm.tolist(),
    }

    metrics_json_path = os.path.join(config["output"]["metrics_dir"], "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrikalar saqlandi: {metrics_json_path}")

    report_path = os.path.join(config["output"]["metrics_dir"], "report.txt")
    with open(report_path, "w") as f:
        f.write("QUYMA DETAL CNN — TEST NATIJALARI\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Accuracy:  {acc*100:.2f}%\n")
        f.write(f"Precision: {prec*100:.2f}%\n")
        f.write(f"Recall:    {rec*100:.2f}%\n")
        f.write(f"F1-Score:  {f1*100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"[INFO] Hisobot saqlandi: {report_path}")


if __name__ == "__main__":
    config = load_config()
    evaluate(config)
