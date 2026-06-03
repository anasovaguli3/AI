"""
predict.py — Bitta rasm uchun prediksiya
- Rasm yo'lini argument sifatida qabul qiladi
- Model bilan sinf va ishonch foizini chiqaradi
- Natijani JSON formatda ko'rsatadi
"""

import argparse
import json
import torch
from PIL import Image
from torchvision import transforms

from utils import load_config, get_device
from model import CastCNN


def predict_image(image_path: str, config: dict) -> dict:
    """
    Bitta rasm uchun prediksiya qiladi.

    Args:
        image_path: Rasm fayl yo'li
        config: Konfiguratsiya dict

    Returns:
        dict: predicted_class, confidence, probabilities
    """
    device = get_device()
    classes = config["data"]["classes"]
    img_size = config["image"]["size"]
    mean = config["image"]["mean"]
    std = config["image"]["std"]

    # Transformatsiya
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Rasmni yuklash
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Modelni yuklash
    num_classes = len(classes)
    model = CastCNN(num_classes=num_classes, dropout=config["model"]["dropout"]).to(device)

    model_path = config["output"]["best_model"]
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Prediksiya
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)

    predicted_class = classes[predicted_idx.item()]
    confidence_value = confidence.item()

    # Barcha sinflar ehtimolliklari
    probs = {cls: round(probabilities[0][i].item(), 4) for i, cls in enumerate(classes)}

    result = {
        "image": image_path,
        "predicted_class": predicted_class,
        "confidence": round(confidence_value, 4),
        "probabilities": probs,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description="Bitta rasm uchun quyma detal prediksiyasi")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Rasm fayl yo'li (masalan: data/processed/test/defect/img_00001.jpeg)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Config fayl yo'li (default: config.yaml)",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    result = predict_image(args.image, config)

    # Natijani chiqarish
    print("\n" + "=" * 50)
    print("PREDIKSIYA NATIJASI")
    print("=" * 50)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("=" * 50)

    # Qisqa natija
    emoji = "⚠️" if result["predicted_class"] == "defect" else "✅"
    print(f"\n{emoji} Sinf: {result['predicted_class'].upper()} — {result['confidence']*100:.1f}%")


if __name__ == "__main__":
    main()
