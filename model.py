"""
model.py — CastCNN Arxitekturasi
3 ta konvolyutsion blok + AdaptiveAvgPool + Classifier
Kirish: RGB rasm (3, 224, 224) → Chiqish: 2 sinf (defect, normal)
"""

import torch
import torch.nn as nn


class CastCNN(nn.Module):
    """
    Quyma detal nosozliklarini aniqlash uchun CNN model.

    Arxitektura:
        Block 1: Conv2d(3→32) → BatchNorm → ReLU → MaxPool
        Block 2: Conv2d(32→64) → BatchNorm → ReLU → MaxPool
        Block 3: Conv2d(64→96) → BatchNorm → ReLU → MaxPool
        Pool:    AdaptiveAvgPool2d(4×4)
        FC:      Linear(1536→128) → ReLU → Dropout → Linear(128→2)
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.4):
        super(CastCNN, self).__init__()

        # Block 1: 3 → 32
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2: 32 → 64
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3: 64 → 96
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Adaptive pooling
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)   # [B, 32, 112, 112]
        x = self.block2(x)   # [B, 64, 56, 56]
        x = self.block3(x)   # [B, 96, 28, 28]
        x = self.pool(x)     # [B, 96, 4, 4]
        x = self.classifier(x)  # [B, 2]
        return x


def count_parameters(model: nn.Module) -> int:
    """Modelning o'qitiladigan parametrlar sonini qaytaradi."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Model testi
    model = CastCNN(num_classes=2, dropout=0.4)
    print(model)
    print(f"\nParametrlar soni: {count_parameters(model):,}")

    # Sinov kiritish
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Kirish o'lchami:  {dummy_input.shape}")
    print(f"Chiqish o'lchami: {output.shape}")
    print(f"Chiqish:          {output}")
