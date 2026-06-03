"""
dataset.py — Rasmlarni yuklash, preprocessing va augmentatsiya
- CastDataset: PyTorch Dataset klassi
- get_transforms: train/val/test uchun transformatsiyalar
- get_dataloaders: DataLoader yaratish
"""

import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import load_config


class CastDataset(Dataset):
    """
    Quyma detal rasmlarini yuklash uchun Dataset.
    Papka strukturasi:
        processed/train/defect/img001.jpeg
        processed/train/normal/img002.jpeg
    """

    def __init__(self, root_dir: str, transform=None):
        """
        Args:
            root_dir: train, val yoki test papkasi yo'li
            transform: torchvision.transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                img_path = os.path.join(cls_dir, img_name)
                if img_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms(config: dict, mode: str = "train"):
    """
    Transformatsiyalarni qaytaradi.

    Args:
        config: config.yaml dan yuklangan dict
        mode: 'train', 'val' yoki 'test'

    Returns:
        torchvision.transforms.Compose
    """
    img_size = config["image"]["size"]
    mean = config["image"]["mean"]
    std = config["image"]["std"]

    if mode == "train":
        transform_list = [
            transforms.Resize((img_size, img_size)),
        ]
        # Augmentatsiya
        if config["augmentation"].get("flip", False):
            transform_list.append(transforms.RandomHorizontalFlip())
        rotation = config["augmentation"].get("rotation", 0)
        if rotation > 0:
            transform_list.append(transforms.RandomRotation(rotation))

        transform_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    else:
        # Val / Test — augmentatsiyasiz
        transform_list = [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]

    return transforms.Compose(transform_list)


def get_dataloaders(config: dict) -> dict:
    """
    Train, val, test DataLoader larni yaratadi.

    Returns:
        dict: {"train": DataLoader, "val": DataLoader, "test": DataLoader}
    """
    processed_dir = config["data"]["processed_dir"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"].get("num_workers", 2)

    loaders = {}
    for split in ["train", "val", "test"]:
        split_dir = os.path.join(processed_dir, split)
        if not os.path.exists(split_dir):
            print(f"[OGOHLANTIRISH] {split_dir} papkasi topilmadi, o'tkazildi.")
            continue

        transform = get_transforms(config, mode=split)
        dataset = CastDataset(root_dir=split_dir, transform=transform)

        shuffle = (split == "train")
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
        print(f"[INFO] {split}: {len(dataset)} ta rasm, {len(loaders[split])} batch")

    return loaders


if __name__ == "__main__":
    config = load_config()
    loaders = get_dataloaders(config)

    if "train" in loaders:
        images, labels = next(iter(loaders["train"]))
        print(f"Batch o'lchami: {images.shape}")
        print(f"Yorliqlar: {labels}")
