"""
utils.py — Yordamchi funksiyalar
- Config yuklash (YAML)
- Tasodifiy seed o'rnatish
- Qurilma (device) aniqlash
"""

import os
import random
import yaml
import torch
import numpy as np


def load_config(config_path: str = "config.yaml") -> dict:
    """YAML konfiguratsiya faylini yuklaydi."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int = 42):
    """Barcha tasodifiy generatorlar uchun seed o'rnatadi (takrorlanish uchun)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Mavjud eng yaxshi qurilmani aniqlaydi: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] Qurilma: CUDA ({torch.cuda.get_device_name(0)})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] Qurilma: MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("[INFO] Qurilma: CPU")
    return device


def ensure_dirs(config: dict):
    """Chiqish papkalarini yaratadi (agar mavjud bo'lmasa)."""
    dirs_to_create = [
        config["output"]["models_dir"],
        config["output"]["plots_dir"],
        config["output"]["metrics_dir"],
        config["data"]["processed_dir"],
    ]
    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)


if __name__ == "__main__":
    # Test
    cfg = load_config()
    print(f"Config yuklandi: {cfg['model']['name']}")
    set_seed(cfg["seed"])
    print("Seed o'rnatildi.")
    device = get_device()
    print(f"Device: {device}")
    ensure_dirs(cfg)
    print("Papkalar yaratildi.")
