"""
split_data.py — Datasetni train/val/test ga bo'lish (80/10/10)
- label.csv dan rasmlarni o'qiydi
- Har bir sinfni alohida aralashtiradi
- Stratified bo'lish (sinf nisbati saqlanadi)
- Rasmlarni processed/ papkasiga nusxalaydi
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

from utils import load_config, set_seed


def split_dataset(config: dict):
    """Ma'lumotlarni train/val/test ga bo'ladi va papkalarga nusxalaydi."""

    seed = config["seed"]
    set_seed(seed)

    # Yo'llar
    raw_dir = config["data"]["raw_dir"]
    label_file = config["data"]["label_file"]
    processed_dir = config["data"]["processed_dir"]
    image_col = config["data"]["image_column"]
    label_col = config["data"]["label_column"]

    # Nisbatlar
    train_ratio = config["split"]["train_ratio"]
    val_ratio = config["split"]["val_ratio"]

    # label.csv ni o'qish
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"[XATO] Label fayl topilmadi: {label_file}")

    df = pd.read_csv(label_file)
    print(f"[INFO] Jami rasmlar: {len(df)}")
    print(f"[INFO] Sinflar taqsimoti:\n{df[label_col].value_counts()}\n")

    # Mavjud rasmlarni filtrlash
    df["exists"] = df[image_col].apply(
        lambda x: os.path.exists(os.path.join(raw_dir, x))
    )
    missing = df[~df["exists"]]
    if len(missing) > 0:
        print(f"[OGOHLANTIRISH] {len(missing)} ta rasm topilmadi, o'tkazildi.")
    df = df[df["exists"]].drop(columns=["exists"])

    # Stratified split: train + temp
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        stratify=df[label_col],
        random_state=seed,
    )

    # temp ni val + test ga bo'lish
    val_relative = val_ratio / (1 - train_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_relative),
        stratify=temp_df[label_col],
        random_state=seed,
    )

    print(f"[INFO] Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Papkalarni yaratish va rasmlarni nusxalash
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for cls in config["data"]["classes"]:
            cls_dir = os.path.join(processed_dir, split_name, cls)
            os.makedirs(cls_dir, exist_ok=True)

        for _, row in split_df.iterrows():
            src = os.path.join(raw_dir, row[image_col])
            dst_dir = os.path.join(processed_dir, split_name, row[label_col])
            dst = os.path.join(dst_dir, row[image_col])
            shutil.copy2(src, dst)

    print("\n[TAYYOR] Ma'lumotlar muvaffaqiyatli bo'lindi!")
    print(f"  Train: {os.path.join(processed_dir, 'train')}")
    print(f"  Val:   {os.path.join(processed_dir, 'val')}")
    print(f"  Test:  {os.path.join(processed_dir, 'test')}")


if __name__ == "__main__":
    config = load_config()
    split_dataset(config)
