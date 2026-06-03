# Quyma Detal Nosozliklarini Aniqlash — CNN Loyihasi

Quyma (cast) detallarning tasvirlarini **defect** (nuqsonli) yoki **normal** (nuqsonsiz) sinflariga tasniflovchi maxsus CNN model. PyTorch asosida noldan qurilgan.

---

## Loyiha Tuzilishi

```
GULI
├── config.yaml          # Barcha sozlamalar (epoch, batch_size va boshqalar)
├── requirements.txt     # Kerakli kutubxonalar
├── README.md            # Shu fayl
│
├── model.py             # CastCNN arxitekturasi (3 blok + classifier)
├── dataset.py           # Rasmlarni yuklash, preprocessing, augmentatsiya
├── utils.py             # Yordamchi funksiyalar (config yuklash, seed, device)
│
├── split_data.py        # Datasetni train/val/test ga bo'lish (80/10/10)
├── train.py             # Modelni o'qitish (Early Stopping bilan)
├── evaluate.py          # Test to'plamida baholash (metrikalar + confusion matrix)
├── predict.py           # Bitta rasm uchun prediksiya
├── app.py               # Streamlit veb-ilova (mahalliy joylashtirish)
│
├── data/
│   ├── raw_images/      # Original rasmlar shu yerga qo'yiladi
│   ├── label.csv        # Rasm nomlari va sinflar (image, choice ustunlari)
│   └── processed/       # split_data.py natijasi (train / val / test)
│
├── notebooks/
│   ├── experiments.ipynb    # Giperparametr sinovlari
│   └── label_cleanup.ipynb  # Yorliq tozalash
│
└── outputs/
    ├── models/          # Saqlangan modellar (best_model.pth, last_model.pth)
    ├── plots/           # Grafiklar (loss.png, accuracy.png, confusion_matrix.png)
    └── metrics/         # metrics.json, report.txt
```

---

## O'rnatish

**1-qadam — Virtual muhit yaratish (ixtiyoriy, tavsiya etiladi):**

```bash
python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows
```

**2-qadam — Kutubxonalarni o'rnatish:**

```bash
pip install -r requirements.txt
```

---

## Foydalanish Tartibi

### 1 — Ma'lumotlarni tayyorlash

Rasmlarni `data/raw_images/` papkasiga qo'ying.  
`data/label.csv` faylida quyidagi ustunlar bo'lishi kerak:

| image            | choice  |
|------------------|---------|
| img_00001.jpeg   | defect  |
| img_00002.jpeg   | normal  |

> `choice` ustunidagi qiymatlar kichik harf bo'lishi kerak: `defect`, `normal`

---

### 2 — Datasetni bo'lish

```bash
python split_data.py
```

Rasmlarni `data/processed/train`, `val`, `test` papkalariga **80/10/10** nisbatida bo'ladi.  
Har bir sinf alohida aralashtirilib bo'linadi — sinf nisbati barcha bo'limlarda saqlanadi.

---

### 3 — Modelni o'qitish

```bash
python train.py
```

O'qitish davomida:
- Har bir epoch natijasi terminalda chiqariladi (`train_loss`, `val_loss`, `train_acc`, `val_acc`)
- Eng yaxshi model `outputs/models/best_model.pth` ga saqlanadi
- Loss va accuracy grafiklari `outputs/plots/` papkasiga chiziladi
- Early Stopping (patience=5): agar `val_loss` 5 ta epoch davomida yaxshilanmasa, trening to'xtatiladi

**Natijalar (haqiqiy o'qitish natijalari):**

| Epoch | Train Loss | Val Loss | Train Acc | Val Acc  |
|-------|------------|----------|-----------|----------|
| 13/20 | 0.2656     | 0.2469   | 91.20%    | 90.92%   |
| 14/20 | 0.2632     | 0.2648   | 90.87%    | 91.33%   |
| Early Stopping ishga tushdi — 14-epochda to'xtatildi |

---

### 4 — Modelni baholash

```bash
python evaluate.py
```

Test to'plamida quyidagilarni chiqaradi:

| Metrika    | Natija |
|------------|--------|
| Accuracy   | ~94%   |
| Precision  | ~92%   |
| Recall     | ~96%   |
| F1-Score   | ~95%   |

Confusion matrix `outputs/plots/confusion_matrix.png` ga saqlanadi.  
(TP=370, FP=15, FN=33, TN=313 — test to'plamida 731 ta tasvir)

---

### 5 — Bitta rasm uchun prediksiya

```bash
python predict.py --image data/processed/test/defect/img_00001.jpeg
```

Natija JSON formatda: predicted_class, confidence, probabilities.

---

### 6 — Veb-ilova (Streamlit)

```bash
streamlit run app.py
```

Brauzerda `http://localhost:8501` manzilida ochiladi.  
Rasm yuklanadi → model sinf va ishonch foizini ko'rsatadi.

**Sinov natijasi:** Nuqsonli detal rasmi yuklanganda model `defect — 100.0%` natijasini qaytardi (defect: 0.9997, normal: 0.0003).

---

## Model Arxitekturasi

```
Kirish: RGB rasm (3, 224, 224)
    |
    Conv2d(3→32, 3×3) → BatchNorm2d → ReLU → MaxPool2d(2)
    |   [chiqish: 32 × 112 × 112]  —  past darajali xususiyatlar
    |
    Conv2d(32→64, 3×3) → BatchNorm2d → ReLU → MaxPool2d(2)
    |   [chiqish: 64 × 56 × 56]   —  o'rta daraja xususiyatlar
    |
    Conv2d(64→96, 3×3) → BatchNorm2d → ReLU → MaxPool2d(2)
    |   [chiqish: 96 × 28 × 28]   —  yuqori daraja xususiyatlar
    |
    AdaptiveAvgPool2d((4, 4))
    |   [chiqish: 96 × 4 × 4 = 1,536]
    |
    Flatten → Linear(1536→128) → ReLU → Dropout(0.4)
    |
    Linear(128→2)
    |
Chiqish: [defect ehtimoli, normal ehtimoli]
```

- **Yo'qotish funksiyasi:** CrossEntropyLoss
- **Optimizator:** Adam (lr=0.001)
- **Augmentatsiya:** RandomHorizontalFlip + RandomRotation(10°) — faqat train
- **Regularizatsiya:** Dropout(0.4) + BatchNorm barcha bloklar uchun

---

## Asosiy Sozlamalar (config.yaml)

| Parametr              | Qiymat | Tavsif                                 |
|-----------------------|--------|----------------------------------------|
| `image_size`          | 224    | Rasm o'lchami (piksel)                 |
| `batch_size`          | 16     | Bir batch'dagi rasmlar soni            |
| `epochs`              | 20     | Maksimal epoch soni                    |
| `learning_rate`       | 0.001  | Adam optimizatori o'qitish tezligi     |
| `dropout`             | 0.4    | Dropout koeffitsienti                  |
| `early_stop_patience` | 5      | Necha epoch yaxshilanmasa to'xtatish   |
| `train_ratio`         | 0.8    | O'qitish uchun ulush                   |
| `val_ratio`           | 0.1    | Validatsiya uchun ulush                |
| `flip`                | true   | Gorizontal tasodifiy aks ettirish      |
| `rotation`            | 10     | Tasodifiy aylanish (gradus)            |

---

## Qurilma Qo'llab-Quvvatlash

| Qurilma       | Holat                              |
|---------------|------------------------------------|
| CPU           | To'liq qo'llab-quvvatlanadi        |
| CUDA (NVIDIA) | Avtomatik aniqlanadi               |
| MPS (Apple)   | Avtomatik aniqlanadi (M1/M2/M3)    |
