"""
app.py — Streamlit Veb-Ilova
Quyma detal rasmini yuklang → Model sinf va ishonch foizini ko'rsatadi.

Ishga tushirish:
    streamlit run app.py
"""

import streamlit as st
import torch
from PIL import Image
from torchvision import transforms

from utils import load_config, get_device
from model import CastCNN


@st.cache_resource
def load_model():
    """Modelni bir marta yuklaydi va keshda saqlaydi."""
    config = load_config()
    device = get_device()
    classes = config["data"]["classes"]

    num_classes = len(classes)
    model = CastCNN(num_classes=num_classes, dropout=config["model"]["dropout"]).to(device)
    model.load_state_dict(torch.load(config["output"]["best_model"], map_location=device))
    model.eval()

    return model, config, device, classes


def predict(image: Image.Image, model, config, device, classes) -> dict:
    """Rasm uchun prediksiya."""
    img_size = config["image"]["size"]
    mean = config["image"]["mean"]
    std = config["image"]["std"]

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_idx = probabilities.max(1)

    predicted_class = classes[predicted_idx.item()]
    confidence_value = confidence.item()
    probs = {cls: round(probabilities[0][i].item(), 4) for i, cls in enumerate(classes)}

    return {
        "predicted_class": predicted_class,
        "confidence": confidence_value,
        "probabilities": probs,
    }


def main():
    st.set_page_config(
        page_title="Quyma Detal Tekshiruvi",
        page_icon="🔍",
        layout="centered",
    )

    st.title("🔍 Quyma Detal Nosozliklarini Aniqlash")
    st.markdown(
        "Quyma detal rasmini yuklang — model uni **nuqsonli (defect)** yoki **normal** ekanligini aniqlaydi."
    )
    st.markdown("---")

    # Modelni yuklash
    try:
        model, config, device, classes = load_model()
    except Exception as e:
        st.error(f"❌ Model yuklanmadi: {e}")
        st.info("Avval `python train.py` buyrug'i bilan modelni o'qiting.")
        return

    # Rasm yuklash
    uploaded_file = st.file_uploader(
        "Rasm tanlang (JPG, JPEG, PNG)",
        type=["jpg", "jpeg", "png"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # Rasmni ko'rsatish
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Yuklangan rasm", use_column_width=True)

        # Prediksiya
        with st.spinner("Tahlil qilinmoqda..."):
            result = predict(image, model, config, device, classes)

        # Natijani ko'rsatish
        with col2:
            predicted_class = result["predicted_class"]
            confidence = result["confidence"]

            if predicted_class == "defect":
                st.error(f"⚠️ **NUQSONLI (DEFECT)**")
            else:
                st.success(f"✅ **NORMAL**")

            st.metric("Ishonch darajasi", f"{confidence * 100:.1f}%")

            st.markdown("**Ehtimolliklar:**")
            for cls, prob in result["probabilities"].items():
                emoji = "🔴" if cls == "defect" else "🟢"
                st.write(f"{emoji} {cls}: {prob * 100:.2f}%")

    # Footer
    st.markdown("---")
    st.caption("CastCNN Model — PyTorch asosida | Quyma detal sifat nazorati")


if __name__ == "__main__":
    main()
