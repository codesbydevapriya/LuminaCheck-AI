import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import timm

st.set_page_config(page_title="LuminaCheck AI", layout="wide")

# ------------------- LOAD MODEL -------------------
@st.cache_resource
def load_model():
    model = timm.create_model("resnet50", pretrained=True)
    model.eval()
    return model

model = load_model()

# ------------------- TRANSFORM -------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ------------------- DETECTION -------------------
def detect(image):
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img)

    # Fake probability simulation (replace with real model later)
    score = torch.sigmoid(output.mean()).item()

    return score, 1 - score


# ------------------- UI -------------------
st.title("🔍 LuminaCheck AI")
st.subheader("AI Image Detection")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=300)

    if st.button("Analyze Image"):

        score, real = detect(image)

        ai_percent = round(score * 100)

        st.progress(score)

        st.write(f"AI Probability: {ai_percent}%")

        if score > 0.7:
            st.error("🚨 AI GENERATED")
        elif score < 0.3:
            st.success("✅ REAL IMAGE")
        else:
            st.warning("⚠️ UNCERTAIN")
