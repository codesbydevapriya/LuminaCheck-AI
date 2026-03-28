import os
import io
import time
from datetime import datetime

import pandas as pd
import requests
import streamlit as st
from PIL import Image

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="LuminaCheck AI", page_icon="🔍", layout="wide")

# =========================
# API KEYS
# =========================
HIVE_API_KEY = str(st.secrets.get("HIVE_API_KEY", os.getenv("HIVE_API_KEY", ""))).strip()
GEMINI_API_KEY = str(st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))).strip()

# =========================
# SESSION STATE
# =========================
if "history" not in st.session_state:
    st.session_state.history = []

# =========================
# HELPER FUNCTIONS
# =========================
def prepare_image_for_api(image: Image.Image) -> bytes:
    if image.mode != "RGB":
        image = image.convert("RGB")

    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG", quality=95)
    img_bytes.seek(0)
    return img_bytes.getvalue()


def extract_scores_from_hive(data: dict):
    try:
        classes = data["status"][0]["response"]["output"][0]["classes"]
    except Exception:
        return None, None

    ai_score = None
    real_score = None

    for item in classes:
        name = item["class"].lower()
        score = float(item["score"])

        if "ai" in name or "fake" in name or "generated" in name:
            ai_score = max(ai_score or 0, score)
        elif "real" in name or "human" in name:
            real_score = max(real_score or 0, score)

    if ai_score is None and real_score is None:
        return None, None

    if ai_score is None:
        ai_score = 1 - real_score

    if real_score is None:
        real_score = 1 - ai_score

    return ai_score, real_score


def detect_with_hive(image_bytes: bytes):
    if not HIVE_API_KEY:
        return None, None, "Missing HIVE_API_KEY in secrets."

    try:
        response = requests.post(
            "https://api.thehive.ai/api/v2/task/sync",
            headers={
                "Authorization": f"Bearer {HIVE_API_KEY}"   # ✅ FIXED HERE
            },
            files={
                "image": ("image.jpg", image_bytes, "image/jpeg")
            },
            timeout=30
        )

        if response.status_code == 403:
            return None, None, "Invalid API key (403 Forbidden)"
        if response.status_code == 401:
            return None, None, "Unauthorized (401)"
        if response.status_code != 200:
            return None, None, f"Error {response.status_code}: {response.text}"

        data = response.json()
        return (*extract_scores_from_hive(data), None)

    except Exception as e:
        return None, None, str(e)


# =========================
# UI
# =========================
st.title("🔍 LuminaCheck AI")
st.write("Upload an image to detect if it's REAL or AI-generated.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Analyze"):
        with st.spinner("Analyzing..."):
            image_bytes = prepare_image_for_api(image)
            ai, real, err = detect_with_hive(image_bytes)

        if err:
            st.error(err)
        else:
            st.success("Analysis Complete")

            st.write(f" AI Probability: {round(ai*100)}%")
            st.write(f" Real Probability: {round(real*100)}%")

            if ai > 0.5:
                st.error("FAKE / AI GENERATED")
            else:
                st.success("REAL IMAGE")

            st.session_state.history.append({
                "Time": datetime.now(),
                "File": uploaded_file.name,
                "AI %": round(ai*100),
                "Real %": round(real*100)
            })

# =========================
# HISTORY
# =========================
if st.session_state.history:
    st.subheader("History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)
