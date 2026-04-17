vimport os
import streamlit as st
from PIL import Image
import requests
import io
import pandas as pd
from datetime import datetime
import google.generativeai as genai

# 🔐 Keys
HIVE_API_KEY = os.environ.get("HIVE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

st.set_page_config(page_title="LuminaCheck AI", layout="wide")

# ------------------- HIVE DETECTION -------------------
def detect_with_hive(image_bytes):
    try:
        response = requests.post(
            "https://api.thehive.ai/api/v2/task/sync",
            headers={"Authorization": f"Token {HIVE_API_KEY}"},
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=15
        )

        if response.status_code != 200:
            return None

        data = response.json()
        classes = data["status"][0]["response"]["output"][0]["classes"]

        for c in classes:
            name = c["class"].lower()
            score = c["score"]

            if "ai" in name or "fake" in name:
                return score

        return None

    except:
        return None


# ------------------- GEMINI -------------------
def detect_with_gemini(image):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = "Is this image AI generated? Give AI percentage only."

        response = model.generate_content([prompt, image])

        text = response.text

        for word in text.split():
            if "%" in word:
                return float(word.replace("%","")) / 100

    except:
        return None


# ------------------- MAIN DETECTOR -------------------
def detect(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")

    # 1. Try Hive
    hive_score = detect_with_hive(img_bytes.getvalue())

    if hive_score is not None:
        return hive_score, 1 - hive_score, "Hive AI"

    # 2. Fallback Gemini
    gemini_score = detect_with_gemini(image)

    if gemini_score is not None:
        return gemini_score, 1 - gemini_score, "Gemini AI"

    # 3. Final fallback
    return 0.5, 0.5, "Fallback"


# ------------------- UI -------------------
st.title("🔍 LuminaCheck AI")
st.caption("Powered by Hive AI + Gemini AI")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=300)

    if st.button("Analyze Image"):

        ai, real, source = detect(image)

        ai_percent = round(ai * 100)

        st.write(f"Detection Source: {source}")
        st.progress(ai)

        st.write(f"AI Probability: {ai_percent}%")

        if ai > 0.7:
            st.error("🚨 AI GENERATED")
        elif ai < 0.3:
            st.success("✅ REAL IMAGE")
        else:
            st.warning("⚠️ UNCERTAIN")
