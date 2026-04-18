import os
import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
import google.generativeai as genai
import re
import time
import numpy as np
import requests
from io import BytesIO

# 🔐 API Keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")

st.set_page_config(page_title="LuminaCheck AI", layout="wide")

# ------------------- SESSION STATE -------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_label" not in st.session_state:
    st.session_state.last_label = None

if "last_reason" not in st.session_state:
    st.session_state.last_reason = None


# ------------------- METADATA -------------------
def analyze_metadata(image):
    try:
        exif = image.getexif()

        if not exif or len(exif) == 0:
            return 0.55

        text = " ".join([str(v).lower() for v in exif.values()])

        if any(x in text for x in ["midjourney", "dalle", "stable diffusion"]):
            return 0.9

        if any(x in text for x in ["photoshop", "gimp", "editor"]):
            return 0.5

        if any(x in text for x in ["canon", "nikon", "sony", "iphone", "camera"]):
            return 0.2

        return 0.4

    except:
        return 0.5


# ------------------- FORENSICS -------------------
def analyze_forensics(image):
    try:
        gray = image.convert("L")
        arr = np.array(gray)
        variance = arr.var()

        if variance < 250:
            return 0.7
        elif variance > 1800:
            return 0.3
        else:
            return 0.5

    except:
        return 0.5


# ------------------- FILENAME -------------------
def analyze_filename(filename):
    name = filename.lower()

    if any(x in name for x in ["ai", "dalle", "midjourney", "generated"]):
        return 0.8

    return 0.4


# ------------------- GEMINI SCORE -------------------
def detect_with_gemini(image):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = """Return a number between 0 and 100 indicating likelihood of AI generation."""

        response = model.generate_content([prompt, image])
        match = re.search(r"\d+", str(response.text))

        if match:
            return float(match.group()) / 100

        return 0.5

    except:
        return 0.5


# ------------------- GEMINI REASON -------------------
def get_reason(image):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = """
Give 3 short reasons why this image might be AI-generated or real.
"""

        response = model.generate_content([prompt, image])

        if response and response.text:
            return response.text.strip()

        return "Low confidence explanation"

    except Exception as e:
        return f"Reason error: {e}"


# ------------------- FINAL DETECTION -------------------
def detect(image, filename):
    gemini_score = detect_with_gemini(image)
    metadata_score = analyze_metadata(image)
    forensics_score = analyze_forensics(image)
    filename_score = analyze_filename(filename)

    base_score = (
        (0.7 * gemini_score) +
        (0.15 * metadata_score) +
        (0.1 * forensics_score) +
        (0.05 * filename_score)
    )

    return base_score


# ------------------- UI -------------------
st.title("🔍 LuminaCheck AI")
st.caption("Hybrid AI Detection System")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])


# ✅ HuggingFace Test Button (FIXED)
if st.button("🧪 Test HuggingFace Connection"):
    api_key = HUGGINGFACE_API_KEY

    if not api_key:
        st.error("HF key missing")
    else:
        try:
            headers = {"Authorization": f"Bearer {api_key}"}

            img_url = "https://upload.wikimedia.org/wikipedia/commons/3/3f/Fronalpstock_big.jpg"
            img_bytes = requests.get(img_url, headers={"User-Agent": "Mozilla/5.0"}).content

            API_URL = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"

            res = requests.post(API_URL, headers=headers, data=img_bytes)

            st.write("Status:", res.status_code)
            st.write("Response:", res.text[:300])

        except Exception as e:
            st.error(f"HF test failed: {e}")


# ------------------- IMAGE ANALYSIS -------------------
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(image, use_container_width=True)

    with col2:
        if st.button("Analyze Image"):

            score = detect(image, uploaded_file.name)
            reason = get_reason(image)

            st.session_state.last_result = score
            st.session_state.last_reason = reason

            if score > 0.8:
                label = "AI Generated"
            elif score < 0.45:
                label = "Likely Real"
            else:
                label = "Suspicious"

            st.session_state.last_label = label

    if st.session_state.last_result is not None:
        score = st.session_state.last_result
        label = st.session_state.last_label
        reason = st.session_state.last_reason

        percent = round(score * 100)

        st.markdown("## Result")
        st.progress(score)
        st.markdown(f"### AI Probability: {percent}%")

        if label == "AI Generated":
            st.error("🚨 AI GENERATED IMAGE")
        elif label == "Likely Real":
            st.success("✅ LIKELY REAL IMAGE")
        else:
            st.warning("⚠️ SUSPICIOUS IMAGE")

        confidence = abs(score - 0.5) * 2
        conf_label = "High" if confidence > 0.7 else "Medium" if confidence > 0.3 else "Low"

        st.markdown(f"**Confidence:** {conf_label}")

        if reason:
            st.markdown("### 🔍 Why this result?")
            st.write(reason)

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "File": uploaded_file.name,
            "Result": label
        })


# ------------------- HISTORY -------------------
if st.session_state.history:
    st.markdown("## Detection History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button("Download CSV Report", csv, "report.csv")
