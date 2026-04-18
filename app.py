import os
import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
import google.generativeai as genai
import re
import time
import numpy as np

# 🔐 API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

st.set_page_config(page_title="LuminaCheck AI", layout="wide")

# ------------------- METADATA -------------------
def analyze_metadata(image):
    try:
        exif = image.getexif()

        if not exif or len(exif) == 0:
            return 0.7

        text = " ".join([str(v).lower() for v in exif.values()])

        if any(x in text for x in ["midjourney", "dalle", "stable diffusion", "ai"]):
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

        if variance < 300:
            return 0.7
        elif variance > 1500:
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


# ------------------- GEMINI -------------------
def detect_with_gemini(image):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = """
        Analyze this image.

        Estimate how likely it is AI-generated.

        Respond ONLY with a number between 0 and 100.
        """

        for _ in range(3):
            try:
                response = model.generate_content([prompt, image])
                text = response.text

                match = re.search(r"\d+", text)
                if match:
                    return float(match.group()) / 100

            except Exception as e:
                if "429" in str(e):
                    time.sleep(5)
                else:
                    break

    except:
        return None

    return None


# ------------------- FINAL DETECTION -------------------
def detect(image, filename):
    gemini_score = detect_with_gemini(image)
    metadata_score = analyze_metadata(image)
    forensics_score = analyze_forensics(image)
    filename_score = analyze_filename(filename)

    if gemini_score is None:
        return None

    final_score = (
        (0.6 * gemini_score) +
        (0.2 * metadata_score) +
        (0.15 * forensics_score) +
        (0.05 * filename_score)
    )

    return final_score


# ------------------- UI -------------------
st.title("🔍 LuminaCheck AI")
st.caption("Hybrid AI Detection System")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(image, use_container_width=True)

    with col2:
        if st.button("Analyze Image"):

            score = detect(image, uploaded_file.name)

            if score is None:
                st.error("Detection failed. Try again.")
                st.stop()

            percent = round(score * 100)

            st.markdown("## Result")

            st.progress(score)

            st.markdown(f"### AI Probability: {percent}%")

            # Classification
            if score > 0.75:
                label = "AI Generated"
                st.error("🚨 AI GENERATED IMAGE")
            elif score < 0.4:
                label = "Likely Real"
                st.success("✅ LIKELY REAL IMAGE")
            else:
                label = "Suspicious"
                st.warning("⚠️ SUSPICIOUS IMAGE")

            # Confidence
            confidence = abs(score - 0.5) * 2

            if confidence > 0.7:
                conf_label = "High"
            elif confidence > 0.3:
                conf_label = "Medium"
            else:
                conf_label = "Low"

            st.markdown(f"**Confidence:** {conf_label}")

            st.markdown("---")
            st.caption("Analysis based on AI model, metadata, image structure, and filename patterns.")


# ------------------- HISTORY -------------------
if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file and score is not None:
    st.session_state.history.append({
        "Time": datetime.now().strftime("%H:%M:%S"),
        "File": uploaded_file.name,
        "Result": label
    })

if st.session_state.history:
    st.markdown("## Detection History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button("Download CSV Report", csv, "report.csv")
