import os
import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from datetime import datetime
import time

st.set_page_config(page_title="LuminaCheck AI", layout="wide")

# ------------------- ANALYSIS -------------------
def analyze(image):
    gray = image.convert("L")
    arr = np.array(gray)

    variance = np.var(arr)
    edges = np.mean(np.abs(np.diff(arr)))

    exif = image.getexif()

    score = 0.5

    # Noise
    if variance < 300:
        score += 0.2
    else:
        score -= 0.1

    # Edges
    if edges < 20:
        score += 0.1
    else:
        score -= 0.05

    # Metadata
    if len(exif) == 0:
        score += 0.2
    else:
        score -= 0.1

    return max(0, min(1, score))


# ------------------- UI -------------------
st.title("🔍 LuminaCheck AI")
st.subheader("Deepfake Detection Analysis")

uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(image, width=300)

    if st.button("Analyze Image"):

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)

        ai = analyze(image)
        real = 1 - ai

        ai_percent = round(ai * 100)

        # Confidence logic
        if ai > 0.75:
            confidence = "High"
            label = "AI Generated"
            color = "red"
        elif ai < 0.25:
            confidence = "High"
            label = "Real"
            color = "green"
        else:
            confidence = "Low"
            label = "Suspicious"
            color = "orange"

        # ------------------- RESULT UI -------------------
        st.markdown("### Result")

        st.progress(ai)

        st.write(f"AI Probability: {ai_percent}%")
        st.write(f"Confidence: {confidence}")
        st.write(f"Classification: {label}")

        if label == "AI Generated":
            st.error("🚨 AI GENERATED IMAGE")
        elif label == "Real":
            st.success("✅ REAL IMAGE")
        else:
            st.warning("⚠️ SUSPICIOUS IMAGE")

        # History
        if "history" not in st.session_state:
            st.session_state.history = []

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "File": uploaded_file.name,
            "Result": label
        })

# ------------------- HISTORY -------------------
if "history" in st.session_state:
    st.subheader("Detection History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)
