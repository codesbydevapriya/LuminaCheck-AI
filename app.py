import os
import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
import time
import google.generativeai as genai
import numpy as np

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

st.set_page_config(page_title="LuminaCheck AI", page_icon="🔍", layout="wide")

# ------------------- FORENSIC ANALYSIS -------------------
def analyze_image(image):
    gray = image.convert("L")
    arr = np.array(gray)

    # variance (noise level)
    variance = np.var(arr)

    # edge detection (simple gradient)
    edges = np.mean(np.abs(np.diff(arr)))

    # sharpness
    sharpness = variance

    score = 0.5

    if variance < 400:
        score += 0.2   # smooth → AI
    else:
        score -= 0.1

    if edges < 20:
        score += 0.1
    else:
        score -= 0.05

    return max(0, min(1, score))


# ------------------- GEMINI -------------------
def detect_with_gemini(image):
    if not GEMINI_API_KEY:
        return None

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = """
        Detect if this image is AI-generated or real.
        Respond only:
        AI: <number>
        """

        response = model.generate_content([prompt, image])
        text = response.text

        for line in text.split("\n"):
            if "AI:" in line:
                return float(line.split(":")[1].strip()) / 100

    except:
        return None


# ------------------- FINAL DECISION -------------------
def detect(image):
    forensic_score = analyze_image(image)
    gemini_score = detect_with_gemini(image)

    if gemini_score is not None:
        final_ai = (0.6 * gemini_score) + (0.4 * forensic_score)
    else:
        final_ai = forensic_score

    return final_ai, 1 - final_ai


# ------------------- UI -------------------
st.title("🔍 LuminaCheck AI")
st.write("Advanced AI + Forensic Fake Image Detection")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    if st.button("Analyze Image"):

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)

        ai, real = detect(image)

        ai_percent = round(ai * 100)
        real_percent = round(real * 100)

        st.subheader("Result")

        st.write(f"AI Probability: {ai_percent}%")
        st.write(f"Real Probability: {real_percent}%")

        if ai > 0.6:
            verdict = "FAKE"
            st.error("🚨 FAKE / AI GENERATED")
        elif ai < 0.4:
            verdict = "REAL"
            st.success("✅ REAL IMAGE")
        else:
            verdict = "UNCERTAIN"
            st.warning("⚠️ UNCERTAIN RESULT")

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "File": uploaded_file.name,
            "Result": verdict
        })


# ------------------- HISTORY -------------------
st.subheader("Detection History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "report.csv")
