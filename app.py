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

# ------------------- SESSION -------------------
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
            return 0.55

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

        if variance < 200:
            return 0.75
        elif variance > 2000:
            return 0.3
        else:
            return 0.5

    except:
        return 0.5


# ------------------- FILENAME -------------------
def analyze_filename(filename):
    name = filename.lower()

    if any(x in name for x in ["ai", "dalle", "midjourney", "generated"]):
        return 0.75

    return 0.45


# ------------------- GEMINI SCORE -------------------
def detect_with_gemini(image):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = """
You are an expert AI image detector.

Return ONLY a number from 0 to 100.

0 = real photo
100 = AI generated

Be strict. Do not guess randomly.
"""

        response = model.generate_content([prompt, image])
        text = str(response.text)

        match = re.search(r"\d+", text)

        if match:
            return float(match.group()) / 100

        return 0.5

    except:
        return 0.5


# ------------------- STRUCTURED REASON -------------------
def get_reason(image):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = """
Analyze the image and return structured reasoning.

Format EXACTLY like this:

AI Indicators:
- ...
- ...

Real Indicators:
- ...
- ...

Final Verdict:
- ...

Keep it short and clear.
"""

        response = model.generate_content([prompt, image])

        if response and response.text:
            return response.text.strip()

        return "No explanation available"

    except Exception as e:
        return f"Reason error: {e}"


# ------------------- FINAL DETECTION -------------------
def detect(image, filename):
    g = detect_with_gemini(image)
    m = analyze_metadata(image)
    f = analyze_forensics(image)
    n = analyze_filename(filename)

    # 🔥 improved weights
    score = (0.65 * g) + (0.2 * m) + (0.1 * f) + (0.05 * n)

    # 🔥 disagreement control
    if max([g, m, f]) - min([g, m, f]) > 0.6:
        score = 0.5

    return score


# ------------------- UI -------------------
st.title("🔍 LuminaCheck AI")
st.caption("AI Image Detection System")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

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

            if score > 0.75:
                label = "AI Generated"
            elif score < 0.4:
                label = "Likely Real"
            else:
                label = "Suspicious"

            st.session_state.last_label = label

    # ------------------- RESULT -------------------
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

        conf = "High" if confidence > 0.7 else "Medium" if confidence > 0.3 else "Low"
        st.markdown(f"**Confidence:** {conf}")

        st.markdown("---")
        st.markdown("### 🔍 Analysis Breakdown")
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
    st.download_button("Download CSV", csv, "report.csv")
