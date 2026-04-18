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
import base64
from io import BytesIO

# 🔐 API Keys
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

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

        prompt = """YOUR SAME FORENSIC PROMPT"""

        for _ in range(3):
            try:
                response = model.generate_content([prompt, image])
                text = str(response.text).strip()

                match = re.search(r"\d+", text)
                if match:
                    score = float(match.group()) / 100
                    return max(0.0, min(1.0, score))

            except Exception as e:
                if "429" in str(e):
                    time.sleep(5)

        return 0.5

    except:
        return 0.5


# ------------------- OPENROUTER REASON -------------------
def get_reason(image):
    try:
        if not OPENROUTER_API_KEY:
            return "OpenRouter API key missing"

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        prompt = """
Analyze this image and explain why it might be AI-generated or real.

Give 3 to 5 short bullet points.
"""

        # 🔥 TRY IMAGE MODEL FIRST (paid but cheap)
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "openai/gpt-4o-mini",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ]
            }
        )

        data = response.json()

        if "choices" in data:
            return data["choices"][0]["message"]["content"]

        # 🔥 FALLBACK → FREE MODEL (no image, still useful)
        fallback = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-chat",
                "messages": [
                    {
                        "role": "user",
                        "content": "Explain common signs of AI-generated images in simple bullet points."
                    }
                ]
            }
        )

        data2 = fallback.json()

        if "choices" in data2:
            return "General analysis:\n" + data2["choices"][0]["message"]["content"]

        return f"API issue: {data}"

    except Exception as e:
        return f"Reason error: {e}"


# ------------------- FINAL DETECTION -------------------
def detect(image, filename):
    gemini_score = detect_with_gemini(image)
    metadata_score = analyze_metadata(image)
    forensics_score = analyze_forensics(image)
    filename_score = analyze_filename(filename)

    gemini_score = max(0.1, min(0.9, gemini_score))

    base_score = (
        (0.7 * gemini_score) +
        (0.15 * metadata_score) +
        (0.1 * forensics_score) +
        (0.05 * filename_score)
    )

    scores = [gemini_score, metadata_score, forensics_score]
    disagreement = max(scores) - min(scores)

    if disagreement > 0.5:
        base_score = (base_score * 0.7) + (0.5 * 0.3)

    if disagreement > 0.7:
        base_score = 0.5

    if 0.4 < base_score < 0.6:
        base_score = 0.5

    return base_score


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

        if confidence > 0.7:
            conf_label = "High"
        elif confidence > 0.3:
            conf_label = "Medium"
        else:
            conf_label = "Low"

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
