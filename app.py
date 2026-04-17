import os
import streamlit as st
from PIL import Image
import requests
import io
import pandas as pd
from datetime import datetime
import google.generativeai as genai
import re

# 🔐 API Keys
HIVE_API_KEY = os.environ.get("HIVE_API_KEY")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

st.set_page_config(page_title="LuminaCheck AI", layout="wide")

# ------------------- HIVE DETECTION -------------------
def detect_with_hive(image_bytes):
    if not HIVE_API_KEY:
        return None

    try:
        response = requests.post(
            "https://api.thehive.ai/api/v2/task/sync",
            headers={"Authorization": f"Token {HIVE_API_KEY}"},
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=10
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


# ------------------- GEMINI DETECTION -------------------
def detect_with_gemini(image):
    if not GEMINI_API_KEY:
        return None

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = """
        Analyze this image and estimate probability that it is AI generated.

        Respond ONLY with a number between 0 and 100.
        Example: 78
        """

        response = model.generate_content([prompt, image])
        text = response.text

        # 🔥 Better extraction
        match = re.search(r"\d+", text)
        if match:
            return float(match.group()) / 100

    except Exception as e:
        st.warning("Gemini failed or quota exceeded")

    return None


# ------------------- MAIN DETECTOR -------------------
def detect(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")

    # 1. Try Hive
    hive_score = detect_with_hive(img_bytes.getvalue())

    if hive_score is not None:
        return hive_score, 1 - hive_score, "Hive AI"

    # 2. Try Gemini
    gemini_score = detect_with_gemini(image)

    if gemini_score is not None:
        return gemini_score, 1 - gemini_score, "Gemini AI"

    # 3. Final fallback (rare)
    return 0.5, 0.5, "Fallback"


# ------------------- UI -------------------
st.title("🔍 LuminaCheck AI")
st.caption("Hybrid Detection: Hive AI + Gemini AI")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=300)

    if st.button("Analyze Image"):

        ai, real, source = detect(image)

        ai_percent = round(ai * 100)

        st.markdown("### Result")
        st.write(f"Detection Source: {source}")
        st.progress(ai)

        st.write(f"AI Probability: {ai_percent}%")

        # ✅ Better classification
        if ai > 0.75:
            result = "Likely AI Generated"
            st.error("🚨 Likely AI Generated Image")
        elif ai < 0.25:
            result = "Likely Real"
            st.success("✅ Likely Real Image")
        else:
            result = "Uncertain"
            st.warning("⚠️ Unable to confidently classify")

        st.info("Result is AI-assisted estimation, not guaranteed.")

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "File": uploaded_file.name,
            "Result": result
        })


# ------------------- HISTORY -------------------
if st.session_state.history:
    st.subheader("Detection History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "report.csv")
