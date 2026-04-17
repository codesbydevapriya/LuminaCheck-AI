import os
import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
import time
import google.generativeai as genai

# 🔐 Load API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

st.set_page_config(page_title="LuminaCheck AI", page_icon="🔍", layout="wide")

# ------------------- DETECTION FUNCTION -------------------
def detect_with_gemini(image):
    if not GEMINI_API_KEY:
        st.error("❌ Gemini API key missing.")
        return None, None

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = """
        Analyze this image and determine if it is AI-generated or real.

        Respond ONLY like this:
        AI: <number>
        REAL: <number>
        """

        response = model.generate_content([prompt, image])
        text = response.text

        ai_score = 0
        real_score = 0

        for line in text.split("\n"):
            if "AI:" in line:
                ai_score = float(line.split(":")[1].strip()) / 100
            if "REAL:" in line:
                real_score = float(line.split(":")[1].strip()) / 100

        return ai_score, real_score

    except Exception as e:
        error_msg = str(e)

        # 🚨 Smart fallback instead of random
        if "429" in error_msg or "quota" in error_msg.lower():
            st.warning("⚠️ Gemini limit reached. Using smart fallback detection.")

            width, height = image.size
            total_pixels = width * height

            if total_pixels > 2000000:
                return 0.3, 0.7   # likely real
            elif total_pixels < 500000:
                return 0.6, 0.4   # likely AI
            else:
                return 0.5, 0.5   # uncertain

        st.error(f"❌ Gemini Error: {error_msg}")
        return None, None


# ------------------- UI -------------------
st.title("🔍 LuminaCheck AI")
st.write("Detect whether an image is REAL or AI-GENERATED")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    if st.button("Analyze Image"):

        progress = st.progress(0)

        for i in range(30):
            time.sleep(0.01)
            progress.progress(i)

        ai, real = detect_with_gemini(image)

        progress.progress(100)

        if ai is not None:
            ai_percent = round(ai * 100)
            real_percent = round(real * 100)

            st.subheader("Result")

            st.write(f"AI Probability: {ai_percent}%")
            st.write(f"Real Probability: {real_percent}%")

            if ai > 0.5:
                verdict = "FAKE"
                st.error("🚨 FAKE / AI GENERATED")
            else:
                verdict = "REAL"
                st.success("✅ REAL IMAGE")

        else:
            verdict = "ERROR"

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
