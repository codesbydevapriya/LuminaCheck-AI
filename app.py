import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
import time
import os
import google.generativeai as genai

# 🔐 Load API key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

st.set_page_config(page_title="LuminaCheck AI", layout="wide")

# ------------------- DETECTION -------------------
def detect(image):
    # -------- Gemini (optional) --------
    ai_score = None

    try:
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            model = genai.GenerativeModel("gemini-2.5-flash")

            prompt = """
            Detect if this image is AI-generated or real.
            Return only:
            AI: <number>
            """

            response = model.generate_content([prompt, image])
            text = response.text

            for line in text.split("\n"):
                if "AI:" in line:
                    ai_score = float(line.split(":")[1].strip()) / 100
                    break
    except:
        ai_score = None

    # -------- Fallback logic --------
    width, height = image.size
    pixels = width * height

    fallback_score = 0.5

    if pixels > 2000000:
        fallback_score = 0.3
    elif pixels < 500000:
        fallback_score = 0.6

    # -------- Combine --------
    if ai_score is not None:
        final_ai = (0.6 * ai_score) + (0.4 * fallback_score)
    else:
        final_ai = fallback_score

    return final_ai, 1 - final_ai


# ------------------- UI -------------------
st.title("🔍 LuminaCheck AI")
st.subheader("AI Image Detection System")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, width=300)

    if st.button("Analyze Image"):

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)

        ai, real = detect(image)

        ai_percent = round(ai * 100)

        st.markdown("### Result")
        st.progress(ai)

        st.write(f"AI Probability: {ai_percent}%")

        # ✅ Better classification (no nonsense)
        if ai > 0.75:
            result = "Likely AI Generated"
            st.error("🚨 Likely AI Generated Image")
        elif ai < 0.25:
            result = "Likely Real"
            st.success("✅ Likely Real Image")
        else:
            result = "Uncertain"
            st.warning("⚠️ Unable to confidently classify")

        st.info("This result is based on AI + heuristic analysis. Not 100% guaranteed.")

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
