import os
import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
import google.generativeai as genai
import re
import time

# 🔐 Load API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# ✅ Debug check (temporary)
if GEMINI_API_KEY:
    st.success("✅ Gemini API Key Loaded")
else:
    st.error("❌ Gemini API Key Missing")

st.set_page_config(page_title="LuminaCheck AI", layout="wide")

# ------------------- GEMINI DETECTION -------------------
def detect(image):
    if not GEMINI_API_KEY:
        return None, None

    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = """
        Analyze this image.

        Estimate how likely it is AI-generated.

        Respond ONLY with a number between 0 and 100.
        """

        # Retry logic
        for _ in range(3):
            try:
                response = model.generate_content([prompt, image])
                text = response.text

                match = re.search(r"\d+", text)
                if match:
                    score = float(match.group()) / 100
                    return score, 1 - score

            except Exception as e:
                if "429" in str(e):
                    st.warning("Rate limit hit, retrying...")
                    time.sleep(5)
                else:
                    break

    except Exception as e:
        st.error(f"Error: {e}")

    return None, None


# ------------------- UI -------------------
st.title("🔍 LuminaCheck AI")
st.caption("AI Image Detection using Gemini")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    if st.button("Analyze Image"):

        ai, real = detect(image)

        if ai is None:
            st.warning("Detection failed. Try again.")
            st.stop()

        ai_percent = round(ai * 100)

        st.markdown("### Result")
        st.progress(ai)

        st.write(f"AI Probability: {ai_percent}%")

        # Classification
        if ai > 0.7:
            result = "Likely AI Generated"
            st.error("🚨 Likely AI Generated")
        elif ai < 0.3:
            result = "Likely Real"
            st.success("✅ Likely Real Image")
        else:
            result = "Uncertain"
            st.warning("⚠️ Unable to confidently classify")

        # Save history
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
