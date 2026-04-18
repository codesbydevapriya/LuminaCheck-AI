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

# Debug check
if GEMINI_API_KEY:
    st.success("✅ Gemini API Key Loaded")
else:
    st.error("❌ Gemini API Key Missing")

st.set_page_config(page_title="LuminaCheck AI", layout="wide")

# ------------------- METADATA ANALYSIS -------------------
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


# ------------------- GEMINI DETECTION -------------------
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
                    st.warning("Rate limit hit, retrying...")
                    time.sleep(5)
                else:
                    break

    except Exception as e:
        st.error(f"Gemini Error: {e}")

    return None


# ------------------- FINAL DETECTION -------------------
def detect(image):
    if not GEMINI_API_KEY:
        return None

    gemini_score = detect_with_gemini(image)
    metadata_score = analyze_metadata(image)

    if gemini_score is None:
        return None

    # 🔥 Weighted combination
    final_score = (0.6 * gemini_score) + (0.2 * metadata_score)

    return final_score, gemini_score, metadata_score


# ------------------- UI -------------------
st.title("🔍 LuminaCheck AI")
st.caption("Hybrid AI Detection (Gemini + Metadata)")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    if st.button("Analyze Image"):

        result = detect(image)

        if result is None:
            st.warning("Detection failed. Try again.")
            st.stop()

        final_score, gemini_score, metadata_score = result

        percent = round(final_score * 100)

        st.markdown("### Result")
        st.progress(final_score)

        st.write(f"AI Probability: {percent}%")

        # Debug (you can remove later)
        st.write(f"Gemini Score: {round(gemini_score*100)}%")
        st.write(f"Metadata Score: {round(metadata_score*100)}%")

        # Classification
        if final_score > 0.75:
            label = "AI Generated"
            st.error("🚨 AI GENERATED IMAGE")
        elif final_score < 0.4:
            label = "Likely Real"
            st.success("✅ LIKELY REAL IMAGE")
        else:
            label = "Suspicious"
            st.warning("⚠️ SUSPICIOUS IMAGE")

        # Confidence
        confidence = abs(final_score - 0.5) * 2
        if confidence > 0.7:
            conf_label = "High"
        elif confidence > 0.3:
            conf_label = "Medium"
        else:
            conf_label = "Low"

        st.write(f"Confidence: {conf_label}")

        # Save history
        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "File": uploaded_file.name,
            "Result": label
        })


# ------------------- HISTORY -------------------
if st.session_state.history:
    st.subheader("Detection History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)

    csv = df.to_csv(index=False)
    st.download_button("Download CSV", csv, "report.csv")
