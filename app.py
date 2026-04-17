import os
import streamlit as st
from PIL import Image
import replicate
import pandas as pd
from datetime import datetime
import tempfile

# 🔐 API token
os.environ["REPLICATE_API_TOKEN"] = os.environ.get("REPLICATE_API_TOKEN")

st.set_page_config(page_title="LuminaCheck AI", layout="wide")

# ------------------- DETECTION FUNCTION -------------------
def detect(image):
    try:
        # Save image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            image_path = tmp.name

        # Run model
        output = replicate.run(
            "cjwbw/deepfake-image-detector:latest",
            input={"image": open(image_path, "rb")}
        )

        # Handle output
        if isinstance(output, list):
            score = float(output[0])
        else:
            score = float(output)

        return score, 1 - score

    except Exception as e:
        st.error(f"Detection failed: {e}")
        return None, None


# ------------------- UI -------------------
st.title("🔍 LuminaCheck AI")
st.caption("AI Fake Image Detection System (Replicate Powered)")

if "history" not in st.session_state:
    st.session_state.history = []

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=300)

    if st.button("Analyze Image"):

        ai, real = detect(image)

        if ai is None:
            st.stop()

        ai_percent = round(ai * 100)

        st.markdown("### Result")
        st.progress(ai)

        st.write(f"AI Probability: {ai_percent}%")

        # 🔥 Better classification
        if ai > 0.7:
            result = "AI Generated"
            st.error("🚨 AI GENERATED IMAGE")
        elif ai < 0.3:
            result = "Real"
            st.success("✅ REAL IMAGE")
        else:
            result = "Suspicious"
            st.warning("⚠️ SUSPICIOUS IMAGE")

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
