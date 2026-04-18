import os
import streamlit as st
from PIL import Image
import replicate
import pandas as pd
from datetime import datetime
import tempfile

If GEMINI_API_KEY exists → show "Key Loaded"
Else → show "Key Missing"

# 🔐 Token
os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY")

st.set_page_config(page_title="LuminaCheck AI", layout="wide")

# ------------------- DETECTION -------------------
def detect(image):
    try:
        # Save temp image
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            image.save(tmp.name)
            image_path = tmp.name

        # ✅ Use a working public model (image classifier style)
        output = replicate.run(
            "methexis-inc/img2prompt",
            input={"image": open(image_path, "rb")}
        )

        text = str(output).lower()

        # 🔥 smarter scoring
        score = 0.5

        if any(word in text for word in ["render", "3d", "illustration", "digital"]):
            score += 0.3

        if any(word in text for word in ["photo", "camera", "lens", "realistic"]):
            score -= 0.2

        score = max(0, min(1, score))

        return score, 1 - score

    except Exception as e:
        st.error(f"Detection failed: {e}")
        return None, None


# ------------------- UI -------------------
st.title("🔍 LuminaCheck AI")
st.caption("AI Image Detection (Hybrid AI Analysis)")

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

        # ✅ Safe classification (no embarrassment)
        if ai > 0.75:
            result = "Likely AI Generated"
            st.error("🚨 Likely AI Generated")
        elif ai < 0.25:
            result = "Likely Real"
            st.success("✅ Likely Real Image")
        else:
            result = "Uncertain"
            st.warning("⚠️ Unable to confidently classify")

        st.info("This system uses AI-assisted analysis. Results are probabilistic.")

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
