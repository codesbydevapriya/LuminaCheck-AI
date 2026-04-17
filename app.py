import os
import streamlit as st
from PIL import Image
import requests
import io
import pandas as pd
from datetime import datetime
import time

# 🔐 Load from Streamlit Secrets
HIVE_ACCESS_KEY = os.environ.get("HIVE_ACCESS_KEY")
HIVE_SECRET_KEY = os.environ.get("HIVE_SECRET_KEY")

st.set_page_config(page_title="LuminaCheck AI", page_icon="🔍", layout="wide")

# ------------------- HIVE V3 FUNCTION -------------------
def detect_with_hive(image_bytes):
    if not HIVE_ACCESS_KEY or not HIVE_SECRET_KEY:
        st.error("❌ Hive API keys missing. Check Streamlit Secrets.")
        return None, None

    url = "https://api.thehive.ai/api/v3/tasks/sync"

    try:
        response = requests.post(
            url,
            headers={
                "x-api-key": HIVE_ACCESS_KEY,
                "x-api-secret": HIVE_SECRET_KEY
            },
            files={"media": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=20
        )

        if response.status_code != 200:
            st.error(f"❌ Hive API Error {response.status_code}")
            st.write(response.text)
            return None, None

        data = response.json()

        # Safe parsing
        output = data.get("output", [])
        if not output:
            st.error("❌ No output from Hive")
            return None, None

        classes = output[0].get("classes", [])

        ai_score = 0
        real_score = 0

        for c in classes:
            name = c.get("class", "").lower()
            score = c.get("score", 0)

            if "ai" in name or "fake" in name or "generated" in name:
                ai_score = score
            elif "real" in name or "human" in name:
                real_score = score

        return ai_score, real_score

    except Exception as e:
        st.error(f"❌ Connection Error: {str(e)}")
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

        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")

        ai, real = detect_with_hive(img_bytes.getvalue())

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
