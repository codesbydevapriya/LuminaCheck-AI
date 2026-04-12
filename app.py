import os
import streamlit as st
from PIL import Image
import requests
import io
import pandas as pd
from datetime import datetime
import time

HIVE_API_KEY = os.environ.get("HIVE_API_KEY")

st.set_page_config(page_title="LuminaCheck AI", page_icon="🔍", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
* { font-family: sans-serif; }
.stApp { background: #f8fafc !important; }
.ts-card { background: #ffffff; border-radius: 16px; padding: 20px; border: 1px solid #e2e8f0; }
.verdict-real { background:#f0fdf4; border:2px solid #22c55e; padding:20px; border-radius:16px; text-align:center;}
.verdict-fake { background:#fef2f2; border:2px solid #ef4444; padding:20px; border-radius:16px; text-align:center;}
</style>
""", unsafe_allow_html=True)

# ---------- HIVE FUNCTION ----------
def detect_with_hive(image_bytes):
    try:
        response = requests.post(
            "https://api.thehive.ai/api/v2/task/sync",
            headers={"Authorization": f"Token {HIVE_API_KEY}"},
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=15
        )

        if response.status_code != 200:
            return None, None

        data = response.json()

        classes = data.get("status", [{}])[0].get("response", {}) \
            .get("output", [{}])[0].get("classes", [])

        ai_score, real_score = 0, 0

        for c in classes:
            name = c.get("class", "").lower()
            score = c.get("score", 0)

            if "ai" in name or "fake" in name or "generated" in name:
                ai_score = max(ai_score, score)
            elif "real" in name or "human" in name:
                real_score = max(real_score, score)

        return ai_score, real_score

    except:
        return None, None


# ---------- SESSION ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- SIDEBAR ----------
page = st.sidebar.radio("", ["Detect", "History", "About"])

# ---------- DETECT PAGE ----------
if page == "Detect":

    st.title("LuminaCheck AI")

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"],
        key="uploader"
    )

    # IMPORTANT FIX: only show image if uploaded
    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, use_container_width=True)

        with col2:
            st.markdown(f"""
            <div class="ts-card">
                <b>File:</b> {uploaded_file.name}<br>
                <b>Size:</b> {round(uploaded_file.size/1024,2)} KB
            </div>
            """, unsafe_allow_html=True)

            if st.button("Analyze Image"):

                img_bytes = io.BytesIO()
                image.save(img_bytes, format="JPEG")

                with st.spinner("Analyzing..."):
                    ai, real = detect_with_hive(img_bytes.getvalue())

                if ai is None:
                    st.error("Detection failed")
                    return

                ai_p = round(ai * 100)
                real_p = round(real * 100)

                if ai > 0.5:
                    verdict = "FAKE"
                    st.markdown(f"""
                    <div class="verdict-fake">
                        FAKE IMAGE<br><br>
                        AI Probability: {ai_p}%
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    verdict = "REAL"
                    st.markdown(f"""
                    <div class="verdict-real">
                        REAL IMAGE<br><br>
                        Real Probability: {real_p}%
                    </div>
                    """, unsafe_allow_html=True)

                # Save history
                st.session_state.history.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "File": uploaded_file.name,
                    "Result": verdict,
                    "AI%": ai_p,
                    "Real%": real_p
                })

# ---------- HISTORY ----------
elif page == "History":

    st.title("History")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No history yet")

# ---------- ABOUT ----------
elif page == "About":

    st.title("About")

    st.write("""
    LuminaCheck AI detects whether an image is real or AI-generated
    using Hive AI detection API.
    """)
