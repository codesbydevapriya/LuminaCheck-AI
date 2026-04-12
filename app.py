import os
import streamlit as st
from PIL import Image
import requests
import io
import pandas as pd
from datetime import datetime
import time

HIVE_API_KEY = os.environ.get("HIVE_API_KEY")

st.set_page_config(page_title="LuminaCheck AI", layout="wide")

# ---------------- STYLE ----------------
st.markdown("""
<style>
.stApp { background: #f8fafc !important; }
.stButton>button {
    background: #0f172a !important;
    color: white !important;
    border-radius: 8px !important;
    padding: 12px 24px !important;
    font-weight: 600 !important;
    width: 100% !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HIVE FUNCTION ----------------
def detect_with_hive(image_bytes):
    try:
        response = requests.post(
            "https://api.thehive.ai/api/v2/task/sync",
            headers={"Authorization": f"Token {HIVE_API_KEY}"},
            files={"image": ("image.jpg", image_bytes)},  # FIXED
            timeout=20
        )

        if response.status_code == 200:
            data = response.json()

            try:
                classes = data["status"][0]["response"]["output"][0]["classes"]

                ai_score = 0
                real_score = 0

                for c in classes:
                    cls = c["class"].lower()

                    if "ai" in cls or "fake" in cls or "generated" in cls:
                        ai_score = c["score"]

                    if "real" in cls or "human" in cls:
                        real_score = c["score"]

                return ai_score, real_score

            except:
                return None, None

        else:
            return None, None

    except:
        return None, None


# ---------------- SESSION ----------------
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------- UI ----------------
st.title("LuminaCheck AI")

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    key=str(time.time())  # FIXED caching bug
)

if uploaded_file is not None:

    # FIXED image bug
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, use_container_width=True)

    with col2:
        st.markdown(f"""
        **File Name:** {uploaded_file.name}  
        **File Size:** {round(uploaded_file.size/1024,2)} KB
        """)

        if st.button("Analyze Image"):

            progress = st.progress(0)

            for i in range(30):
                time.sleep(0.01)
                progress.progress(i)

            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")

            ai, real = detect_with_hive(img_bytes.getvalue())

            for i in range(30, 100):
                time.sleep(0.005)
                progress.progress(i)

            progress.empty()

            if ai is not None:

                ai_percent = round(ai * 100)
                real_percent = round(real * 100)

                st.markdown("### Detection Result")

                colA, colB = st.columns(2)
                colA.metric("AI Probability", f"{ai_percent}%")
                colB.metric("Real Probability", f"{real_percent}%")

                if ai > 0.5:
                    verdict = "FAKE / AI GENERATED"
                    st.error(verdict)
                else:
                    verdict = "REAL IMAGE"
                    st.success(verdict)

            else:
                verdict = "ERROR"
                st.error("Detection failed. Check API key.")

            # Save history
            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "File": uploaded_file.name,
                "Result": verdict
            })


# ---------------- HISTORY ----------------
st.markdown("---")
st.subheader("Detection History")

if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)
else:
    st.write("No detections yet")
