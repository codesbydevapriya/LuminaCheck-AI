import streamlit as st
from PIL import Image
import requests
import io
import pandas as pd
from datetime import datetime
import time
import google.generativeai as genai

# FIXED KEYS
HIVE_API_KEY = st.secrets["HIVE_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="LuminaCheck AI", page_icon="", layout="wide")

# ---------- CSS (UNCHANGED) ----------
st.markdown("""
<style>
* { font-family: 'Inter', sans-serif; }
.stApp { background: #f8fafc !important; }
.stButton>button { background: #0f172a !important; color: white !important; border-radius: 8px !important; }
.ts-card { background: white; border-radius: 16px; padding: 20px; }
.verdict-real { background: #f0fdf4; border: 2px solid #22c55e; padding: 20px; border-radius: 16px; text-align:center; }
.verdict-fake { background: #fef2f2; border: 2px solid #ef4444; padding: 20px; border-radius: 16px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ---------- HIVE DETECTION ----------
def detect_with_hive(image_bytes):
    try:
        response = requests.post(
            "https://api.thehive.ai/api/v2/task/sync",
            headers={"Authorization": f"Bearer {HIVE_API_KEY}"},  # FIXED
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            timeout=15
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
                    elif "real" in cls or "human" in cls:
                        real_score = c["score"]

                return ai_score, real_score

            except:
                st.error("Parsing error")
                return None, None

        else:
            st.error(f"API Error: {response.status_code}")
            return None, None

    except Exception as e:
        st.error("Connection error")
        return None, None


# ---------- UI ----------
st.title("LuminaCheck AI")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    with col2:
        st.markdown(f"""
        <div class="ts-card">
        <b>File:</b> {uploaded_file.name}<br>
        <b>Size:</b> {uploaded_file.size/1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)

        if st.button("Analyze Image"):

            progress = st.progress(0)

            for i in range(30):
                time.sleep(0.02)
                progress.progress(i)

            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")

            hive_ai, hive_real = detect_with_hive(img_bytes.getvalue())

            for i in range(30, 100):
                time.sleep(0.01)
                progress.progress(i)

            progress.empty()

            if hive_ai is not None:
                ai = round(hive_ai * 100)
                real = round(hive_real * 100)

                st.write("AI:", ai, "%")
                st.write("Real:", real, "%")

                if hive_ai > 0.5:
                    verdict = "FAKE"
                    st.markdown(f"""
                    <div class="verdict-fake">
                    FAKE IMAGE ({ai}%)
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    verdict = "REAL"
                    st.markdown(f"""
                    <div class="verdict-real">
                    REAL IMAGE ({real}%)
                    </div>
                    """, unsafe_allow_html=True)

            else:
                verdict = "ERROR"

            if "history" not in st.session_state:
                st.session_state.history = []

            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "File": uploaded_file.name,
                "Result": verdict
            })


# ---------- HISTORY ----------
if st.checkbox("Show History"):
    if "history" in st.session_state:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)


# ---------- CHAT ----------
st.subheader("Ask AI")

question = st.text_input("Ask something")

if st.button("Ask"):
    if question:
        model = genai.GenerativeModel("gemini-2.5-flash")
        res = model.generate_content(question)
        st.write(res.text)
