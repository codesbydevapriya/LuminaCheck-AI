import streamlit as st
from PIL import Image
import requests
import io
import pandas as pd
from datetime import datetime
import time
import google.generativeai as genai

# KEYS
HIVE_API_KEY = st.secrets["HIVE_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="LuminaCheck AI", layout="wide")

# ---------------- HIVE DETECTION ----------------
def detect_with_hive(image_bytes):
    try:
        response = requests.post(
            "https://api.thehive.ai/api/v2/task/sync",
            headers={
                "Authorization": f"Bearer {HIVE_API_KEY}"
            },
            files={
                "media": ("image.jpg", image_bytes, "image/jpeg")  # FIXED
            },
            timeout=20
        )

        if response.status_code == 200:
            data = response.json()

            try:
                output = data["status"][0]["response"]["output"][0]["classes"]

                ai_score = 0
                real_score = 0

                for c in output:
                    label = c["class"].lower()

                    if any(x in label for x in ["ai", "generated", "fake"]):
                        ai_score = c["score"]

                    if any(x in label for x in ["real", "human"]):
                        real_score = c["score"]

                return ai_score, real_score

            except Exception as e:
                st.error("Parsing error")
                st.write(data)
                return None, None

        else:
            st.error(f"API Error: {response.status_code}")
            st.write(response.text)
            return None, None

    except Exception as e:
        st.error("Connection error")
        st.write(str(e))
        return None, None


# ---------------- UI ----------------
st.title("LuminaCheck AI")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True)

    with col2:
        st.write("File:", uploaded_file.name)
        st.write("Size:", f"{uploaded_file.size/1024:.1f} KB")

        if st.button("Analyze Image"):

            progress = st.progress(0)

            for i in range(30):
                time.sleep(0.02)
                progress.progress(i)

            img_bytes = io.BytesIO()
            image.save(img_bytes, format="JPEG")

            ai, real = detect_with_hive(img_bytes.getvalue())

            for i in range(30, 100):
                time.sleep(0.01)
                progress.progress(i)

            progress.empty()

            if ai is not None:
                ai_percent = round(ai * 100)
                real_percent = round(real * 100)

                st.write("AI Probability:", ai_percent, "%")
                st.write("Real Probability:", real_percent, "%")

                if ai > 0.5:
                    verdict = "FAKE"
                    st.error(f"FAKE IMAGE ({ai_percent}%)")
                else:
                    verdict = "REAL"
                    st.success(f"REAL IMAGE ({real_percent}%)")
            else:
                verdict = "ERROR"

            if "history" not in st.session_state:
                st.session_state.history = []

            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "File": uploaded_file.name,
                "Result": verdict
            })


# ---------------- HISTORY ----------------
if st.checkbox("Show History"):
    if "history" in st.session_state:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)


# ---------------- CHAT ----------------
st.subheader("Ask AI")

question = st.text_input("Ask something")

if st.button("Ask"):
    if question:
        model = genai.GenerativeModel("gemini-2.5-flash")
        res = model.generate_content(
            f"Answer shortly: {question}"
        )
        st.write(res.text)
