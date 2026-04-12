import streamlit as st
from PIL import Image
import requests
import io
import pandas as pd
from datetime import datetime
import time
import google.generativeai as genai

HIVE_API_KEY = st.secrets["HIVE_API_KEY"]
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]

genai.configure(api_key=GEMINI_API_KEY)

st.set_page_config(page_title="LuminaCheck AI", layout="wide")

def detect_with_hive(image_bytes):
    try:
        if not HIVE_API_KEY:
            st.error("API key not found")
            return None, None

        response = requests.post(
            "https://api.thehive.ai/api/v2/task/sync",
            headers={
                "Authorization": f"Bearer {HIVE_API_KEY}"
            },
            files={
                "image": ("image.jpg", image_bytes, "image/jpeg")
            },
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

                    if "ai" in cls or "generated" in cls or "fake" in cls:
                        ai_score = c["score"]

                    elif "real" in cls or "human" in cls:
                        real_score = c["score"]

                return ai_score, real_score

            except:
                st.error("Error parsing API response")
                return None, None

        elif response.status_code == 403:
            st.error("API authentication failed. Check API key.")
            return None, None

        else:
            st.error(f"API Error: {response.status_code}")
            return None, None

    except Exception:
        st.error("Connection error. Try again later.")
        return None, None


if "history" not in st.session_state:
    st.session_state.history = []

st.title("LuminaCheck AI")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, use_container_width=True)

    if st.button("Analyze Image"):

        img_bytes = io.BytesIO()
        image.save(img_bytes, format="JPEG")

        hive_ai, hive_real = detect_with_hive(img_bytes.getvalue())

        if hive_ai is not None:
            hive_percent = round(hive_ai * 100)
            real_percent = round(hive_real * 100)

            st.write("AI Probability:", hive_percent, "%")
            st.write("Real Probability:", real_percent, "%")

            if hive_ai > 0.5:
                verdict = "FAKE"
                st.error("Fake or AI Generated Image")
            else:
                verdict = "REAL"
                st.success("Real Image")

        else:
            verdict = "ERROR"

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "File": uploaded_file.name,
            "Result": verdict
        })


if st.checkbox("Show History"):
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df)


st.subheader("Ask AI")

user_input = st.text_input("Ask about image detection")

if st.button("Ask"):
    if user_input:
        model = genai.GenerativeModel("gemini-2.5-flash")

        response = model.generate_content(
            f"Explain in simple words: {user_input}"
        )

        st.write(response.text)
```
