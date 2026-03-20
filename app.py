import os
import streamlit as st
from PIL import Image
import google.generativeai as genai
import pandas as pd
from datetime import datetime

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

st.set_page_config(page_title="LuminaCheck AI", page_icon="🔍", layout="wide")

st.markdown("""
    <style>
    .stButton>button { background-color: #4CAF50; color: white; border-radius: 10px; padding: 10px 24px; font-size: 16px; }
    </style>
""", unsafe_allow_html=True)

page = st.sidebar.radio("Navigation", ["🔍 Detect", "📋 History", "ℹ️ About"])
st.sidebar.markdown("---")
st.sidebar.write(" 👋 Welcome to LuminaCheck AI!")
st.sidebar.write("📌 Upload an image and detect if it is REAL or FAKE using AI.")

if "history" not in st.session_state:
    st.session_state.history = []

if page == "🔍 Detect":
    st.title("🔍 LuminaCheck AI")
    st.subheader("Where Light Reveals Truth")
    st.markdown("---")
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=400)
        st.success("✅ Image uploaded successfully!")

        if st.button("🚀 Detect Now!"):
            with st.spinner("🤖 AI analyzing your image..."):
                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content([
                    image,
                    """You are a forensic image authentication expert.
Analyze this image and determine if it is REAL or AI-GENERATED/FAKE.

Check for these signs of AI generation:
1. Unnatural or overly smooth skin texture
2. Perfect symmetry in faces
3. Distorted or extra fingers/hands
4. Impossible or inconsistent lighting
5. Overly perfect hair or clothing
6. Blurred or inconsistent backgrounds
7. Text or numbers that look slightly wrong

Be strict but fair. Give your honest assessment.

Reply ONLY in this exact format:
Verdict: [REAL or AI-GENERATED or FAKE]
Confidence: [0-100%]
Reason: [2-3 specific visual clues]"""
                ])
                result = response.text
                st.markdown("---")
                st.subheader("🧠 AI Detection Result:")
                if "FAKE" in result.upper() or "AI-GENERATED" in result.upper():
                    st.error(f"⚠️ {result}")
                    verdict = "FAKE/AI-GENERATED"
                else:
                    st.success(f"✅ {result}")
                    verdict = "REAL"

                st.session_state.history.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "File": uploaded_file.name,
                    "Result": verdict,
                    "Details": result[:120]
                })

elif page == "📋 History":
    st.title("📋 Detection History")
    st.markdown("---")
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False)
        st.download_button("📥 Download Report (CSV)", csv, "lumina_report.csv", "text/csv")
    else:
        st.info("No detections yet. Go to Detect page and upload an image!")

elif page == "ℹ️ About":
    st.title("ℹ️ About LuminaCheck AI")
    st.markdown("---")
    st.markdown("""
    ## 🔍 What is LuminaCheck AI?
    LuminaCheck AI is a **Final Year BCA Project** that uses **Google Gemini AI** to detect whether an image is REAL, FAKE, or AI-GENERATED.

    ## 🛠️ Technologies Used
    - **Python** — Core programming language
    - **Streamlit** — Web application framework
    - **Google Gemini AI** — Image analysis engine
    - **Pillow** — Image processing
    - **Pandas** — Data management

    ## 👩‍💻 Developed By
    **Devapriya** — BCA Final Year Student
    """)
