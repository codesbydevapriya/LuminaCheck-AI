import os
import streamlit as st
from PIL import Image
import google.generativeai as genai
import pandas as pd
from datetime import datetime
import time

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

st.set_page_config(page_title="LuminaCheck AI", page_icon="🔍", layout="wide")

st.markdown("""
    <style>
    /* Main background */
    .stApp { background-color: #0e1117; }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1a1a2e; }
    
    /* Button */
    .stButton>button {
        background: linear-gradient(135deg, #4CAF50, #2e7d32);
        color: white;
        border-radius: 12px;
        padding: 12px 28px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
        transition: all 0.3s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(76, 175, 80, 0.5);
    }

    /* Title styling */
    h1 { color: #ffffff !important; font-size: 2.5rem !important; }
    h2 { color: #c9a84c !important; }
    h3 { color: #a0c4ff !important; }

    /* Card style for results */
    .result-card {
        background: #1e2a3a;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid #4CAF50;
    }

    /* Upload area */
    [data-testid="stFileUploader"] {
        border: 2px dashed #c9a84c;
        border-radius: 12px;
        padding: 10px;
    }

    /* Scanning animation */
    @keyframes scan {
        0% { top: 0%; }
        100% { top: 100%; }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .scanning-text {
        animation: pulse 1s infinite;
        color: #4CAF50;
        font-size: 18px;
        font-weight: bold;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

LOGO_SVG = """
<svg width="55" height="55" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <circle cx="90" cy="90" r="70" fill="#1a1a2e"/>
  <circle cx="90" cy="90" r="70" fill="none" stroke="#c9a84c" stroke-width="6"/>
  <ellipse cx="90" cy="90" rx="30" ry="30" fill="#c9a84c"/>
  <ellipse cx="90" cy="90" rx="18" ry="18" fill="#1a1a2e"/>
  <ellipse cx="90" cy="90" rx="9" ry="9" fill="#c9a84c" opacity="0.6"/>
  <circle cx="98" cy="82" r="5" fill="white" opacity="0.9"/>
  <line x1="130" y1="130" x2="155" y2="155" stroke="#c9a84c" stroke-width="10" stroke-linecap="round"/>
</svg>"""

page = st.sidebar.radio("Navigation", ["🔍 Detect", "📋 History", "ℹ️ About"])
st.sidebar.markdown("---")
st.sidebar.markdown(LOGO_SVG, unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#c9a84c; font-weight:bold; font-size:16px;'>LuminaCheck AI</p>", unsafe_allow_html=True)
st.sidebar.write("👋 Welcome to LuminaCheck AI!")
st.sidebar.write("📌 Upload an image and detect if it is REAL or FAKE using AI.")
st.sidebar.markdown("---")
st.sidebar.markdown("<p style='color:#555; font-size:12px;'>Powered by Google Gemini AI</p>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

if page == "🔍 Detect":
    col1, col2 = st.columns([1, 8])
    with col1:
        st.markdown(LOGO_SVG, unsafe_allow_html=True)
    with col2:
        st.title("LuminaCheck AI")
        st.markdown("<p style='color:#c9a84c; font-size:18px; font-style:italic;'>Where Light Reveals Truth</p>", unsafe_allow_html=True)
    st.markdown("---")

    uploaded_file = st.file_uploader("📤 Upload an Image to Analyze", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 1])
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="📸 Uploaded Image", use_container_width=True)
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.success("✅ Image uploaded successfully!")
            st.markdown(f"**File:** {uploaded_file.name}")
            st.markdown(f"**Size:** {uploaded_file.size / 1024:.1f} KB")
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🔍 Analyze Image Now"):
                progress_bar = st.progress(0)
                status = st.empty()

                status.markdown("<div class='scanning-text'>🔍 Initializing AI Scanner...</div>", unsafe_allow_html=True)
                for i in range(30):
                    time.sleep(0.03)
                    progress_bar.progress(i)

                status.markdown("<div class='scanning-text'>🤖 Gemini AI Analyzing Image...</div>", unsafe_allow_html=True)

                model = genai.GenerativeModel("gemini-2.5-flash")
                response = model.generate_content([
                    image,
                    """You are a forensic image authentication expert.
Analyze this image and determine if it is REAL or AI-GENERATED/FAKE.
Check for: unnatural skin, perfect symmetry, distorted hands, impossible lighting, overly perfect features, fake background blur.
Be strict but fair. Only say REAL if 100% sure it is a genuine photograph.
Reply ONLY in this exact format:
Verdict: [REAL or AI-GENERATED or FAKE]
Confidence: [0-100%]
Reason: [2-3 specific visual clues]"""
                ])

                for i in range(30, 100):
                    time.sleep(0.02)
                    progress_bar.progress(i)

                status.markdown("<div class='scanning-text'>✅ Analysis Complete!</div>", unsafe_allow_html=True)
                progress_bar.progress(100)
                time.sleep(0.5)
                progress_bar.empty()
                status.empty()

                result = response.text
                st.markdown("---")
                st.subheader("🧠 AI Detection Result")

                if "FAKE" in result.upper() or "AI-GENERATED" in result.upper():
                    st.error(f"⚠️ **FAKE / AI-GENERATED DETECTED**\n\n{result}")
                    verdict = "FAKE/AI-GENERATED"
                else:
                    st.success(f"✅ **REAL IMAGE VERIFIED**\n\n{result}")
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
        st.info("📭 No detections yet. Go to Detect page and upload an image!")

elif page == "ℹ️ About":
    col1, col2 = st.columns([1, 8])
    with col1:
        st.markdown(LOGO_SVG, unsafe_allow_html=True)
    with col2:
        st.title("About LuminaCheck AI")
    st.markdown("---")
    st.markdown("""
    ## 🔍 What is LuminaCheck AI?
    LuminaCheck AI is a **Final Year BCA Project** that uses **Google Gemini AI** to detect whether an image is **REAL**, **FAKE**, or **AI-GENERATED**.

    ## 🛠️ Technologies Used
    - **Python** — Core programming language
    - **Streamlit** — Web application framework
    - **Google Gemini AI** — Image analysis engine
    - **Pillow** — Image processing
    - **Pandas** — Data management

    ## 🌐 Live Links
    - **App:** https://luminacheck-ai.streamlit.app
    - **GitHub:** https://github.com/codesbydevapriya/LuminaCheck-AI

    ## 👩‍💻 Developed By
    **Devapriya** — BCA Final Year Student | March 2026
    """)
