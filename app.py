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
st.sidebar.write("👋 Welcome Devapriya!")
st.sidebar.write("📌 Upload an image and detect if it is REAL or FAKE using AI.")

if "history" not in st.session_state:
    st.session_state.history = []

def analyze_image(model, image, prompt):
    response = model.generate_content([image, prompt])
    return response.text

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
            model = genai.GenerativeModel("gemini-2.5-flash")

            prompt1 = """You are a strict AI image forensics expert.
Analyze this image for AI generation signs:
- Unnatural skin, hair, or texture
- Perfect symmetry
- Weird hands or fingers
- Impossible lighting or shadows
- Fake background blur
- Overly perfect features
- Text or numbers that look wrong

Be very strict. If ANY doubt exists, say AI-GENERATED.
Only say REAL if 100% sure it is a camera photograph.

Reply ONLY in this format:
Verdict: [REAL or AI-GENERATED or FAKE]
Confidence: [0-100%]
Reason: [2-3 specific clues]"""

            prompt2 = """You are a digital media authentication specialist.
Your task: determine if this image was taken by a real camera or generated/manipulated by AI.

Look carefully for:
1. Skin pores and natural imperfections
2. Natural lighting inconsistencies
3. Background realism
4. Object edges and boundaries
5. Hair strand details
6. Eye reflections

If the image looks TOO perfect or professional, it may be AI-generated.
Real photos have natural imperfections.

Reply ONLY in this format:
Verdict: [REAL or AI-GENERATED or FAKE]
Confidence: [0-100%]
Reason: [2-3 specific clues]"""

            with st.spinner("🔍 First analysis..."):
                result1 = analyze_image(model, image, prompt1)

            with st.spinner("🔎 Second analysis..."):
                result2 = analyze_image(model, image, prompt2)

            st.markdown("---")
            st.subheader("🧠 AI Detection Result:")

            fake1 = "FAKE" in result1.upper() or "AI-GENERATED" in result1.upper()
            fake2 = "FAKE" in result2.upper() or "AI-GENERATED" in result2.upper()

            if fake1 and fake2:
                verdict = "AI-GENERATED / FAKE"
                st.error(f"⚠️ **Both analyses agree: {verdict}**")
                st.error(f"Analysis 1: {result1}")
                st.error(f"Analysis 2: {result2}")
            elif not fake1 and not fake2:
                verdict = "REAL"
                st.success(f"✅ **Both analyses agree: REAL**")
                st.success(f"Analysis 1: {result1}")
                st.success(f"Analysis 2: {result2}")
            else:
                verdict = "UNCERTAIN"
                st.warning(f"⚠️ **Analyses disagree — UNCERTAIN**")
                st.warning(f"Analysis 1: {result1}")
                st.warning(f"Analysis 2: {result2}")
                st.info("💡 This image has mixed signals — could be heavily edited or partially AI-generated.")

            st.session_state.history.append({
                "Time": datetime.now().strftime("%H:%M:%S"),
                "File": uploaded_file.name,
                "Result": verdict,
                "Details": result1[:100]
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
