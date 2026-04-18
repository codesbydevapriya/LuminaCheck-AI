import os
import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
import google.generativeai as genai
import re
import time
import numpy as np

# 🔐 API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

st.set_page_config(page_title="LuminaCheck AI", layout="wide")

# ------------------- SESSION STATE -------------------
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "last_label" not in st.session_state:
    st.session_state.last_label = None

# ------------------- METADATA -------------------
def analyze_metadata(image):
    try:
        exif = image.getexif()

        if not exif or len(exif) == 0:
            return 0.7

        text = " ".join([str(v).lower() for v in exif.values()])

        if any(x in text for x in ["midjourney", "dalle", "stable diffusion", "ai"]):
            return 0.9

        if any(x in text for x in ["photoshop", "gimp", "editor"]):
            return 0.5

        if any(x in text for x in ["canon", "nikon", "sony", "iphone", "camera"]):
            return 0.2

        return 0.4

    except:
        return 0.5


# ------------------- FORENSICS -------------------
def analyze_forensics(image):
    try:
        gray = image.convert("L")
        arr = np.array(gray)
        variance = arr.var()

        if variance < 300:
            return 0.7
        elif variance > 1500:
            return 0.3
        else:
            return 0.5

    except:
        return 0.5


# ------------------- FILENAME -------------------
def analyze_filename(filename):
    name = filename.lower()

    if any(x in name for x in ["ai", "dalle", "midjourney", "generated"]):
        return 0.8

    return 0.4


# ------------------- GEMINI -------------------
def detect_with_gemini(image):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = """
        You are an expert forensic image analyst specializing in detecting AI-generated images. 
You will analyze the uploaded image with extreme precision across every possible dimension.

Examine ALL of the following aspects thoroughly before reaching a conclusion:

TEXTURE & SURFACE ANALYSIS:
- Skin pores, hair strands, fabric weave — are they consistent and physically plausible?
- Do surfaces have realistic imperfections, wear, and variation?
- Are textures too smooth, repetitive, or unnaturally perfect?

GEOMETRIC & STRUCTURAL INTEGRITY:
- Count fingers, teeth, ears, eyes — are quantities and symmetry correct?
- Do hands, feet, and limbs follow correct anatomical proportions?
- Are straight lines truly straight (walls, floors, furniture edges)?
- Do glasses, jewelry, and accessories have consistent shape across the full object?

LIGHTING & SHADOW CONSISTENCY:
- Is there a single coherent light source, or do shadows contradict each other?
- Do reflections in eyes, glasses, and shiny surfaces match the environment?
- Is subsurface scattering on skin physically realistic?

BACKGROUND & ENVIRONMENT:
- Is background text legible and correctly spelled?
- Do background objects maintain consistent perspective and scale?
- Are there repeated or mirrored elements in the background?
- Do edges between subject and background show blending artifacts or halos?

FINE DETAIL EXAMINATION:
- Zoom into ears — are the folds anatomically correct?
- Examine hairline edges — are individual strands distinguishable or merged into blobs?
- Check jewelry, buttons, zippers — are they symmetrical and physically coherent?
- Look at eyes closely — are catchlights, pupils, irises, and veins realistic?

AI ARTIFACT DETECTION:
- Are there any areas of unusual smoothness surrounded by detail?
- Do facial features drift or look slightly "melted"?
- Is there inconsistent resolution — some areas sharp, others inexplicably soft?
- Are patterns (tiles, fabric, wallpaper) coherent or do they break down on inspection?

METADATA INDICATORS (visual):
- Does the overall aesthetic feel "too perfect" with no motion blur, lens distortion, or chromatic aberration?
- Is the depth of field physically consistent with the apparent focal length?
- Are there any watermarks, signatures, or generation artifacts visible?

CROSS-CHECK:
After examining every aspect above, weigh the evidence. Consider that:
- A single major anomaly (e.g., 6 fingers, impossible shadow) is strong evidence of AI generation.
- Multiple minor anomalies compound into high confidence.
- Photorealistic images can still be AI-generated — absence of obvious flaws does not mean real.

OUTPUT INSTRUCTION:
Respond with ONLY a single integer between 0 and 100.
0 = Certainly real photograph.
100 = Certainly AI-generated.
No explanation. No text. No punctuation. Just the number.
        """

        for _ in range(3):
            try:
                response = model.generate_content([prompt, image])
                text = response.text

                match = re.search(r"\d+", text)
                if match:
                    return float(match.group()) / 100

            except Exception as e:
                if "429" in str(e):
                    time.sleep(5)
                else:
                    break

    except:
        return None

    return None


# ------------------- FINAL DETECTION -------------------
def detect(image, filename):
    gemini_score = detect_with_gemini(image)
    metadata_score = analyze_metadata(image)
    forensics_score = analyze_forensics(image)
    filename_score = analyze_filename(filename)

    if gemini_score is None:
        return None

    final_score = (
        (0.6 * gemini_score) +
        (0.2 * metadata_score) +
        (0.15 * forensics_score) +
        (0.05 * filename_score)
    )

    return final_score


# ------------------- UI -------------------
st.title("🔍 LuminaCheck AI")
st.caption("Hybrid AI Detection System")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.image(image, use_container_width=True)

    with col2:
        if st.button("Analyze Image"):

            score = detect(image, uploaded_file.name)

            if score is None:
                st.error("Detection failed. Try again.")
                st.stop()

            st.session_state.last_result = score

            # Classification
            if score > 0.75:
                label = "AI Generated"
            elif score < 0.4:
                label = "Likely Real"
            else:
                label = "Suspicious"

            st.session_state.last_label = label

    # ------------------- RESULT DISPLAY -------------------
    if st.session_state.last_result is not None:
        score = st.session_state.last_result
        label = st.session_state.last_label

        percent = round(score * 100)

        st.markdown("## Result")
        st.progress(score)
        st.markdown(f"### AI Probability: {percent}%")

        if label == "AI Generated":
            st.error("🚨 AI GENERATED IMAGE")
        elif label == "Likely Real":
            st.success("✅ LIKELY REAL IMAGE")
        else:
            st.warning("⚠️ SUSPICIOUS IMAGE")

        confidence = abs(score - 0.5) * 2

        if confidence > 0.7:
            conf_label = "High"
        elif confidence > 0.3:
            conf_label = "Medium"
        else:
            conf_label = "Low"

        st.markdown(f"**Confidence:** {conf_label}")

        st.markdown("---")
        st.caption("Analysis based on AI model, metadata, image structure, and filename patterns.")

        # Save history safely
        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "File": uploaded_file.name,
            "Result": label
        })


# ------------------- HISTORY -------------------
if st.session_state.history:
    st.markdown("## Detection History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)

    csv = df.to_csv(index=False)
    st.download_button("Download CSV Report", csv, "report.csv")
