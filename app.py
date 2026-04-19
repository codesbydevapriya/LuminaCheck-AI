import os
import base64
import json
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, ImageFilter
import pandas as pd
from datetime import datetime
import google.generativeai as genai
import re
import time
import numpy as np
import io

# ─── CONFIGURATION ────────────────────────────────────────────────────────────
# Ensure your environment variable is set or replace with your string for testing
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

st.set_page_config(
    page_title="LuminaCheck AI | Professional Forensic Suite",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="collapsed",
)

# Professional Dark Theme UI
st.markdown("""
<style>
#MainMenu, header, footer, .stDeployButton { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: #0c0c10; color: #e8e6f0; }
[data-testid="stFileUploader"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE MANAGEMENT ──────────────────────────────────────────────────
for key, default in {
    "history": [],
    "last_result": None,
    "ui_phase": "upload",
    "current_filename": None,
    "current_file_bytes": None,
    "ready_to_analyze": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── CORE FORENSIC ENGINE ──────────────────────────────────────────────────────

def analyze_metadata(image: Image.Image) -> tuple:
    """Analyzes EXIF data for camera signatures vs AI software traces."""
    try:
        exif = image.getexif()
        if not exif:
            return 0.40, "No EXIF found (Typical for web/social media images)"
        
        meta_str = " ".join([str(v).lower() for v in exif.values()])
        
        # Check for AI generation software tags
        ai_signatures = ["midjourney", "dalle", "stable diffusion", "firefly", "flux", "gan"]
        if any(sig in meta_str for sig in ai_signatures):
            return 0.98, "Confirmed AI signature in Metadata"
            
        # Check for legitimate camera hardware tags
        camera_brands = ["canon", "nikon", "sony", "iphone", "samsung", "pixel", "fujifilm"]
        if any(brand in meta_str for brand in camera_brands):
            return 0.05, "Authentic Hardware Metadata detected"
            
        return 0.30, "Metadata present, hardware unspecified"
    except Exception:
        return 0.50, "Metadata parsing error"

def analyze_pixel_forensics(image: Image.Image) -> tuple:
    """Calculates statistical anomalies in pixel distribution and edge energy."""
    try:
        img_res = image.resize((256, 256)).convert("RGB")
        gray = img_res.convert("L")
        arr = np.array(gray, dtype=np.float32)
        
        # Variance check (AI images often have 'too perfect' noise distributions)
        variance = arr.var()
        
        # Edge Energy (Laplacian) - Real photos have sharp high-frequency noise
        laplacian = gray.filter(ImageFilter.FIND_EDGES)
        edge_energy = np.array(laplacian, dtype=np.float32).mean()
        
        # RGB Channel Correlation
        r, g, b = img_res.split()
        r_arr, g_arr, b_arr = np.array(r).flatten(), np.array(g).flatten(), np.array(b).flatten()
        
        # Safety: Add tiny noise to avoid division by zero in flat images
        if np.std(r_arr) < 0.1 or np.std(g_arr) < 0.1:
            avg_corr = 0.5
        else:
            rg_corr = np.corrcoef(r_arr, g_arr)[0, 1]
            rb_corr = np.corrcoef(r_arr, b_arr)[0, 1]
            avg_corr = (abs(rg_corr) + abs(rb_corr)) / 2
            if np.isnan(avg_corr): avg_corr = 0.5

        # Scoring Logic
        f_score = 0.45 # Neutral start
        if edge_energy < 6: f_score += 0.25  # Smoothness suggests AI
        if avg_corr > 0.98: f_score += 0.15   # Perfect correlation suggests synthesis
        
        return round(min(0.95, f_score), 2), f"Edge: {edge_energy:.1f} | Corr: {avg_corr:.2f}"
    except Exception as e:
        return 0.50, f"Forensic calculation failed: {str(e)}"

def detect_with_gemini(image: Image.Image) -> tuple:
    """Uses LLM Vision to detect semantic artifacts (fingers, lighting, skin)."""
    if not GEMINI_API_KEY:
        return 0.50, "Gemini API Key missing"
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = """Act as a professional forensic analyst. Analyze this image for:
        1. Skin/Texture smoothness (AI tends to be 'plastic')
        2. Anatomical logic (fingers, eyes, symmetry)
        3. Shadow/Lighting consistency
        
        Provide the result in this format:
        SCORE: [0-100]
        REASON: [Max 2 sentences]"""

        response = model.generate_content([prompt, image])
        text = response.text
        
        score_match = re.search(r"SCORE:\s*(\d+)", text)
        reason_match = re.search(r"REASON:\s*(.+)", text)
        
        score = float(score_match.group(1)) / 100 if score_match else 0.5
        reason = reason_match.group(1).strip() if reason_match else "Semantic check complete."
        return score, reason
    except Exception as e:
        return 0.50, f"Vision Analysis Error: {str(e)}"

def run_fusion_analysis(image: Image.Image, filename: str) -> dict:
    """Fuses all signals using a weighted scoring model."""
    g_score, g_reason = detect_with_gemini(image)
    m_score, m_note = analyze_metadata(image)
    f_score, f_note = analyze_pixel_forensics(image)
    
    # Professional Weighting: Vision (55%), Forensics (25%), Metadata (20%)
    final_score = (g_score * 0.55) + (f_score * 0.25) + (m_score * 0.20)
    
    return {
        "score": round(final_score, 2),
        "label": "AI GENERATED" if final_score > 0.7 else ("REAL PHOTO" if final_score < 0.3 else "SUSPICIOUS"),
        "reason": g_reason,
        "technical_log": f"Signals -> Vision: {g_score} | Forensics: {f_note} | Meta: {m_note}"
    }

# ─── INTERFACE LOGIC ───────────────────────────────────────────────────────────

# Handle File Upload
uploaded_file = st.file_uploader("Upload", type=["jpg", "jpeg", "png"], key="forensic_upload")

if uploaded_file and st.session_state.ui_phase == "upload":
    st.session_state.current_file_bytes = uploaded_file.read()
    st.session_state.current_filename = uploaded_file.name
    st.session_state.ui_phase = "analyzing"
    st.rerun()

if st.session_state.ui_phase == "analyzing":
    with st.spinner("Executing Forensic Multi-Pass Scan..."):
        img = Image.open(io.BytesIO(st.session_state.current_file_bytes)).convert("RGB")
        result = run_fusion_analysis(img, st.session_state.current_filename)
        st.session_state.last_result = result
        st.session_state.ui_phase = "results"
        st.rerun()

if st.session_state.ui_phase == "results":
    res = st.session_state.last_result
    st.title("Forensic Analysis Report")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(st.session_state.current_file_bytes, use_column_width=True)
    
    with col2:
        st.metric("AI Probability", f"{int(res['score']*100)}%")
        st.subheader(f"Verdict: {res['label']}")
        st.info(f"**Analysis Note:** {res['reason']}")
        with st.expander("View Raw Forensic Logs"):
            st.code(res['technical_log'])
            
    if st.button("New Scan"):
        st.session_state.ui_phase = "upload"
        st.rerun()
