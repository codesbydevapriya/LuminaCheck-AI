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
