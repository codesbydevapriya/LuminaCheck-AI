import os
import streamlit as st
from PIL import Image
import requests
import io
import pandas as pd
from datetime import datetime
import time

HIVE_API_KEY = os.environ.get("HIVE_API_KEY")

st.set_page_config(page_title="LuminaCheck AI", page_icon="🔍", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Inter', sans-serif; box-sizing: border-box; }
.stApp { background: #f8fafc !important; }
[data-testid="stSidebar"] { background: #ffffff !important; border-right: 1px solid #e2e8f0 !important; }
[data-testid="stSidebar"] * { color: #1e293b !important; }
.stButton>button { background: #0f172a !important; color: white !important; border-radius: 8px !important; padding: 12px 24px !important; font-size: 14px !important; font-weight: 600 !important; border: none !important; transition: all 0.2s ease !important; width: 100% !important; }
.stButton>button:hover { background: #1e293b !important; transform: translateY(-1px) !important; box-shadow: 0 4px 12px rgba(15,23,42,0.3) !important; }
h1 { color: #0f172a !important; font-size: 2.8rem !important; font-weight: 800 !important; -webkit-text-fill-color: #0f172a !important; }
[data-testid="stFileUploader"] { background: #ffffff !important; border: 2px dashed #cbd5e1 !important; border-radius: 16px !important; padding: 30px !important; transition: all 0.3s ease !important; }
[data-testid="stFileUploader"]:hover { border-color: #6366f1 !important; box-shadow: 0 0 0 4px rgba(99,102,241,0.1) !important; }
.hero-section { background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%); border-radius: 20px; padding: 50px 40px; margin-bottom: 30px; position: relative; overflow: hidden; }
.hero-section::before { content: ''; position: absolute; top: -50%; right: -10%; width: 400px; height: 400px; background: radial-gradient(circle, rgba(99,102,241,0.3) 0%, transparent 70%); border-radius: 50%; }
.ts-card { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 16px; padding: 24px; transition: all 0.2s ease; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
.ts-card:hover { border-color: #6366f1; box-shadow: 0 4px 20px rgba(99,102,241,0.15); transform: translateY(-2px); }
.stat-number { font-size: 2.5rem; font-weight: 800; color: #ffffff; line-height: 1; margin: 0; }
.stat-label { font-size: 13px; color: rgba(255,255,255,0.6); margin-top: 4px; }
.ts-tag { display: inline-block; background: #f1f5f9; color: #475569; border: 1px solid #e2e8f0; border-radius: 6px; padding: 4px 12px; font-size: 12px; font-weight: 500; margin: 3px; }
.verdict-real { background: #f0fdf4; border: 2px solid #22c55e; border-radius: 16px; padding: 30px; text-align: center; box-shadow: 0 0 30px rgba(34,197,94,0.15); animation: fadeIn 0.5s ease-out; }
.verdict-fake { background: #fef2f2; border: 2px solid #ef4444; border-radius: 16px; padding: 30px; text-align: center; box-shadow: 0 0 30px rgba(239,68,68,0.15); animation: fadeIn 0.5s ease-out; }
.verdict-badge-real { background: #22c55e; color: white; font-size: 20px; font-weight: 700; padding: 12px 32px; border-radius: 50px; display: inline-block; letter-spacing: 1px; margin-bottom: 15px; }
.verdict-badge-fake { background: #ef4444; color: white; font-size: 20px; font-weight: 700; padding: 12px 32px; border-radius: 50px; display: inline-block; letter-spacing: 1px; margin-bottom: 15px; }
.scan-container { border-radius: 16px; overflow: hidden; border: 2px solid #e2e8f0; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
.chat-msg-user { background: #0f172a; color: white; border-radius: 18px 18px 4px 18px; padding: 12px 18px; margin: 8px 0; margin-left: 20%; font-size: 14px; }
.chat-msg-ai { background: #ffffff; border: 1px solid #e2e8f0; color: #334155; border-radius: 18px 18px 18px 4px; padding: 12px 18px; margin: 8px 0; margin-right: 20%; font-size: 14px; line-height: 1.7; }
.chat-widget { background: #ffffff; border: 1px solid #e2e8f0; border-radius: 20px; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
.chat-header { background: #0f172a; padding: 16px 20px; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
</style>
""", unsafe_allow_html=True)

LOGO_SVG = """<svg width="36" height="36" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <defs><radialGradient id="g1" cx="50%" cy="50%" r="50%">
    <stop offset="0%" style="stop-color:#6366f1"/>
    <stop offset="100%" style="stop-color:#06b6d4"/>
  </radialGradient></defs>
  <circle cx="90" cy="90" r="72" fill="#0f172a"/>
  <circle cx="90" cy="90" r="72" fill="none" stroke="url(#g1)" stroke-width="5"/>
  <ellipse cx="90" cy="90" rx="32" ry="32" fill="url(#g1)"/>
  <ellipse cx="90" cy="90" rx="19" ry="19" fill="#0f172a"/>
  <ellipse cx="90" cy="90" rx="10" ry="10" fill="url(#g1)" opacity="0.7"/>
  <circle cx="99" cy="81" r="5" fill="white" opacity="0.9"/>
  <line x1="132" y1="132" x2="155" y2="155" stroke="#6366f1" stroke-width="10" stroke-linecap="round"/>
</svg>"""

def detect_with_hive(image_bytes):
    try:
        response = requests.post(
            "https://api.thehive.ai/api/v2/task/sync",
            headers={"Authorization": f"Token {HIVE_API_KEY}"},
            files={"image": ("image.jpg", image_bytes, "image/jpeg")},
            data={"model": "ai_generated_image_detection"},
            timeout=15
        )
        if response.status_code == 200:
            data = response.json()
            classes = data["status"][0]["response"]["output"][0]["classes"]
            ai_score = 0
            real_score = 0
            for c in classes:
                if c["class"] == "ai_generated":
                    ai_score = c["score"]
                elif c["class"] == "real":
                    real_score = c["score"]
            return ai_score, real_score
    except Exception as e:
        return None, None
    return None, None

st.sidebar.markdown(f"""
<div style='display:flex; align-items:center; gap:10px; padding:10px 0; margin-bottom:10px;'>
    {LOGO_SVG}
    <div>
        <p style='color:#0f172a; font-weight:700; font-size:16px; margin:0;'>LuminaCheck AI</p>
        <p style='color:#64748b; font-size:11px; margin:0;'>Image Authentication</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<hr style='border-color:#e2e8f0; margin:10px 0;'>", unsafe_allow_html=True)
page = st.sidebar.radio("", ["Detect", "History", "About"], label_visibility="collapsed")
st.sidebar.markdown("<hr style='border-color:#e2e8f0; margin:10px 0;'>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#64748b; font-size:12px;'>Upload any image to detect if it is REAL or FAKE using Hive AI forensics.</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#94a3b8; font-size:11px; margin-top:20px;'>Powered by Hive AI</p>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def show_chat_widget():
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="chat-widget">
        <div class="chat-header">
            <p style='color:white; font-weight:700; font-size:15px; margin:0;'>AI Assistant</p>
            <p style='color:rgba(255,255,255,0.5); font-size:11px; margin:0;'>Ask anything about image forensics</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.chat_history:
        st.markdown("""
        <div style='background:#f8fafc; border:1px solid #e2e8f0; border-top:none; border-radius:0 0 16px 16px; padding:20px; text-align:center;'>
            <p style='color:#64748b; font-size:13px; margin-bottom:10px;'>Ask me anything about image detection!</p>
            <span class="ts-tag">What is a deepfake?</span>
            <span class="ts-tag">How does AI detect fake images?</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history[-6:]:
            if msg["role"] == "user":
                st.markdown(f'<div class="chat-msg-user">{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="chat-msg-ai">{msg["content"]}</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([5, 1, 1])
    with col1:
        user_input = st.text_input("", placeholder="Ask anything...", key="chat_bottom", label_visibility="collapsed")
    with col2:
        send = st.button("Send", key="send_bottom")
    with col3:
        if st.button("Clear", key="clear_bottom"):
            st.session_state.chat_history = []
            st.rerun()

    if send and user_input:
        import google.generativeai as genai
        genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.spinner("Thinking..."):
            model = genai.GenerativeModel("gemini-2.5-flash")
            response = model.generate_content(f"""You are LuminaCheck AI Assistant, expert in image forensics and deepfake detection.
Answer in 2-3 short sentences only.
Question: {user_input}""")
        st.session_state.chat_history.append({"role": "assistant", "content": response.text})
        st.rerun()

if page == "Detect":
    st.markdown("""
    <div class="hero-section">
        <div style='position:relative; z-index:1;'>
            <p style='color:rgba(255,255,255,0.6); font-size:13px; margin:0 0 8px 0; letter-spacing:2px; text-transform:uppercase;'>AI Image Detection</p>
            <h1 style='color:white !important; font-size:2.5rem; font-weight:800; margin:0 0 15px 0; -webkit-text-fill-color:white !important;'>LuminaCheck AI</h1>
            <p style='color:rgba(255,255,255,0.7); font-size:16px; margin:0 0 25px 0; max-width:600px;'>
                Fast, accurate AI image authentication powered by Hive AI specialized detection engine.
            </p>
            <div style='display:flex; gap:30px; flex-wrap:wrap;'>
                <div><p class='stat-number'>Hive AI</p><p class='stat-label'>Specialized Detection</p></div>
                <div style='width:1px; background:rgba(255,255,255,0.1);'></div>
                <div><p class='stat-number'>Fast</p><p class='stat-label'>Sub-second Results</p></div>
                <div style='width:1px; background:rgba(255,255,255,0.1);'></div>
                <div><p class='stat-number'>Free</p><p class='stat-label'>No Cost Detection</p></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="ts-card">
            <p style='color:#6366f1; font-weight:700; font-size:15px; margin:0;'>Hive AI Engine</p>
            <p style='color:#64748b; font-size:13px; margin:6px 0 0 0;'>Specialized AI image detection model</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="ts-card">
            <p style='color:#06b6d4; font-weight:700; font-size:15px; margin:0;'>Confidence Score</p>
            <p style='color:#64748b; font-size:13px; margin:6px 0 0 0;'>0-100% AI probability rating</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="ts-card">
            <p style='color:#10b981; font-weight:700; font-size:15px; margin:0;'>Instant Results</p>
            <p style='color:#64748b; font-size:13px; margin:6px 0 0 0;'>Fast detection with progress bar</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="ts-card" style='margin-bottom:20px;'>
        <p style='color:#0f172a; font-weight:700; font-size:16px; margin:0 0 5px 0;'>Upload Image</p>
        <p style='color:#64748b; font-size:13px; margin:0;'>Supports JPG, JPEG, PNG up to 200MB</p>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

    if uploaded_file is not None:
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            image = Image.open(uploaded_file)
            st.markdown('<div class="scan-container">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="ts-card">
                <p style='color:#0f172a; font-weight:700; font-size:15px; margin:0 0 15px 0;'>File Details</p>
                <div style='background:#f8fafc; border-radius:8px; padding:12px; margin-bottom:10px;'>
                    <p style='color:#64748b; font-size:12px; margin:0;'>File Name</p>
                    <p style='color:#0f172a; font-weight:600; font-size:14px; margin:4px 0 0 0;'>{uploaded_file.name}</p>
                </div>
                <div style='background:#f8fafc; border-radius:8px; padding:12px; margin-bottom:10px;'>
                    <p style='color:#64748b; font-size:12px; margin:0;'>File Size</p>
                    <p style='color:#0f172a; font-weight:600; font-size:14px; margin:4px 0 0 0;'>{uploaded_file.size / 1024:.1f} KB</p>
                </div>
                <div style='background:#f0fdf4; border:1px solid #86efac; border-radius:8px; padding:12px;'>
                    <p style='color:#166534; font-weight:600; font-size:13px; margin:0;'>Ready for analysis</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("Analyze Image"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.markdown("<p style='color:#6366f1; font-size:14px; font-weight:600;'>Preparing image...</p>", unsafe_allow_html=True)
                for i in range(0, 25):
                    time.sleep(0.02)
                    progress_bar.progress(i)

                img_bytes = io.BytesIO()
                image.save(img_bytes, format="JPEG")

                status_text.markdown("<p style='color:#6366f1; font-size:14px; font-weight:600;'>Hive AI scanning...</p>", unsafe_allow_html=True)
                for i in range(25, 50):
                    time.sleep(0.02)
                    progress_bar.progress(i)

                hive_ai, hive_real = detect_with_hive(img_bytes.getvalue())

                for i in range(50, 95):
                    time.sleep(0.01)
                    progress_bar.progress(i)

                status_text.markdown("<p style='color:#22c55e; font-size:14px; font-weight:600;'>Analysis complete!</p>", unsafe_allow_html=True)
                progress_bar.progress(100)
                time.sleep(0.4)
                progress_bar.empty()
                status_text.empty()

                st.markdown("---")

                if hive_ai is not None:
                    hive_percent = round(hive_ai * 100)
                    real_percent = round(hive_real * 100)

                    st.markdown(f"""
                    <div class="ts-card" style='margin-bottom:15px;'>
                        <p style='color:#0f172a; font-weight:700; font-size:15px; margin:0 0 15px 0;'>Detection Score</p>
                        <div style='display:flex; gap:20px;'>
                            <div style='flex:1; background:#fef2f2; border-radius:12px; padding:15px; text-align:center;'>
                                <p style='color:#ef4444; font-size:32px; font-weight:800; margin:0;'>{hive_percent}%</p>
                                <p style='color:#64748b; font-size:12px; margin:4px 0 0 0;'>AI Generated Probability</p>
                            </div>
                            <div style='flex:1; background:#f0fdf4; border-radius:12px; padding:15px; text-align:center;'>
                                <p style='color:#22c55e; font-size:32px; font-weight:800; margin:0;'>{real_percent}%</p>
                                <p style='color:#64748b; font-size:12px; margin:4px 0 0 0;'>Real Image Probability</p>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    if hive_ai > 0.5:
                        verdict = "FAKE/AI-GENERATED"
                        st.markdown(f"""
                        <div class="verdict-fake">
                            <div class="verdict-badge-fake">FAKE / AI-GENERATED</div>
                            <p style='color:#7f1d1d; font-size:15px; margin-top:10px;'>This image has a <b>{hive_percent}%</b> probability of being AI-generated or fake.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        verdict = "REAL"
                        st.markdown(f"""
                        <div class="verdict-real">
                            <div class="verdict-badge-real">REAL IMAGE VERIFIED</div>
                            <p style='color:#14532d; font-size:15px; margin-top:10px;'>This image has a <b>{real_percent}%</b> probability of being a genuine photograph.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    verdict = "ERROR"
                    st.error("Detection failed. Please try again.")

                st.session_state.history.append({
                    "Time": datetime.now().strftime("%H:%M:%S"),
                    "File": uploaded_file.name,
                    "Result": verdict,
                    "Details": f"AI: {round((hive_ai or 0)*100)}% | Real: {round((hive_real or 0)*100)}%"
                })

    show_chat_widget()

elif page == "History":
    st.markdown("""<h1 style='color:#0f172a !important; -webkit-text-fill-color:#0f172a !important;'>Detection History</h1>""", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#e2e8f0;'>", unsafe_allow_html=True)
    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False)
        st.download_button("Download Report (CSV)", csv, "lumina_report.csv", "text/csv")
    else:
        st.markdown("""
        <div class="ts-card" style='text-align:center; padding:50px;'>
            <p style='color:#64748b; font-size:18px; font-weight:600; margin:0;'>No detections yet</p>
            <p style='color:#94a3b8; font-size:14px;'>Go to Detect page and upload an image</p>
        </div>
        """, unsafe_allow_html=True)
    show_chat_widget()

elif page == "About":
    st.markdown("""<h1 style='color:#0f172a !important; -webkit-text-fill-color:#0f172a !important;'>About LuminaCheck AI</h1>""", unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#e2e8f0;'>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="ts-card">
            <p style='color:#6366f1; font-weight:700; font-size:15px; margin:0 0 10px 0;'>What is LuminaCheck AI?</p>
            <p style='color:#475569; font-size:14px; line-height:1.7; margin:0;'>
            LuminaCheck AI uses Hive AI — a specialized image detection engine — to accurately determine whether a digital image is REAL, FAKE, or AI-GENERATED with instant results.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="ts-card">
            <p style='color:#06b6d4; font-weight:700; font-size:15px; margin:0 0 10px 0;'>Technologies Used</p>
            <p style='color:#475569; font-size:14px; line-height:2; margin:0;'>
            Python | Streamlit<br>
            Hive AI (Specialized Detection)<br>
            Google Gemini AI (Chatbot)<br>
            Streamlit Cloud
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="ts-card" style='text-align:center; background:linear-gradient(135deg,#0f172a,#1e293b); border:none;'>
        <p style='color:#94a3b8; font-size:13px; margin:0 0 5px 0;'>Developed By</p>
        <p style='color:#ffffff; font-size:22px; font-weight:700; margin:0;'>Devapriya</p>
        <p style='color:#64748b; font-size:13px; margin:5px 0 15px 0;'>BCA Final Year Student | March 2026</p>
        <span class="ts-tag" style='background:rgba(255,255,255,0.1); color:#94a3b8; border-color:rgba(255,255,255,0.1);'>luminacheck-ai.streamlit.app</span>
    </div>
    """, unsafe_allow_html=True)
    show_chat_widget()
