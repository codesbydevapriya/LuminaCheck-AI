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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

.stApp { background: #050a14 !important; }

.stApp::before {
    content: '';
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    animation: bgMove 10s ease-in-out infinite alternate;
    pointer-events: none;
    z-index: 0;
}

@keyframes bgMove {
    0% { background: radial-gradient(ellipse at 20% 50%, rgba(0,212,170,0.08) 0%, transparent 50%), radial-gradient(ellipse at 80% 20%, rgba(0,153,255,0.06) 0%, transparent 50%); }
    50% { background: radial-gradient(ellipse at 70% 30%, rgba(0,212,170,0.08) 0%, transparent 50%), radial-gradient(ellipse at 20% 70%, rgba(0,153,255,0.06) 0%, transparent 50%); }
    100% { background: radial-gradient(ellipse at 30% 20%, rgba(0,212,170,0.08) 0%, transparent 50%), radial-gradient(ellipse at 60% 80%, rgba(0,153,255,0.06) 0%, transparent 50%); }
}

.orb { position: fixed; border-radius: 50%; filter: blur(60px); pointer-events: none; z-index: 0; opacity: 0.15; }
.orb1 { width: 400px; height: 400px; background: #00d4aa; top: -100px; left: -100px; animation: orbFloat1 15s ease-in-out infinite; }
.orb2 { width: 300px; height: 300px; background: #0099ff; bottom: -50px; right: -50px; animation: orbFloat2 12s ease-in-out infinite; }
.orb3 { width: 200px; height: 200px; background: #c9a84c; top: 50%; left: 50%; animation: orbFloat3 18s ease-in-out infinite; }

@keyframes orbFloat1 { 0%, 100% { transform: translate(0,0) scale(1); } 33% { transform: translate(100px,80px) scale(1.1); } 66% { transform: translate(-50px,150px) scale(0.9); } }
@keyframes orbFloat2 { 0%, 100% { transform: translate(0,0) scale(1); } 50% { transform: translate(-120px,-80px) scale(1.2); } }
@keyframes orbFloat3 { 0%, 100% { transform: translate(-50%,-50%) scale(1); } 33% { transform: translate(-30%,-70%) scale(1.3); } 66% { transform: translate(-70%,-30%) scale(0.8); } }
@keyframes twinkle { 0%, 100% { opacity: 0.3; transform: scale(1); } 50% { opacity: 1; transform: scale(1.5); } }
@keyframes float { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-8px); } }
@keyframes glow { 0%, 100% { box-shadow: 0 0 20px rgba(0,212,170,0.3); } 50% { box-shadow: 0 0 40px rgba(0,212,170,0.7); } }
@keyframes slideIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
@keyframes rotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@keyframes redPulse { 0%, 100% { box-shadow: 0 0 30px rgba(255,59,48,0.4); } 50% { box-shadow: 0 0 60px rgba(255,59,48,0.8); } }
@keyframes greenPulse { 0%, 100% { box-shadow: 0 0 30px rgba(0,212,170,0.4); } 50% { box-shadow: 0 0 60px rgba(0,212,170,0.8); } }

[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1421 0%, #111827 100%) !important; border-right: 1px solid #1e3a5f; }
.stButton>button { background: linear-gradient(135deg, #00d4aa, #0099ff) !important; color: white !important; border-radius: 12px !important; padding: 14px 28px !important; font-size: 15px !important; font-weight: 600 !important; border: none !important; box-shadow: 0 0 20px rgba(0,212,170,0.3) !important; transition: all 0.3s ease !important; width: 100% !important; }
.stButton>button:hover { transform: translateY(-3px) !important; box-shadow: 0 0 35px rgba(0,212,170,0.6) !important; }
h1 { background: linear-gradient(135deg, #00d4aa, #0099ff, #c9a84c); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.8rem !important; font-weight: 700 !important; }
[data-testid="stFileUploader"] { background: #0d1421; border: 2px dashed #1e3a5f; border-radius: 16px; }

.hero-logo { animation: float 3s ease-in-out infinite; display: inline-block; }
.sidebar-logo { animation: float 4s ease-in-out infinite; }
.scan-container { position: relative; overflow: hidden; border-radius: 16px; border: 2px solid #00d4aa; animation: glow 2s ease-in-out infinite; }
.stat-card { background: linear-gradient(135deg, rgba(13,20,33,0.8), rgba(17,24,39,0.8)); border: 1px solid #1e3a5f; border-radius: 16px; padding: 20px; text-align: center; transition: all 0.3s ease; animation: slideIn 0.5s ease-out; backdrop-filter: blur(10px); }
.stat-card:hover { border-color: #00d4aa; transform: translateY(-5px); box-shadow: 0 10px 30px rgba(0,212,170,0.2); }
.spinning-loader { width: 50px; height: 50px; border: 4px solid #1e3a5f; border-top: 4px solid #00d4aa; border-radius: 50%; animation: rotate 1s linear infinite; margin: 0 auto; }
.tag { display: inline-block; background: rgba(0,212,170,0.15); border: 1px solid #00d4aa; color: #00d4aa; border-radius: 20px; padding: 4px 12px; font-size: 12px; font-weight: 500; margin: 3px; }

.verdict-real {
    background: linear-gradient(135deg, rgba(0,212,170,0.2), rgba(0,153,255,0.1));
    border: 3px solid #00d4aa;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    animation: greenPulse 2s ease-in-out infinite, slideIn 0.6s ease-out;
}
.verdict-fake {
    background: linear-gradient(135deg, rgba(255,59,48,0.2), rgba(255,100,50,0.1));
    border: 3px solid #ff3b30;
    border-radius: 20px;
    padding: 30px;
    text-align: center;
    animation: redPulse 2s ease-in-out infinite, slideIn 0.6s ease-out;
}
.verdict-badge-real {
    background: #00d4aa;
    color: #050a14;
    font-size: 22px;
    font-weight: 800;
    padding: 12px 30px;
    border-radius: 50px;
    display: inline-block;
    letter-spacing: 2px;
    margin-bottom: 15px;
}
.verdict-badge-fake {
    background: #ff3b30;
    color: white;
    font-size: 22px;
    font-weight: 800;
    padding: 12px 30px;
    border-radius: 50px;
    display: inline-block;
    letter-spacing: 2px;
    margin-bottom: 15px;
}
.star { position: fixed; width: 2px; height: 2px; background: white; border-radius: 50%; animation: twinkle ease-in-out infinite; pointer-events: none; z-index: 0; }
</style>

<div class="orb orb1"></div>
<div class="orb orb2"></div>
<div class="orb orb3"></div>

<div class="star" style="top:5%;left:10%;animation-duration:2s;"></div>
<div class="star" style="top:15%;left:25%;animation-duration:3s;animation-delay:0.5s;width:3px;height:3px;background:#00d4aa;"></div>
<div class="star" style="top:8%;left:45%;animation-duration:2.5s;animation-delay:1s;"></div>
<div class="star" style="top:20%;left:65%;animation-duration:4s;animation-delay:0.3s;width:3px;height:3px;background:#0099ff;"></div>
<div class="star" style="top:12%;left:80%;animation-duration:2s;animation-delay:1.5s;"></div>
<div class="star" style="top:35%;left:5%;animation-duration:3s;animation-delay:0.8s;width:3px;height:3px;background:#c9a84c;"></div>
<div class="star" style="top:45%;left:90%;animation-duration:2.5s;animation-delay:0.2s;"></div>
<div class="star" style="top:60%;left:15%;animation-duration:3.5s;animation-delay:1.2s;width:3px;height:3px;background:#00d4aa;"></div>
<div class="star" style="top:70%;left:35%;animation-duration:2s;animation-delay:0.6s;"></div>
<div class="star" style="top:80%;left:55%;animation-duration:4s;animation-delay:0.9s;width:3px;height:3px;background:#0099ff;"></div>
<div class="star" style="top:90%;left:75%;animation-duration:2.5s;animation-delay:1.8s;"></div>
<div class="star" style="top:25%;left:50%;animation-duration:3s;animation-delay:0.4s;width:3px;height:3px;background:#c9a84c;"></div>
<div class="star" style="top:55%;left:70%;animation-duration:2s;animation-delay:1.1s;"></div>
<div class="star" style="top:40%;left:40%;animation-duration:3.5s;animation-delay:0.7s;width:3px;height:3px;background:#00d4aa;"></div>
<div class="star" style="top:75%;left:88%;animation-duration:2.5s;animation-delay:1.4s;"></div>
""", unsafe_allow_html=True)

LOGO_SVG = """<svg width="55" height="55" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <defs><radialGradient id="g1" cx="50%" cy="50%" r="50%">
    <stop offset="0%" style="stop-color:#00d4aa"/>
    <stop offset="100%" style="stop-color:#0099ff"/>
  </radialGradient></defs>
  <circle cx="90" cy="90" r="72" fill="#0d1421"/>
  <circle cx="90" cy="90" r="72" fill="none" stroke="url(#g1)" stroke-width="5"/>
  <circle cx="90" cy="90" r="72" fill="none" stroke="#c9a84c" stroke-width="1.5" stroke-dasharray="8,6" opacity="0.4"/>
  <ellipse cx="90" cy="90" rx="32" ry="32" fill="url(#g1)"/>
  <ellipse cx="90" cy="90" rx="19" ry="19" fill="#0d1421"/>
  <ellipse cx="90" cy="90" rx="10" ry="10" fill="url(#g1)" opacity="0.7"/>
  <circle cx="99" cy="81" r="6" fill="white" opacity="0.9"/>
  <line x1="132" y1="132" x2="158" y2="158" stroke="#c9a84c" stroke-width="11" stroke-linecap="round"/>
  <line x1="132" y1="132" x2="158" y2="158" stroke="url(#g1)" stroke-width="6" stroke-linecap="round"/>
</svg>"""

LOGO_BIG = """<svg width="80" height="80" viewBox="0 0 200 200" xmlns="http://www.w3.org/2000/svg">
  <defs><radialGradient id="g2" cx="50%" cy="50%" r="50%">
    <stop offset="0%" style="stop-color:#00d4aa"/>
    <stop offset="100%" style="stop-color:#0099ff"/>
  </radialGradient></defs>
  <circle cx="90" cy="90" r="72" fill="#0d1421"/>
  <circle cx="90" cy="90" r="72" fill="none" stroke="url(#g2)" stroke-width="5"/>
  <circle cx="90" cy="90" r="72" fill="none" stroke="#c9a84c" stroke-width="1.5" stroke-dasharray="8,6" opacity="0.4"/>
  <ellipse cx="90" cy="90" rx="32" ry="32" fill="url(#g2)"/>
  <ellipse cx="90" cy="90" rx="19" ry="19" fill="#0d1421"/>
  <ellipse cx="90" cy="90" rx="10" ry="10" fill="url(#g2)" opacity="0.7"/>
  <circle cx="99" cy="81" r="6" fill="white" opacity="0.9"/>
  <line x1="132" y1="132" x2="158" y2="158" stroke="#c9a84c" stroke-width="11" stroke-linecap="round"/>
  <line x1="132" y1="132" x2="158" y2="158" stroke="url(#g2)" stroke-width="6" stroke-linecap="round"/>
</svg>"""

page = st.sidebar.radio("Navigation", ["🔍 Detect", "📋 History", "ℹ️ About"])
st.sidebar.markdown("---")
st.sidebar.markdown(f'<div class="sidebar-logo">{LOGO_SVG}</div>', unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#00d4aa; font-weight:700; font-size:17px; margin:5px 0;'>LuminaCheck AI</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#555; font-size:12px; font-style:italic;'>Where Light Reveals Truth</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("<p style='color:#8899aa; font-size:13px;'>👋 Welcome to LuminaCheck AI!</p>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='color:#8899aa; font-size:12px;'>📌 Upload any image to detect if it is REAL or FAKE using advanced AI forensics.</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.markdown("<p style='color:#333; font-size:11px; text-align:center;'>⚡ Powered by Google Gemini AI</p>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

if page == "🔍 Detect":
    col1, col2 = st.columns([1, 7])
    with col1:
        st.markdown(f'<div class="hero-logo">{LOGO_BIG}</div>', unsafe_allow_html=True)
    with col2:
        st.title("LuminaCheck AI")
        st.markdown("<p style='color:#8899aa; font-size:16px; margin-top:-10px;'>Advanced AI-Powered Image Authenticity Detection</p>", unsafe_allow_html=True)
        st.markdown("""
        <span class="tag">🤖 Gemini AI</span>
        <span class="tag">🔬 Forensic Analysis</span>
        <span class="tag">⚡ Real-time</span>
        <span class="tag">🛡️ Secure</span>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="stat-card">
            <h2 style='color:#00d4aa; font-size:2rem; margin:0;'>AI</h2>
            <p style='color:#8899aa; margin:5px 0 0 0; font-size:13px;'>Powered Detection</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="stat-card">
            <h2 style='color:#0099ff; font-size:2rem; margin:0;'>3x</h2>
            <p style='color:#8899aa; margin:5px 0 0 0; font-size:13px;'>Analysis Modes</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="stat-card">
            <h2 style='color:#c9a84c; font-size:2rem; margin:0;'>⚡</h2>
            <p style='color:#8899aa; margin:5px 0 0 0; font-size:13px;'>Instant Results</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8899aa; font-size:14px;'>📤 Upload Image</p>", unsafe_allow_html=True)
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
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="stat-card">
                <p style='color:#00d4aa; font-weight:600; font-size:15px; margin:0;'>📁 File Details</p>
                <hr style='border-color:#1e3a5f; margin:10px 0;'>
                <p style='color:#8899aa; font-size:13px; margin:5px 0;'>📄 <b style='color:#fff;'>{uploaded_file.name}</b></p>
                <p style='color:#8899aa; font-size:13px; margin:5px 0;'>💾 Size: <b style='color:#fff;'>{uploaded_file.size / 1024:.1f} KB</b></p>
                <p style='color:#00d4aa; font-size:13px; margin:5px 0;'>✅ Ready for analysis</p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            if st.button("🔍 Analyze Image Now"):
                result_placeholder = st.empty()
                for msg in [
                    "🔍 Initializing forensic scanner...",
                    "🧬 Loading AI model...",
                    "🤖 Gemini AI analyzing image...",
                    "📊 Processing results..."
                ]:
                    result_placeholder.markdown(f"""
                    <div style='text-align:center; padding:30px;'>
                        <div class='spinning-loader'></div>
                        <p style='color:#00d4aa; font-size:16px; margin-top:20px; font-weight:600;'>{msg}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    time.sleep(0.8)

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
                result = response.text
                result_placeholder.empty()

                if "FAKE" in result.upper() or "AI-GENERATED" in result.upper():
                    verdict = "FAKE/AI-GENERATED"
                    st.markdown(f"""
                    <div class="verdict-fake">
                        <div style='font-size:60px; margin-bottom:10px;'>🚨</div>
                        <div class="verdict-badge-fake">⚠️ FAKE / AI-GENERATED</div>
                        <p style='color:#ffaa99; font-size:14px; margin-top:15px; line-height:1.7;'>{result}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    verdict = "REAL"
                    st.markdown(f"""
                    <div class="verdict-real">
                        <div style='font-size:60px; margin-bottom:10px;'>✅</div>
                        <div class="verdict-badge-real">✅ REAL IMAGE VERIFIED</div>
                        <p style='color:#99ffee; font-size:14px; margin-top:15px; line-height:1.7;'>{result}</p>
                    </div>
                    """, unsafe_allow_html=True)

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
        st.markdown("""
        <div style='text-align:center; padding:50px;'>
            <p style='font-size:50px;'>📭</p>
            <p style='color:#8899aa; font-size:18px;'>No detections yet</p>
            <p style='color:#555; font-size:14px;'>Go to Detect page and upload an image!</p>
        </div>
        """, unsafe_allow_html=True)

elif page == "ℹ️ About":
    col1, col2 = st.columns([1, 8])
    with col1:
        st.markdown(f'<div class="hero-logo">{LOGO_BIG}</div>', unsafe_allow_html=True)
    with col2:
        st.title("About LuminaCheck AI")
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="stat-card">
            <h3 style='color:#00d4aa;'>🔍 What is LuminaCheck AI?</h3>
            <p style='color:#8899aa; font-size:14px; line-height:1.7;'>
            LuminaCheck AI is a Final Year BCA Project that uses Google Gemini Vision AI to detect whether a digital image is REAL, FAKE, or AI-GENERATED.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="stat-card">
            <h3 style='color:#0099ff;'>🛠️ Technologies Used</h3>
            <p style='color:#8899aa; font-size:14px; line-height:1.8;'>
            🐍 Python &nbsp;|&nbsp; ⚡ Streamlit<br>
            🤖 Google Gemini AI<br>
            🖼️ Pillow &nbsp;|&nbsp; 📊 Pandas<br>
            ☁️ Streamlit Cloud
            </p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="stat-card" style='text-align:center;'>
        <h3 style='color:#c9a84c;'>👩‍💻 Developed By</h3>
        <p style='color:#fff; font-size:20px; font-weight:600;'>Devapriya</p>
        <p style='color:#8899aa;'>BCA Final Year Student | March 2026</p>
        <br>
        <span class="tag">🌐 luminacheck-ai.streamlit.app</span>
        <span class="tag">📂 github.com/codesbydevapriya</span>
    </div>
    """, unsafe_allow_html=True)
