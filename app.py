import os
import streamlit as st
from PIL import Image
import google.generativeai as genai
import pandas as pd
from datetime import datetime
import time

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

st.set_page_config(page_title="LuminaCheck AI", page_icon="🔍", layout="wide")

# Animated background particles
st.markdown("""
<div id="particles-bg"></div>
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

#particles-bg {
    position: fixed;
    top: 0; left: 0;
    width: 100%; height: 100%;
    pointer-events: none;
    z-index: 0;
    overflow: hidden;
}

.particle {
    position: fixed;
    border-radius: 50%;
    pointer-events: none;
    animation: floatUp linear infinite;
    opacity: 0;
}

@keyframes floatUp {
    0% { transform: translateY(100vh) scale(0); opacity: 0; }
    10% { opacity: 0.6; }
    90% { opacity: 0.3; }
    100% { transform: translateY(-10vh) scale(1); opacity: 0; }
}
@keyframes drift {
    0%, 100% { margin-left: 0px; }
    50% { margin-left: 30px; }
}

.stApp { background: #050a14 !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1421 0%, #111827 100%) !important;
    border-right: 1px solid #1e3a5f;
}
.stButton>button {
    background: linear-gradient(135deg, #00d4aa, #0099ff) !important;
    color: white !important;
    border-radius: 12px !important;
    padding: 14px 28px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    border: none !important;
    box-shadow: 0 0 20px rgba(0, 212, 170, 0.3) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}
.stButton>button:hover {
    transform: translateY(-3px) !important;
    box-shadow: 0 0 35px rgba(0, 212, 170, 0.6) !important;
}
h1 {
    background: linear-gradient(135deg, #00d4aa, #0099ff, #c9a84c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
}
[data-testid="stFileUploader"] {
    background: #0d1421;
    border: 2px dashed #1e3a5f;
    border-radius: 16px;
}
.stSuccess {
    background: rgba(0, 212, 170, 0.1) !important;
    border: 1px solid #00d4aa !important;
    border-radius: 12px !important;
}
.stError {
    background: rgba(255, 59, 48, 0.1) !important;
    border: 1px solid #ff3b30 !important;
    border-radius: 12px !important;
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
}
@keyframes glow {
    0%, 100% { box-shadow: 0 0 20px rgba(0,212,170,0.3); }
    50% { box-shadow: 0 0 40px rgba(0,212,170,0.7); }
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes rotate {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
}
.hero-logo { animation: float 3s ease-in-out infinite; display: inline-block; }
.sidebar-logo { animation: float 4s ease-in-out infinite; }
.scan-container {
    position: relative;
    overflow: hidden;
    border-radius: 16px;
    border: 2px solid #00d4aa;
    animation: glow 2s ease-in-out infinite;
}
.stat-card {
    background: linear-gradient(135deg, #0d1421, #111827);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    transition: all 0.3s ease;
    animation: slideIn 0.5s ease-out;
}
.stat-card:hover {
    border-color: #00d4aa;
    transform: translateY(-5px);
    box-shadow: 0 10px 30px rgba(0,212,170,0.2);
}
.verdict-box {
    border-radius: 16px;
    padding: 25px;
    animation: slideIn 0.6s ease-out;
    text-align: center;
}
.verdict-real {
    background: linear-gradient(135deg, rgba(0,212,170,0.15), rgba(0,153,255,0.15));
    border: 2px solid #00d4aa;
    box-shadow: 0 0 30px rgba(0,212,170,0.2);
}
.verdict-fake {
    background: linear-gradient(135deg, rgba(255,59,48,0.15), rgba(255,100,50,0.15));
    border: 2px solid #ff3b30;
    box-shadow: 0 0 30px rgba(255,59,48,0.2);
}
.spinning-loader {
    width: 50px; height: 50px;
    border: 4px solid #1e3a5f;
    border-top: 4px solid #00d4aa;
    border-radius: 50%;
    animation: rotate 1s linear infinite;
    margin: 0 auto;
}
.tag {
    display: inline-block;
    background: rgba(0,212,170,0.15);
    border: 1px solid #00d4aa;
    color: #00d4aa;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 12px;
    font-weight: 500;
    margin: 3px;
}
</style>

<script>
const colors = ['#00d4aa', '#0099ff', '#c9a84c', '#7c3aed', '#00d4aa'];
const bg = document.getElementById('particles-bg');

function createParticle() {
    const p = document.createElement('div');
    p.className = 'particle';
    const size = Math.random() * 6 + 2;
    const x = Math.random() * 100;
    const duration = Math.random() * 15 + 8;
    const delay = Math.random() * 5;
    const color = colors[Math.floor(Math.random() * colors.length)];
    p.style.cssText = `
        width: ${size}px;
        height: ${size}px;
        left: ${x}%;
        background: ${color};
        box-shadow: 0 0 ${size * 2}px ${color};
        animation-duration: ${duration}s;
        animation-delay: ${delay}s;
    `;
    bg.appendChild(p);
    setTimeout(() => p.remove(), (duration + delay) * 1000);
}

setInterval(createParticle, 300);
for(let i = 0; i < 20; i++) setTimeout(createParticle, i * 150);
</script>
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
                    <div class="verdict-box verdict-fake">
                        <h2 style='color:#ff3b30; margin:0;'>⚠️ FAKE / AI-GENERATED</h2>
                        <p style='color:#8899aa; font-size:14px; margin-top:15px;'>{result}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    verdict = "REAL"
                    st.markdown(f"""
                    <div class="verdict-box verdict-real">
                        <h2 style='color:#00d4aa; margin:0;'>✅ REAL IMAGE VERIFIED</h2>
                        <p style='color:#8899aa; font-size:14px; margin-top:15px;'>{result}</p>
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
