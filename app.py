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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

st.set_page_config(page_title="LuminaCheck AI", layout="wide")

st.markdown("""
<style>
#MainMenu, header, footer {display:none;}
.block-container {padding:0;}
.stApp {background:#0c0c10;}
[data-testid="stFileUploader"] {display:none;}
</style>
""", unsafe_allow_html=True)

# ---------------- STATE ----------------
for k,v in {"history":[], "last_result":None}.items():
    if k not in st.session_state:
        st.session_state[k]=v

# ---------------- METADATA ----------------
def analyze_metadata(image):
    try:
        exif=image.getexif()
        if not exif: return 0.45,"No EXIF"
        text=" ".join([str(v).lower() for v in exif.values()])
        if "midjourney" in text or "dalle" in text:
            return 0.92,"AI tool in EXIF"
        if "canon" in text or "iphone" in text:
            return 0.12,"Camera detected"
        return 0.38,"Neutral EXIF"
    except:
        return 0.45,"EXIF error"

# ---------------- FORENSICS ----------------
def analyze_forensics(image):
    try:
        gray=image.convert("L").resize((256,256))
        arr=np.array(gray)
        var=arr.var()
        if var<200 or var>3000: return 0.7,"Abnormal variance"
        if 400<var<1800: return 0.3,"Natural variance"
        return 0.5,"Neutral"
    except:
        return 0.5,"Error"

# ---------------- FILENAME ----------------
def analyze_filename(name):
    n=name.lower()
    if "ai" in n or "generated" in n:
        return 0.75,"AI keyword"
    return 0.4,"Neutral"

# ---------------- GEMINI ----------------
def resize_for_gemini(img):
    w,h=img.size
    if max(w,h)<768: return img
    r=768/max(w,h)
    return img.resize((int(w*r),int(h*r)))

def detect_with_gemini(image):
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model=genai.GenerativeModel("gemini-2.5-flash")

        prompt="""
You are a forensic AI image analyst.

Respond EXACTLY:

SCORE: [0-100]

AI_INDICATORS:
- bullet
- bullet

REAL_INDICATORS:
- bullet
- bullet

SUMMARY:
One short line
"""

        img=resize_for_gemini(image)

        for _ in range(3):
            try:
                r=model.generate_content([prompt,img])
                text=r.text.strip()

                score=0.5
                ai=[]
                real=[]
                summary=""

                m=re.search(r"SCORE:\s*(\d+)",text)
                if m: score=int(m.group(1))/100

                ai_block=re.search(r"AI_INDICATORS:(.*?)(REAL_INDICATORS:|SUMMARY:)",text,re.S)
                if ai_block:
                    ai=re.findall(r"-\s*(.+)",ai_block.group(1))

                real_block=re.search(r"REAL_INDICATORS:(.*?)(SUMMARY:)",text,re.S)
                if real_block:
                    real=re.findall(r"-\s*(.+)",real_block.group(1))

                sm=re.search(r"SUMMARY:\s*(.+)",text,re.S)
                if sm: summary=sm.group(1).strip()

                return score,{"ai":ai,"real":real,"summary":summary}

            except Exception as e:
                if "429" in str(e): time.sleep(2)

        return 0.5,{"ai":[],"real":[],"summary":"Unavailable"}

    except:
        return 0.5,{"ai":[],"real":[],"summary":"Error"}

# ---------------- FUSION ----------------
def detect(image,filename):
    g_score,reason=detect_with_gemini(image)
    m_score,_=analyze_metadata(image)
    f_score,_=analyze_forensics(image)
    n_score,_=analyze_filename(filename)

    final=(0.6*g_score)+(0.2*m_score)+(0.15*f_score)+(0.05*n_score)
    return round(final,3),reason,g_score,m_score,f_score,n_score

def classify(s):
    if s>0.75:return"AI Generated"
    if s<0.35:return"Likely Real"
    return"Suspicious"

# ---------------- UI ----------------
uploaded=st.file_uploader("upload",type=["jpg","png","jpeg"],label_visibility="collapsed")

result_json="null"

if uploaded:
    img=Image.open(uploaded).convert("RGB")
    score,reason,g,m,f,n=detect(img,uploaded.name)
    label=classify(score)

    st.session_state.history.append({"File":uploaded.name,"Score":round(score*100),"Result":label})

    buf=io.BytesIO()
    img.save(buf,format="JPEG")
    img_b64=base64.b64encode(buf.getvalue()).decode()

    result_json=json.dumps({
        "score":score,
        "label":label,
        "reason":reason,
        "gemini":g,
        "meta":m,
        "forensic":f,
        "file":n,
        "img":img_b64
    })

HTML=f"""
<div style="color:white;padding:20px;font-family:sans-serif">
<h2>LuminaCheck AI</h2>
<script>
let d={result_json};

if(d && d.score!=undefined){{
document.write("<h3>"+Math.round(d.score*100)+"% AI</h3>");
document.write("<b>"+d.label+"</b><br><br>");

document.write("<b>AI Indicators:</b><ul>"+(d.reason.ai||[]).map(x=>"<li>"+x+"</li>").join("")+"</ul>");
document.write("<b>Real Indicators:</b><ul>"+(d.reason.real||[]).map(x=>"<li>"+x+"</li>").join("")+"</ul>");
document.write("<b>Summary:</b> "+d.reason.summary);
}}
</script>
</div>
"""

components.html(HTML,height=600)
