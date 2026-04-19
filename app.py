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

# ─── CONFIG ────────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

st.set_page_config(
    page_title="LuminaCheck AI",
    layout="wide",
    page_icon="🔍",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
#MainMenu, header, footer, .stDeployButton { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }
.stApp { background: #0c0c10; }
[data-testid="stFileUploader"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
for key, default in {
    "history": [],
    "last_result": None,
    "ui_phase": "upload",
    "current_filename": None,
    "upload_progress": 0,
    "current_file": None,
    "ready_to_analyze": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─── ANALYSIS FUNCTIONS ────────────────────────────────────────────────────────
def analyze_metadata(image: Image.Image) -> tuple:
    try:
        exif = image.getexif()
        if not exif or len(exif) == 0:
            return 0.45, "No EXIF data (neutral)"
        text = " ".join([str(v).lower() for v in exif.values()])
        ai_tools = ["midjourney", "dalle", "stable diffusion", "runway",
                    "firefly", "ideogram", "leonardo", "novita", "flux"]
        if any(x in text for x in ai_tools):
            return 0.92, "AI generator name found in EXIF"
        edit_tools = ["photoshop", "gimp", "affinity", "capture one", "lightroom"]
        if any(x in text for x in edit_tools):
            return 0.45, "Edited with photo software (inconclusive)"
        cameras = ["canon", "nikon", "sony", "fujifilm", "olympus", "panasonic",
                   "leica", "iphone", "samsung", "pixel", "android", "dji"]
        if any(x in text for x in cameras):
            return 0.12, "Camera make/model present"
        return 0.38, "EXIF present but no camera identifier"
    except Exception:
        return 0.45, "EXIF read error"

def analyze_forensics(image: Image.Image) -> tuple:
    try:
        img_resized = image.resize((256, 256))
        gray = img_resized.convert("L")
        arr = np.array(gray, dtype=np.float32)
        variance = arr.var()
        lap = gray.filter(ImageFilter.FIND_EDGES)
        lap_arr = np.array(lap, dtype=np.float32)
        edge_energy = lap_arr.mean()
        r, g, b = img_resized.split()
        r_arr = np.array(r, dtype=np.float32).flatten()
        g_arr = np.array(g, dtype=np.float32).flatten()
        b_arr = np.array(b, dtype=np.float32).flatten()
        rg_corr = float(np.corrcoef(r_arr, g_arr)[0, 1])
        rb_corr = float(np.corrcoef(r_arr, b_arr)[0, 1])
        avg_corr = (abs(rg_corr) + abs(rb_corr)) / 2

        var_score = 0.25 if (variance < 200 or variance > 3000) else (0.65 if 400 < variance < 1800 else 0.45)
        edge_score = 0.72 if edge_energy < 8 else (0.25 if edge_energy > 25 else 0.50)
        corr_score = 0.70 if avg_corr > 0.92 else (0.25 if avg_corr < 0.70 else 0.45)

        score = (0.4 * var_score) + (0.35 * edge_score) + (0.25 * corr_score)
        note = f"var={variance:.0f}, edge={edge_energy:.1f}, corr={avg_corr:.2f}"
        return round(score, 3), note
    except Exception:
        return 0.45, "Forensics error"

def analyze_filename(filename: str) -> tuple:
    name = filename.lower()
    ai_patterns = ["ai", "dalle", "midjourney", "generated", "flux",
                   "stable", "diffusion", "sdxl", "firefly", "ideogram",
                   "runway", "leonardoai", "novelai"]
    if any(x in name for x in ai_patterns):
        return 0.78, "AI keyword in filename"
    camera_patterns = ["img_", "dsc", "dcim", "p_20", "photo_", "snap", "cam"]
    if any(x in name for x in camera_patterns):
        return 0.15, "Camera-style filename"
    return 0.40, "Neutral filename"

def resize_for_gemini(image: Image.Image, max_px: int = 768) -> Image.Image:
    w, h = image.size
    if max(w, h) <= max_px:
        return image
    ratio = max_px / max(w, h)
    return image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

def detect_with_gemini(image: Image.Image) -> tuple:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        small_img = resize_for_gemini(image, max_px=768)
        prompt = """You are a forensic AI image analyst.

Analyze this image for signs of AI generation vs real photography.

Check:
- Skin/texture smoothness (AI is unnaturally smooth)
- Lighting physics (AI often has inconsistent light sources)
- Background coherence and detail degradation
- Hair/finger/edge sharpness (AI frequently fails here)
- Noise grain (real cameras have natural grain; AI images lack it)
- Facial symmetry (AI tends toward uncanny perfection)

Respond in this EXACT format (no extra text):
SCORE: [0-100]
REASON: [2-3 sentence explanation]

Where 0 = definitely real photo, 100 = definitely AI generated."""

        for attempt in range(3):
            try:
                response = model.generate_content(
                    [prompt, small_img],
                    generation_config=genai.GenerationConfig(
                        max_output_tokens=180,
                        temperature=0.1,
                    )
                )
                text = response.text.strip()
                score_match = re.search(r"SCORE:\s*(\d+)", text)
                reason_match = re.search(r"REASON:\s*(.+)", text, re.DOTALL)
                if score_match:
                    score = float(score_match.group(1)) / 100
                    score = max(0.0, min(1.0, score))
                    reason = reason_match.group(1).strip() if reason_match else "Analysis complete."
                    return score, reason
            except Exception as e:
                if "429" in str(e):
                    time.sleep(4 * (attempt + 1))
                else:
                    break
        return 0.5, "Gemini analysis unavailable."
    except Exception:
        return 0.5, "Gemini error."

def detect(image: Image.Image, filename: str) -> dict:
    gemini_score, gemini_reason   = detect_with_gemini(image)
    meta_score,   meta_note       = analyze_metadata(image)
    forensic_score, forensic_note = analyze_forensics(image)
    fname_score,  fname_note      = analyze_filename(filename)

    gemini_clipped = max(0.08, min(0.92, gemini_score))
    weights = {"gemini": 0.60, "metadata": 0.18, "forensics": 0.16, "filename": 0.06}
    base_score = (
        weights["gemini"]    * gemini_clipped +
        weights["metadata"]  * meta_score +
        weights["forensics"] * forensic_score +
        weights["filename"]  * fname_score
    )

    reliable_scores = [gemini_clipped, meta_score, forensic_score]
    spread = max(reliable_scores) - min(reliable_scores)
    if spread > 0.55:
        pull = (spread - 0.55) * 0.6
        base_score = base_score * (1 - pull) + 0.5 * pull

    final = round(max(0.0, min(1.0, base_score)), 3)
    return {
        "score":          final,
        "gemini_score":   gemini_clipped,
        "meta_score":     meta_score,
        "forensic_score": forensic_score,
        "fname_score":    fname_score,
        "reason":         gemini_reason,
        "meta_note":      meta_note,
        "forensic_note":  forensic_note,
        "fname_note":     fname_note,
        "spread":         round(spread, 3),
    }

def classify(score: float) -> str:
    if score >= 0.75:   return "AI Generated"
    elif score <= 0.35: return "Likely Real"
    else:               return "Suspicious"

def confidence_label(score: float) -> str:
    dist = abs(score - 0.5)
    if dist >= 0.35:   return "High Confidence"
    elif dist >= 0.18: return "Medium Confidence"
    else:              return "Low Confidence"

# ─── STATE MACHINE LOGIC ───────────────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "upload", type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed", key="img_upload"
)

# Step 1: file just picked → switch to "uploading" to show upload animation, store file bytes immediately
if uploaded_file is not None and st.session_state.ui_phase == "upload":
    st.session_state.current_filename = uploaded_file.name
    st.session_state.current_file_bytes = uploaded_file.read()  # read now before rerun clears it
    st.session_state.ui_phase = "uploading"
    st.session_state.ready_to_analyze = False
    st.rerun()

# Step 2: show uploading animation for one render, then flip to analyzing
if st.session_state.ui_phase == "uploading":
    st.session_state.ui_phase = "analyzing"
    st.rerun()

# Step 3: show analyzing animation for one render (ready_to_analyze=False), then do work
if st.session_state.ui_phase == "analyzing":
    if not st.session_state.ready_to_analyze:
        # First pass: just render the animation, schedule work for next rerun
        st.session_state.ready_to_analyze = True
        st.rerun()
    else:
        # Second pass: do the actual analysis
        image = Image.open(io.BytesIO(st.session_state.current_file_bytes)).convert("RGB")
        result = detect(image, st.session_state.current_filename)

        label = classify(result["score"])
        conf = confidence_label(result["score"])

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "File": st.session_state.current_filename,
            "Score": f"{round(result['score'] * 100)}%",
            "Result": label,
        })
        st.session_state.last_result = result
        st.session_state.last_image = image
        st.session_state.ui_phase = "results"
        st.session_state.ready_to_analyze = False
        st.rerun()

# ─── GENERATE PHASE DATA ────────────────────────────────────────────────────────
phase_data = {"phase": st.session_state.ui_phase}

if st.session_state.ui_phase == "uploading":
    phase_data.update({
        "filename": st.session_state.current_filename,
        "progress": 80,
        "filesize": len(st.session_state.get("current_file_bytes", b"")),
    })
elif st.session_state.ui_phase == "analyzing":
    phase_data["progress"] = 0
if st.session_state.ui_phase == "results":
    result = st.session_state.last_result
    image = st.session_state.get("last_image")
    if image is None:
        image = Image.new("RGB", (100, 100), color=(20, 20, 30))

    buf = io.BytesIO()
    thumb = image.copy()
    thumb.thumbnail((640, 640))
    thumb.save(buf, format="JPEG", quality=82)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    phase_data.update({
        "score": result["score"],
        "gemini_score": result["gemini_score"],
        "meta_score": result["meta_score"],
        "forensic_score": result["forensic_score"],
        "fname_score": result["fname_score"],
        "label": classify(result["score"]),
        "conf": confidence_label(result["score"]),
        "reason": result["reason"],
        "meta_note": result["meta_note"],
        "forensic_note": result["forensic_note"],
        "fname_note": result["fname_note"],
        "spread": result["spread"],
        "filename": st.session_state.current_filename,
        "img_b64": img_b64,
        "history": st.session_state.history[-8:],
    })

result_json = json.dumps(phase_data)

# ─── HTML TEMPLATE ─────────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0c0c10;--surface:#13131a;--surface2:#1a1a24;
  --border:#2a2a3a;--accent:#c8a96e;--accent2:#7e6baa;
  --text:#e8e6f0;--muted:#7a7890;
  --real:#4ecb8d;--ai:#e05c5c;--sus:#d4a84b;
}
html,body{background:var(--bg);color:var(--text);font-family:'Outfit',sans-serif;min-height:100vh}
.wrap{max-width:860px;margin:0 auto;padding:2.5rem 1.5rem}

.header{text-align:center;margin-bottom:2.5rem}
.logo{font-family:'DM Serif Display',serif;font-size:2.6rem;color:var(--accent);letter-spacing:0.5px}
.logo span{color:var(--accent2)}
.logo-tag{font-size:1rem;color:var(--muted);font-family:'DM Mono',monospace;vertical-align:middle;margin-left:4px}
.tagline{font-size:0.72rem;letter-spacing:0.22em;text-transform:uppercase;color:var(--muted);margin-top:0.45rem;font-family:'DM Mono',monospace}

.drop-zone{
  border:1px dashed #3a3a50;border-radius:18px;
  padding:3.5rem 2rem;text-align:center;cursor:pointer;
  transition:all 0.25s ease;background:var(--surface);
}
.drop-zone:hover,.drop-zone.drag{border-color:var(--accent);background:var(--surface2)}
.drop-icon{width:56px;height:56px;margin:0 auto 1.1rem;opacity:0.55}
.drop-title{font-size:1rem;font-weight:500;margin-bottom:0.3rem}
.drop-sub{font-size:0.78rem;color:var(--muted)}

.upload-progress-wrap{display:none;border-radius:18px;border:1px solid var(--border);background:var(--surface);padding:2.5rem 2rem;text-align:center;}
.upload-progress-wrap.active{display:block}
.up-icon{width:52px;height:52px;margin:0 auto 1rem;position:relative}
.up-icon svg{width:52px;height:52px}
.up-arrow-anim{animation:upFloat 0.9s ease-in-out infinite}
@keyframes upFloat{0%{transform:translateY(4px);opacity:0.4}50%{transform:translateY(-4px);opacity:1}100%{transform:translateY(4px);opacity:0.4}}
.up-title{font-size:0.95rem;font-weight:500;margin-bottom:0.25rem;color:var(--text)}
.up-sub{font-size:0.75rem;color:var(--muted);font-family:'DM Mono',monospace;margin-bottom:1.4rem}
.up-bar-wrap{height:4px;background:var(--surface2);border-radius:2px;overflow:hidden;position:relative;margin:0 auto;max-width:340px}
.up-bar-shimmer{position:absolute;inset:0;background:linear-gradient(90deg,transparent 0%,var(--accent) 40%,var(--accent2) 60%,transparent 100%);background-size:200% 100%;animation:shimmer 1.4s linear infinite;border-radius:2px}
@keyframes shimmer{0%{background-position:200% center}100%{background-position:-200% center}}
.up-bar-fill{height:100%;border-radius:2px;transition:width 0.35s ease;width:0%;background:linear-gradient(90deg,var(--accent),var(--accent2));}
.up-pct{font-size:0.7rem;color:var(--muted);font-family:'DM Mono',monospace;margin-top:0.5rem}

.analysis-wrap{display:none;border-radius:18px;border:1px solid var(--border);background:var(--surface);padding:2rem 2rem;text-align:center;}
.analysis-wrap.active{display:block}
.scan-orb{position:relative;width:110px;height:110px;margin:0 auto 1.5rem}
.orb-ring{position:absolute;inset:0;border-radius:50%;border:1px solid var(--accent);opacity:0;animation:orbPulse 2s ease-out infinite;}
.orb-ring:nth-child(2){animation-delay:0.7s;border-color:var(--accent2)}
.orb-ring:nth-child(3){animation-delay:1.4s;border-color:var(--accent)}
@keyframes orbPulse{0%{transform:scale(0.6);opacity:0.8}100%{transform:scale(1.6);opacity:0}}
.orb-core{position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);width:52px;height:52px;border-radius:50%;background:radial-gradient(circle at 35% 35%,#2a2040,#13131a);border:1px solid var(--accent2);display:flex;align-items:center;justify-content:center;}
.orb-core svg{width:24px;height:24px;animation:rotateSlow 4s linear infinite}
@keyframes rotateSlow{to{transform:rotate(360deg)}}
.scan-line-box{position:relative;width:100%;max-width:380px;height:3px;margin:0 auto 1.8rem;background:var(--surface2);border-radius:2px;overflow:hidden;}
.scan-line{position:absolute;height:100%;width:40%;border-radius:2px;background:linear-gradient(90deg,transparent,var(--accent),var(--accent2),transparent);animation:scanSlide 1.8s ease-in-out infinite;}
@keyframes scanSlide{0%{left:-40%}100%{left:140%}}
.analysis-title{font-size:1rem;font-weight:500;margin-bottom:0.3rem;color:var(--text)}
.analysis-sub{font-size:0.74rem;color:var(--muted);font-family:'DM Mono',monospace;margin-bottom:1.6rem;min-height:1.4em;transition:opacity 0.4s}
.steps-list{list-style:none;text-align:left;max-width:320px;margin:0 auto;display:flex;flex-direction:column;gap:0.55rem}
.step-item{display:flex;align-items:center;gap:0.75rem;font-size:0.8rem;font-family:'DM Mono',monospace;color:var(--muted);padding:0.45rem 0.75rem;border-radius:8px;transition:all 0.4s ease;}
.step-item.done{color:var(--real)}
.step-item.active{color:var(--accent);background:rgba(200,169,110,0.08);border:1px solid rgba(200,169,110,0.18)}
.step-item.pending{color:var(--muted)}
.step-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;background:currentColor}
.step-item.active .step-dot{animation:dotPulse 0.8s ease-in-out infinite}
@keyframes dotPulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:0.4;transform:scale(0.6)}}
.step-check{font-size:0.85rem;flex-shrink:0}

.main-grid{display:none;grid-template-columns:1fr 1.35fr;gap:1.5rem;margin-top:1.5rem;align-items:start}
.img-frame{border-radius:14px;overflow:hidden;border:1px solid var(--border);background:var(--surface)}
.img-frame img{width:100%;max-height:260px;object-fit:cover;display:block}
.img-meta{font-family:'DM Mono',monospace;font-size:0.7rem;color:var(--muted);padding:0.5rem 0.85rem;border-top:1px solid var(--border)}
.scan-btn{width:100%;margin-top:1.2rem;padding:0.9rem;background:linear-gradient(135deg,#c8a96e 0%,#7e6baa 100%);border:none;border-radius:11px;color:#fff;font-family:'Outfit',sans-serif;font-size:0.95rem;font-weight:600;cursor:pointer;letter-spacing:0.04em;transition:opacity 0.2s;display:none;}
.scan-btn:hover{opacity:0.87}

.result-card{background:var(--surface);border:1px solid var(--border);border-radius:16px;padding:1.6rem;display:none}
.verdict-row{display:flex;align-items:center;justify-content:space-between;margin-bottom:1.2rem}
.verdict-badge{font-family:'DM Mono',monospace;font-size:0.68rem;letter-spacing:0.13em;text-transform:uppercase;padding:0.3rem 0.9rem;border-radius:20px;font-weight:500}
.badge-ai{background:rgba(224,92,92,0.15);color:var(--ai);border:1px solid rgba(224,92,92,0.35)}
.badge-real{background:rgba(78,203,141,0.15);color:var(--real);border:1px solid rgba(78,203,141,0.35)}
.badge-sus{background:rgba(212,168,75,0.15);color:var(--sus);border:1px solid rgba(212,168,75,0.35)}
.conf-tag{font-size:0.7rem;color:var(--muted);font-family:'DM Mono',monospace}
.prob-num{font-family:'DM Serif Display',serif;font-size:3.4rem;line-height:1;color:var(--text);margin-bottom:0.2rem}
.prob-label{font-size:0.72rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.14em;font-family:'DM Mono',monospace}
.gauge-bar{height:5px;border-radius:3px;background:var(--surface2);margin:1rem 0;overflow:hidden;position:relative}
.gauge-track{position:absolute;inset:0;background:linear-gradient(90deg,var(--real) 0%,var(--sus) 50%,var(--ai) 100%);opacity:0.2;border-radius:3px}
.gauge-fill{height:100%;border-radius:3px;transition:width 1s cubic-bezier(.4,0,.2,1);background:linear-gradient(90deg,var(--real),var(--sus),var(--ai));background-size:860px 100%;position:relative}
.signals{display:grid;grid-template-columns:1fr 1fr;gap:0.7rem;margin-top:1.1rem}
.sig-card{background:var(--surface2);border-radius:10px;padding:0.8rem 0.95rem}
.sig-name{font-size:0.68rem;color:var(--muted);font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:0.09em;margin-bottom:0.45rem}
.sig-bar-wrap{height:3px;background:var(--border);border-radius:2px;margin-bottom:0.4rem;overflow:hidden}
.sig-bar-fill{height:100%;border-radius:2px;transition:width 1.1s ease}
.sig-score{font-size:0.78rem;font-weight:500;font-family:'DM Mono',monospace}
.reason-box{margin-top:1.1rem;padding:1rem;background:var(--surface2);border-radius:10px;border-left:3px solid var(--accent2);}
.reason-title{font-size:0.68rem;color:var(--accent2);text-transform:uppercase;letter-spacing:0.13em;font-family:'DM Mono',monospace;margin-bottom:0.45rem}
.reason-text{font-size:0.83rem;color:var(--muted);line-height:1.7}
.tech-toggle{font-size:0.72rem;color:var(--muted);font-family:'DM Mono',monospace;cursor:pointer;margin-top:0.9rem;display:inline-block;text-decoration:underline;text-underline-offset:3px;background:none;border:none;}
.tech-notes{display:none;margin-top:0.6rem;font-size:0.75rem;color:var(--muted);font-family:'DM Mono',monospace;line-height:1.9;padding:0.75rem;background:var(--bg);border-radius:8px}
.divider{height:1px;background:var(--border);margin:2.2rem 0}
.history-wrap{display:none}
.history-title{font-size:0.7rem;color:var(--muted);text-transform:uppercase;letter-spacing:0.16em;font-family:'DM Mono',monospace;margin-bottom:0.8rem}
.history-row{display:flex;align-items:center;gap:0.75rem;padding:0.6rem 0;border-bottom:1px solid var(--border);font-size:0.8rem}
.history-row:last-child{border-bottom:none}
.history-file{flex:1;color:var(--text);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.history-score{font-family:'DM Mono',monospace;font-size:0.72rem;color:var(--muted)}
.dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.dot-ai{background:var(--ai)}.dot-real{background:var(--real)}.dot-sus{background:var(--sus)}
.csv-btn{margin-top:1rem;padding:0.5rem 1.1rem;border:1px solid var(--border);border-radius:8px;background:transparent;color:var(--muted);font-family:'DM Mono',monospace;font-size:0.72rem;cursor:pointer;transition:all 0.2s;}
.csv-btn:hover{border-color:var(--accent);color:var(--accent)}
#dataContainer{display:none;}
@media(max-width:640px){.main-grid{grid-template-columns:1fr}.signals{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div class="logo">Lumina<span>Check</span><span class="logo-tag">AI</span></div>
    <div class="tagline">Forensic Image Authentication &nbsp;·&nbsp; Gemini 2.5 Flash + Hybrid Signals</div>
  </div>

  <!-- DROP ZONE -->
  <div class="drop-zone" id="dropZone">
    <div class="drop-icon">
      <svg viewBox="0 0 56 56" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="6" y="12" width="44" height="34" rx="5" stroke="#c8a96e" stroke-width="1.4"/>
        <circle cx="20" cy="23" r="4.5" stroke="#7e6baa" stroke-width="1.4"/>
        <path d="M6 38l13-11 9 8 7-6 15 15" stroke="#c8a96e" stroke-width="1.4" stroke-linejoin="round"/>
        <path d="M36 4v12M31 9l5-5 5 5" stroke="#c8a96e" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <div class="drop-title">Drop an image here to analyze</div>
    <div class="drop-sub">or click to browse &nbsp;·&nbsp; JPG &nbsp;PNG &nbsp;WEBP</div>
  </div>

  <!-- UPLOAD PROGRESS -->
  <div class="upload-progress-wrap" id="uploadProgressWrap">
    <div class="up-icon">
      <svg viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="4" y="14" width="44" height="32" rx="5" stroke="#c8a96e" stroke-width="1.4"/>
        <path class="up-arrow-anim" d="M26 28V8M20 14l6-6 6 6" stroke="#c8a96e" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </div>
    <div class="up-title">Uploading image…</div>
    <div class="up-sub" id="upSubText">Reading file data</div>
    <div class="up-bar-wrap">
      <div class="up-bar-shimmer"></div>
      <div class="up-bar-fill" id="upBarFill"></div>
    </div>
    <div class="up-pct" id="upPct">0%</div>
  </div>

  <!-- ANALYSIS SCREEN -->
  <div class="analysis-wrap" id="analysisWrap">
    <div class="scan-orb">
      <div class="orb-ring"></div>
      <div class="orb-ring"></div>
      <div class="orb-ring"></div>
      <div class="orb-core">
        <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="12" cy="12" r="9" stroke="#c8a96e" stroke-width="1.2" stroke-dasharray="3 3"/>
          <circle cx="12" cy="12" r="4" stroke="#7e6baa" stroke-width="1.2"/>
          <circle cx="12" cy="12" r="1.5" fill="#c8a96e"/>
        </svg>
      </div>
    </div>
    <div class="scan-line-box"><div class="scan-line"></div></div>
    <div class="analysis-title">Analyzing image…</div>
    <div class="analysis-sub" id="analysisSub">Running forensic checks</div>
    <ul class="steps-list" id="stepsList">
      <li class="step-item active" id="step0"><span class="step-dot"></span>Gemini vision analysis</li>
      <li class="step-item pending" id="step1"><span class="step-dot"></span>EXIF metadata scan</li>
      <li class="step-item pending" id="step2"><span class="step-dot"></span>Pixel forensics</li>
      <li class="step-item pending" id="step3"><span class="step-dot"></span>Signal fusion</li>
    </ul>
  </div>

  <!-- RESULTS GRID -->
  <div class="main-grid" id="mainGrid">
    <div>
      <div class="img-frame">
        <img id="resultImg" src="" alt="Analyzed image"/>
        <div class="img-meta" id="imgMeta">—</div>
      </div>
      <button class="scan-btn" id="scanAnotherBtn" onclick="resetToUpload()">Scan another image</button>
    </div>
    <div class="result-card" id="resultCard">
      <div class="verdict-row">
        <span class="verdict-badge" id="verdictBadge">—</span>
        <span class="conf-tag" id="confTag">—</span>
      </div>
      <div class="prob-num" id="probNum">—%</div>
      <div class="prob-label">AI probability score</div>
      <div class="gauge-bar">
        <div class="gauge-track"></div>
        <div class="gauge-fill" id="gaugeFill" style="width:0%"></div>
      </div>
      <div class="signals" id="signalsGrid"></div>
      <div class="reason-box" id="reasonBox">
        <div class="reason-title">Gemini analysis</div>
        <div class="reason-text" id="reasonText">—</div>
      </div>
      <button class="tech-toggle" onclick="toggleTech()">show technical details</button>
      <div class="tech-notes" id="techNotes"></div>
    </div>
  </div>

  <div class="divider" id="histDivider" style="display:none"></div>
  <div class="history-wrap" id="historyWrap">
    <div class="history-title">Scan history</div>
    <div id="historyList"></div>
    <button class="csv-btn" onclick="exportCSV()">Export CSV</button>
  </div>

  <div id="dataContainer">PHASE_DATA_PLACEHOLDER</div>
</div>

<script>
(function(){
  const raw = document.getElementById('dataContainer').textContent.trim();
  let data;
  try { data = JSON.parse(raw); } catch(e) { return; }

  const phase = data.phase;

  const dropZone      = document.getElementById('dropZone');
  const uploadWrap    = document.getElementById('uploadProgressWrap');
  const analysisWrap  = document.getElementById('analysisWrap');
  const mainGrid      = document.getElementById('mainGrid');
  const resultCard    = document.getElementById('resultCard');
  const scanBtn       = document.getElementById('scanAnotherBtn');
  const histDivider   = document.getElementById('histDivider');
  const historyWrap   = document.getElementById('historyWrap');

  dropZone.addEventListener('click', () => {
    const fu = window.parent.document.querySelector('[data-testid="stFileUploader"] input[type="file"]');
    if (fu) fu.click();
  });
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag');
    const fu = window.parent.document.querySelector('[data-testid="stFileUploader"] input[type="file"]');
    if (fu && e.dataTransfer.files.length) {
      const dt = new DataTransfer();
      dt.items.add(e.dataTransfer.files[0]);
      fu.files = dt.files;
      fu.dispatchEvent(new Event('change', { bubbles: true }));
    }
  });

  if (phase === 'upload') {
    dropZone.style.display = 'block';
  }

  // FIX: "uploading" phase removed; this block kept for safety but will never trigger
  else if (phase === 'uploading') {
    dropZone.style.display = 'none';
    uploadWrap.classList.add('active');
    const pct = data.progress || 0;
    document.getElementById('upBarFill').style.width = pct + '%';
    document.getElementById('upPct').textContent = pct + '%';
    const kb = Math.round((data.filesize || 0) / 1024);
    document.getElementById('upSubText').textContent =
      data.filename ? data.filename + (kb ? ' · ' + kb + ' KB' : '') : 'Reading file data';
  }

  else if (phase === 'analyzing' || phase === 'results') {
    dropZone.style.display = 'none';
    analysisWrap.classList.add('active');

    // ── Cinematic 5-step animation (always plays, regardless of phase) ──
    const TOTAL_MS = 5000;
    const steps    = ['step0','step1','step2','step3'];
    const subs     = ['Running Gemini vision model…','Scanning EXIF metadata…','Running pixel forensics…','Fusing all signals…'];
    const subEl    = document.getElementById('analysisSub');

    // Animated progress bar across full 5 s
    const progressBar = document.createElement('div');
    progressBar.style.cssText = 'height:3px;border-radius:2px;background:var(--surface2);overflow:hidden;position:relative;max-width:380px;margin:1.2rem auto 0';
    const progressFill = document.createElement('div');
    progressFill.style.cssText = 'height:100%;border-radius:2px;background:linear-gradient(90deg,var(--accent),var(--accent2));width:0%;transition:width 0.25s linear';
    progressBar.appendChild(progressFill);
    analysisWrap.appendChild(progressBar);

    // Particle burst overlay
    const canvas = document.createElement('canvas');
    canvas.style.cssText = 'position:fixed;inset:0;pointer-events:none;z-index:10;opacity:0;transition:opacity 0.5s';
    canvas.width = window.innerWidth; canvas.height = window.innerHeight;
    document.body.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    let particles = [];
    function spawnParticles() {
      const cx = canvas.width/2, cy = canvas.height/2;
      for (let i=0;i<60;i++) {
        const angle = Math.random()*Math.PI*2;
        const speed = 2 + Math.random()*5;
        const colors = ['#c8a96e','#7e6baa','#4ecb8d','#e05c5c','#e8e6f0'];
        particles.push({ x:cx, y:cy, vx:Math.cos(angle)*speed, vy:Math.sin(angle)*speed,
          life:1, decay:0.012+Math.random()*0.015, r:2+Math.random()*3,
          color:colors[Math.floor(Math.random()*colors.length)] });
      }
    }
    function animParticles() {
      ctx.clearRect(0,0,canvas.width,canvas.height);
      particles = particles.filter(p=>p.life>0);
      particles.forEach(p => {
        p.x+=p.vx; p.y+=p.vy; p.vy+=0.08; p.life-=p.decay;
        ctx.globalAlpha = p.life;
        ctx.beginPath(); ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
        ctx.fillStyle = p.color; ctx.fill();
      });
      if (particles.length) requestAnimationFrame(animParticles);
      else ctx.clearRect(0,0,canvas.width,canvas.height);
    }

    // Countdown ticker shown in sub text
    let cur = 0;
    const stepInterval = TOTAL_MS / steps.length; // 1250ms each

    function markStep(i, done) {
      const el = document.getElementById(steps[i]);
      if (done) {
        el.className = 'step-item done';
        el.innerHTML = '<span class="step-check">✓</span>' + el.innerText;
      } else {
        el.className = 'step-item active';
        subEl.textContent = subs[i];
      }
    }

    // Tick progress bar every 100ms
    const startTime = Date.now();
    const tickInterval = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const pct = Math.min(100, (elapsed / TOTAL_MS) * 100);
      progressFill.style.width = pct + '%';

      // Advance steps at even intervals
      const newStep = Math.min(steps.length - 1, Math.floor(elapsed / stepInterval));
      if (newStep > cur) {
        markStep(cur, true);
        cur = newStep;
        markStep(cur, false);
      }
    }, 100);

    markStep(0, false);

    // At 4.2 s — flash particles so they finish by 5 s
    setTimeout(() => {
      canvas.style.opacity = '1';
      spawnParticles(); animParticles();
    }, 4200);

    // At 5 s — reveal results
    setTimeout(() => {
      clearInterval(tickInterval);
      canvas.style.opacity = '0';
      // Mark all steps done
      steps.forEach((_, i) => markStep(i, true));
      progressFill.style.width = '100%';
      subEl.textContent = 'Analysis complete ✓';

      // Brief flash then crossfade to results
      setTimeout(() => {
        analysisWrap.style.transition = 'opacity 0.6s ease';
        analysisWrap.style.opacity = '0';
        setTimeout(() => {
          analysisWrap.style.display = 'none';
          showResults();
        }, 600);
      }, 350);
    }, TOTAL_MS);
  }

  function showResults() {
    if (!data.score && data.score !== 0) return; // no result data yet (pure analyzing phase)

    mainGrid.style.display = 'grid';
    mainGrid.style.opacity = '0';
    mainGrid.style.transition = 'opacity 0.7s ease';
    resultCard.style.display = 'block';
    scanBtn.style.display = 'block';

    const score = data.score;
    const pct   = Math.round(score * 100);
    const label = data.label;

    if (data.img_b64) {
      document.getElementById('resultImg').src = 'data:image/jpeg;base64,' + data.img_b64;
    }
    document.getElementById('imgMeta').textContent = data.filename || '—';

    const badge = document.getElementById('verdictBadge');
    badge.textContent = label;
    badge.className = 'verdict-badge ' +
      (label === 'AI Generated' ? 'badge-ai' : label === 'Likely Real' ? 'badge-real' : 'badge-sus');

    document.getElementById('confTag').textContent = data.conf;
    document.getElementById('probNum').textContent = '—%';

    // Fade grid in
    requestAnimationFrame(() => { mainGrid.style.opacity = '1'; });

    // Animate score counter after fade
    setTimeout(() => {
      let n = 0;
      const target = pct;
      const duration = 1200;
      const step = target / (duration / 16);
      const el = document.getElementById('probNum');
      const counter = setInterval(() => {
        n = Math.min(target, n + step);
        el.textContent = Math.round(n) + '%';
        if (n >= target) clearInterval(counter);
      }, 16);
    }, 400);

    setTimeout(() => { document.getElementById('gaugeFill').style.width = pct + '%'; }, 500);

    const signals = [
      { name: 'Gemini Vision',  score: data.gemini_score },
      { name: 'EXIF Metadata',  score: data.meta_score },
      { name: 'Pixel Forensics',score: data.forensic_score },
      { name: 'Filename',       score: data.fname_score },
    ];
    const grid = document.getElementById('signalsGrid');
    grid.innerHTML = signals.map(s => {
      const sp  = Math.round(s.score * 100);
      const col = s.score >= 0.65 ? 'var(--ai)' : s.score <= 0.40 ? 'var(--real)' : 'var(--sus)';
      return `<div class="sig-card">
        <div class="sig-name">${s.name}</div>
        <div class="sig-bar-wrap"><div class="sig-bar-fill" id="sb_${s.name.replace(/ /g,'')}" style="width:0%;background:${col}"></div></div>
        <div class="sig-score" style="color:${col}">${sp}%</div>
      </div>`;
    }).join('');
    // Animate signal bars staggered
    signals.forEach((s, i) => {
      setTimeout(() => {
        const el = document.getElementById('sb_' + s.name.replace(/ /g,''));
        if (el) el.style.width = Math.round(s.score * 100) + '%';
      }, 600 + i * 150);
    });

    if (data.reason) document.getElementById('reasonText').textContent = data.reason;

    document.getElementById('techNotes').innerHTML =
      `meta: ${data.meta_note}<br>forensics: ${data.forensic_note}<br>filename: ${data.fname_note}<br>signal spread: ${data.spread}`;

    if (data.history && data.history.length > 0) {
      histDivider.style.display = 'block';
      historyWrap.style.display = 'block';
      const list = document.getElementById('historyList');
      list.innerHTML = data.history.slice().reverse().map(h => {
        const dotCls = h.Result === 'AI Generated' ? 'dot-ai' : h.Result === 'Likely Real' ? 'dot-real' : 'dot-sus';
        return `<div class="history-row">
          <span class="dot ${dotCls}"></span>
          <span class="history-file">${h.File}</span>
          <span class="history-score">${h.Score} · ${h.Result}</span>
          <span class="history-score">${h.Time}</span>
        </div>`;
      }).join('');
    }
  }

  window.toggleTech = function() {
    const el = document.getElementById('techNotes');
    const btn = document.querySelector('.tech-toggle');
    const visible = el.style.display === 'block';
    el.style.display = visible ? 'none' : 'block';
    btn.textContent = visible ? 'show technical details' : 'hide technical details';
  };

  window.exportCSV = function() {
    if (!data.history) return;
    const rows = [['Time','File','Score','Result'], ...data.history.map(h => [h.Time, h.File, h.Score, h.Result])];
    const csv = rows.map(r => r.join(',')).join('\\n');
    const a = document.createElement('a');
    a.href = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv);
    a.download = 'luminacheck_history.csv';
    a.click();
  };

  window.resetToUpload = function() {
    const fu = window.parent.document.querySelector('[data-testid="stFileUploader"] input[type="file"]');
    if (fu) fu.click();
  };
})();
</script>
</body>
</html>
""".replace("PHASE_DATA_PLACEHOLDER", result_json)

components.html(HTML, height=900, scrolling=True)
