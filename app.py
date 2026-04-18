import os
import streamlit as st
from PIL import Image, ImageFilter
import pandas as pd
from datetime import datetime
import google.generativeai as genai
import re
import time
import numpy as np
import io

# ─── API KEY ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

st.set_page_config(page_title="LuminaCheck AI", layout="wide", page_icon="🔍")

# ─── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

.stApp { background: #0a0a0f; }

h1 { font-family: 'Syne', sans-serif; font-weight: 800; letter-spacing: -1px; }
h2, h3 { font-family: 'Syne', sans-serif; font-weight: 700; }

.stProgress > div > div {
    background: linear-gradient(90deg, #00f5a0, #00d9f5);
    border-radius: 4px;
}

.metric-box {
    background: #13131f;
    border: 1px solid #2a2a3f;
    border-radius: 12px;
    padding: 16px 20px;
    margin: 8px 0;
    font-family: 'Space Mono', monospace;
    font-size: 13px;
}

.signal-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 6px 0;
}

.signal-fill {
    height: 8px;
    border-radius: 4px;
    background: linear-gradient(90deg, #7c3aed, #c026d3);
    transition: width 0.4s ease;
}

.tag-ai    { background:#3f0000; color:#ff6b6b; border:1px solid #ff6b6b; padding:4px 12px; border-radius:6px; font-weight:700; }
.tag-real  { background:#003f10; color:#6bffaa; border:1px solid #6bffaa; padding:4px 12px; border-radius:6px; font-weight:700; }
.tag-sus   { background:#3f2a00; color:#ffc06b; border:1px solid #ffc06b; padding:4px 12px; border-radius:6px; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ─── SESSION STATE ─────────────────────────────────────────────────────────────
for key, default in {
    "history": [],
    "last_result": None,
    "last_label": None,
    "last_reason": None,
    "last_signals": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 1 — METADATA
# ══════════════════════════════════════════════════════════════════════════════
def analyze_metadata(image: Image.Image) -> tuple[float, str]:
    """
    Returns (score 0–1, note).
    Score near 1 = likely AI, near 0 = likely real.
    """
    try:
        exif = image.getexif()

        if not exif or len(exif) == 0:
            # No EXIF at all — neutral, not AI-leaning
            # Many web/WhatsApp images strip EXIF legitimately
            return 0.45, "No EXIF data (neutral)"

        text = " ".join([str(v).lower() for v in exif.values()])

        # Strong AI signals in EXIF
        ai_tools = ["midjourney", "dalle", "stable diffusion", "runway",
                    "firefly", "ideogram", "leonardo", "novita", "flux"]
        if any(x in text for x in ai_tools):
            return 0.92, "AI generator name found in EXIF"

        # Editing software — suspicious but not conclusive
        edit_tools = ["photoshop", "gimp", "affinity", "capture one", "lightroom"]
        if any(x in text for x in edit_tools):
            return 0.45, "Edited with photo software (inconclusive)"

        # Camera hardware = strong real signal
        cameras = ["canon", "nikon", "sony", "fujifilm", "olympus", "panasonic",
                   "leica", "iphone", "samsung", "pixel", "android", "dji"]
        if any(x in text for x in cameras):
            return 0.12, "Camera make/model present"

        # Has EXIF but no camera info — mildly suspicious
        return 0.38, "EXIF present but no camera identifier"

    except Exception:
        return 0.45, "EXIF read error"


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 2 — FORENSICS (improved)
# ══════════════════════════════════════════════════════════════════════════════
def analyze_forensics(image: Image.Image) -> tuple[float, str]:
    """
    Multi-metric forensic analysis:
    - Pixel variance (smoothness)
    - High-frequency noise via Laplacian
    - Color channel correlation (AI images are often too balanced)
    Returns (score 0–1, note).
    """
    try:
        img_resized = image.resize((256, 256))
        gray = img_resized.convert("L")
        arr = np.array(gray, dtype=np.float32)

        # 1. Variance — AI images tend to have intermediate, smooth variance
        variance = arr.var()

        # 2. Laplacian edge energy — real photos have natural sharp/soft mix
        lap = gray.filter(ImageFilter.FIND_EDGES)
        lap_arr = np.array(lap, dtype=np.float32)
        edge_energy = lap_arr.mean()

        # 3. Color channel correlation
        r, g, b = img_resized.split()
        r_arr = np.array(r, dtype=np.float32).flatten()
        g_arr = np.array(g, dtype=np.float32).flatten()
        b_arr = np.array(b, dtype=np.float32).flatten()

        # AI images often have unnaturally high inter-channel correlation
        rg_corr = float(np.corrcoef(r_arr, g_arr)[0, 1])
        rb_corr = float(np.corrcoef(r_arr, b_arr)[0, 1])
        avg_corr = (abs(rg_corr) + abs(rb_corr)) / 2

        # Score components (each 0–1, higher = more AI-like)
        # Variance: very low (<200) or very high (>3000) = real extremes; mid = AI
        if variance < 200 or variance > 3000:
            var_score = 0.25
        elif 400 < variance < 1800:
            var_score = 0.65
        else:
            var_score = 0.45

        # Edge energy: AI often has unnaturally uniform or absent edges
        if edge_energy < 8:
            edge_score = 0.72   # very smooth — AI-like
        elif edge_energy > 25:
            edge_score = 0.25   # high noise — real-like
        else:
            edge_score = 0.50

        # Color correlation: > 0.92 is unnaturally correlated
        if avg_corr > 0.92:
            corr_score = 0.70
        elif avg_corr < 0.70:
            corr_score = 0.25
        else:
            corr_score = 0.45

        score = (0.4 * var_score) + (0.35 * edge_score) + (0.25 * corr_score)
        note = f"var={variance:.0f}, edge={edge_energy:.1f}, corr={avg_corr:.2f}"

        return round(score, 3), note

    except Exception:
        return 0.45, "Forensics error"


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 3 — FILENAME
# ══════════════════════════════════════════════════════════════════════════════
def analyze_filename(filename: str) -> tuple[float, str]:
    name = filename.lower()

    ai_patterns = ["ai", "dalle", "midjourney", "generated", "flux",
                   "stable", "diffusion", "sdxl", "firefly", "ideogram",
                   "runway", "leonardoai", "novelai"]

    if any(x in name for x in ai_patterns):
        return 0.78, "AI keyword in filename"

    # Camera naming patterns: IMG_, DSC_, DCIM, P_20...
    camera_patterns = ["img_", "dsc", "dcim", "p_20", "photo_", "snap", "cam"]
    if any(x in name for x in camera_patterns):
        return 0.15, "Camera-style filename"

    return 0.40, "Neutral filename"


# ══════════════════════════════════════════════════════════════════════════════
#  SIGNAL 4 — GEMINI (token-efficient, reasoning-first)
# ══════════════════════════════════════════════════════════════════════════════
def resize_for_gemini(image: Image.Image, max_px: int = 768) -> Image.Image:
    """Resize image to reduce token cost while preserving aspect ratio."""
    w, h = image.size
    if max(w, h) <= max_px:
        return image
    ratio = max_px / max(w, h)
    return image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)


def detect_with_gemini(image: Image.Image) -> tuple[float, str]:
    """
    Single Gemini call that returns BOTH a score AND brief reasoning.
    This saves one full API call vs. the old two-call approach.
    Returns (score 0–1, reason_text).
    """
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")

        # Resize before sending — reduces tokens significantly
        small_img = resize_for_gemini(image, max_px=768)

        # Chain-of-thought prompt: reason THEN score
        # Combined into one call to halve API usage
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
                        max_output_tokens=180,   # tight limit saves tokens
                        temperature=0.1,          # low temp = consistent scoring
                    )
                )
                text = response.text.strip()

                # Parse SCORE
                score_match = re.search(r"SCORE:\s*(\d+)", text)
                reason_match = re.search(r"REASON:\s*(.+)", text, re.DOTALL)

                if score_match:
                    score = float(score_match.group(1)) / 100
                    score = max(0.0, min(1.0, score))
                    reason = reason_match.group(1).strip() if reason_match else "Analysis complete."
                    return score, reason

            except Exception as e:
                if "429" in str(e):
                    time.sleep(4 * (attempt + 1))   # exponential backoff
                else:
                    break

        return 0.5, "Gemini analysis unavailable."

    except Exception:
        return 0.5, "Gemini error."


# ══════════════════════════════════════════════════════════════════════════════
#  FUSION ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def detect(image: Image.Image, filename: str) -> dict:
    """
    Fuses all signals into a final score.
    Returns a full result dict with all sub-scores and reasoning.
    """
    # Run all signals
    gemini_score, gemini_reason  = detect_with_gemini(image)
    meta_score,   meta_note      = analyze_metadata(image)
    forensic_score, forensic_note = analyze_forensics(image)
    fname_score,  fname_note     = analyze_filename(filename)

    # ── Soft-clip Gemini (avoid extreme anchoring on one signal)
    gemini_clipped = max(0.08, min(0.92, gemini_score))

    # ── Weighted fusion
    # Gemini is most accurate but we reduce its weight slightly vs before
    # to give forensics more voice (now that forensics is better)
    weights = {
        "gemini":    0.60,
        "metadata":  0.18,
        "forensics": 0.16,
        "filename":  0.06,
    }

    base_score = (
        weights["gemini"]    * gemini_clipped +
        weights["metadata"]  * meta_score +
        weights["forensics"] * forensic_score +
        weights["filename"]  * fname_score
    )

    # ── Disagreement handling (only penalize if RELIABLE signals disagree)
    # Filename is excluded from disagreement calc (too weak)
    reliable_scores = [gemini_clipped, meta_score, forensic_score]
    spread = max(reliable_scores) - min(reliable_scores)

    if spread > 0.55:
        # Blend toward neutral — but don't collapse fully to 0.5
        pull = (spread - 0.55) * 0.6   # max ~27% pull toward neutral
        base_score = base_score * (1 - pull) + 0.5 * pull

    # ── Preserve gradient in uncertain zone (no hard collapse to 0.5)
    # Instead of collapsing 0.4–0.6 to 0.5, just leave it as-is
    # The confidence display will communicate uncertainty to the user

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
    if score >= 0.75:
        return "AI Generated"
    elif score <= 0.35:
        return "Likely Real"
    else:
        return "Suspicious / Uncertain"


def confidence_label(score: float) -> str:
    dist = abs(score - 0.5)
    if dist >= 0.35:
        return "High"
    elif dist >= 0.18:
        return "Medium"
    else:
        return "Low"


# ══════════════════════════════════════════════════════════════════════════════
#  UI
# ══════════════════════════════════════════════════════════════════════════════
st.title("🔍 LuminaCheck AI")
st.caption("Hybrid Forensic Detection · Gemini 2.5 Flash + Image Forensics + Metadata")

uploaded_file = st.file_uploader("Upload an image to analyze", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1.3], gap="large")

    with col1:
        st.image(image, use_container_width=True, caption=uploaded_file.name)

    with col2:
        if st.button("🔬 Analyze Image", use_container_width=True):
            with st.spinner("Running forensic analysis…"):
                result = detect(image, uploaded_file.name)
                label  = classify(result["score"])

                st.session_state.last_result  = result
                st.session_state.last_label   = label
                st.session_state.last_reason  = result["reason"]
                st.session_state.last_signals = result

                st.session_state.history.append({
                    "Time":   datetime.now().strftime("%H:%M:%S"),
                    "File":   uploaded_file.name,
                    "Score":  f"{round(result['score'] * 100)}%",
                    "Result": label,
                })

# ─── RESULT PANEL ──────────────────────────────────────────────────────────────
if st.session_state.last_result and isinstance(st.session_state.last_result, dict):
    result = st.session_state.last_result
    label  = st.session_state.last_label
    score  = result["score"]
    pct    = round(score * 100)
    conf   = confidence_label(score)

    st.markdown("---")
    st.markdown("## 📊 Analysis Result")

    # Big verdict
    tag_class = {"AI Generated": "tag-ai", "Likely Real": "tag-real"}.get(label, "tag-sus")
    verdict_icon = {"AI Generated": "🚨", "Likely Real": "✅"}.get(label, "⚠️")
    st.markdown(
        f'<span class="{tag_class}">{verdict_icon} {label}</span>',
        unsafe_allow_html=True
    )
    st.markdown(f"### AI Probability: **{pct}%** &nbsp;·&nbsp; Confidence: **{conf}**")
    st.progress(score)

    # Signal breakdown
    st.markdown("#### Signal Breakdown")

    signals = [
        ("🤖 Gemini Vision",   result["gemini_score"],   "60% weight"),
        ("🗂 Metadata",        result["meta_score"],     "18% weight"),
        ("🔬 Image Forensics", result["forensic_score"], "16% weight"),
        ("📁 Filename",        result["fname_score"],    "6% weight"),
    ]

    for name, sig_score, weight in signals:
        bar_pct = round(sig_score * 100)
        color = "#ff6b6b" if sig_score > 0.65 else ("#6bffaa" if sig_score < 0.35 else "#ffc06b")
        st.markdown(
            f"""<div class="metric-box">
                <div style="display:flex;justify-content:space-between;margin-bottom:6px">
                    <span>{name}</span>
                    <span style="color:{color};font-weight:700">{bar_pct}% AI</span>
                    <span style="opacity:0.45;font-size:11px">{weight}</span>
                </div>
                <div style="background:#1e1e2e;border-radius:4px;height:8px">
                    <div style="width:{bar_pct}%;height:8px;border-radius:4px;background:{color}"></div>
                </div>
            </div>""",
            unsafe_allow_html=True
        )

    # Notes
    with st.expander("📋 Technical Notes"):
        st.markdown(f"- **Metadata:** {result['meta_note']}")
        st.markdown(f"- **Forensics:** {result['forensic_note']}")
        st.markdown(f"- **Filename:** {result['fname_note']}")
        st.markdown(f"- **Signal spread:** {result['spread']} ({'high disagreement' if result['spread'] > 0.5 else 'signals agree'})")

    # Gemini reason
    if result["reason"]:
        st.markdown("#### 🔍 Gemini Analysis")
        st.info(result["reason"])

# ─── HISTORY ───────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("## 🕒 Detection History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ Download CSV Report",
        df.to_csv(index=False),
        "luminacheck_report.csv",
        mime="text/csv"
    )
