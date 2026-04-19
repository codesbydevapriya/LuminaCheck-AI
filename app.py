import numpy as np
from PIL import Image
import io

# Optional: safe cv2 import
try:
    import cv2
    CV2_AVAILABLE = True
except:
    CV2_AVAILABLE = False


# ---------------- NEW DETECTORS ----------------

def analyze_recompression(image):
    try:
        g0 = np.array(image.convert("L").resize((256,256)))

        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=70)
        buf.seek(0)

        rec = Image.open(buf).convert("L").resize((256,256))
        g1 = np.array(rec)

        diff = np.mean(np.abs(g0 - g1))

        if diff > 8:
            return 0.7, "Recompression unstable"
        elif diff < 3:
            return 0.3, "Stable compression"
        else:
            return 0.5, "Neutral"

    except:
        return 0.5, "Recompression error"


def analyze_patch_consistency(image):
    try:
        img = np.array(image.convert("L").resize((256,256)))  # ✅ FIXED

        patches = []
        step = 32

        for i in range(0, 256-step, step):
            for j in range(0, 256-step, step):
                patches.append(img[i:i+step, j:j+step].flatten())

        if len(patches) < 2:
            return 0.5, "Insufficient patches"

        sims = []
        for i in range(len(patches)-1):
            c = np.corrcoef(patches[i], patches[i+1])[0,1]
            if not np.isnan(c):
                sims.append(c)

        if not sims:
            return 0.5, "Invalid correlation"

        avg_sim = np.mean(sims)

        if avg_sim > 0.92:
            return 0.75, "Repeating texture"
        elif avg_sim < 0.7:
            return 0.3, "Natural variation"
        else:
            return 0.5, "Neutral"

    except:
        return 0.5, "Patch error"


def analyze_edges(image):
    try:
        if not CV2_AVAILABLE:
            return 0.5, "cv2 not available"

        img = np.array(image.convert("L").resize((256,256)))

        edges = cv2.Canny(img, 50, 150)
        gy, gx = np.gradient(edges.astype(float))
        randomness = np.std(gx) + np.std(gy)

        if randomness < 5:
            return 0.7, "Structured edges"
        elif randomness > 12:
            return 0.3, "Natural edges"
        else:
            return 0.5, "Neutral"

    except:
        return 0.5, "Edge error"


# ---------------- UPDATED DETECT ----------------

def detect(image: Image.Image, filename: str) -> dict:
    gemini_score, gemini_reason = detect_with_gemini(image)
    meta_score, meta_note = analyze_metadata(image)
    forensic_score, forensic_note = analyze_forensics(image)
    fname_score, fname_note = analyze_filename(filename)

    # NEW SIGNALS
    recomp_score, _ = analyze_recompression(image)
    patch_score, _ = analyze_patch_consistency(image)
    edge_score, _ = analyze_edges(image)

    g = max(0.1, min(0.95, gemini_score))

    base = (
        0.55 * g +
        0.10 * meta_score +
        0.08 * forensic_score +
        0.05 * fname_score +
        0.07 * recomp_score +
        0.08 * patch_score +
        0.07 * edge_score
    )

    # AGGRESSIVE BOOSTS
    if patch_score > 0.7:
        base += 0.08

    if recomp_score > 0.7:
        base += 0.07

    if edge_score > 0.65:
        base += 0.05

    if 0.3 < g < 0.6:
        base += 0.12

    final = max(0.0, min(1.0, base))

    return {
        "score": round(final, 3),
        "gemini_score": g,
        "meta_score": meta_score,
        "forensic_score": forensic_score,
        "fname_score": fname_score,
        "recomp_score": recomp_score,
        "patch_score": patch_score,
        "edge_score": edge_score,
        "reason": gemini_reason,
        "meta_note": meta_note,
        "forensic_note": forensic_note,
        "fname_note": fname_note,
    }


# ---------------- CLASSIFICATION ----------------

def classify(score: float) -> str:
    if score >= 0.60:
        return "AI Generated"
    elif score <= 0.20:
        return "Likely Real"
    else:
        return "Suspicious"
