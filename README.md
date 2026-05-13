# LuminaCheck AI  
### Where Light Reveals Truth

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=for-the-badge&logo=streamlit)
![Google Gemini](https://img.shields.io/badge/Google_Gemini-2.5_Flash-green?style=for-the-badge&logo=google)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet18-orange?style=for-the-badge&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> LuminaCheck AI is a cinematic AI-powered forensic image analysis platform that detects whether an image is REAL, SUSPICIOUS, or AI-GENERATED using hybrid AI signals, metadata inspection, and pixel-level forensics.

---

# Live Demo

🌐 https://luminacheck-ai.streamlit.app

---

# Features

| Feature | Description |
|---|---|
| AI Image Detection | Detects AI-generated, manipulated, or real images |
| Gemini Vision Analysis | Uses Google Gemini 2.5 Flash for visual reasoning |
| Pixel Forensics | Analyzes texture variance, edge energy, and RGB correlations |
| EXIF Metadata Scan | Detects editing tools and AI generators |
| Filename Intelligence | Flags suspicious AI-related filenames |
| Hybrid Signal Fusion | Combines multiple forensic signals into one score |
| Confidence Meter | High, Medium, or Low confidence verdict |
| Animated UI | Cinematic dark-mode interface with smooth transitions |
| Detection History | Stores recent scan history during session |
| CSV Export | Download scan reports instantly |
| Mobile Responsive | Works on desktop and mobile browsers |

---

# Detection Pipeline

LuminaCheck AI uses a multi-layer forensic system:

1. Upload image
2. Gemini Vision analyzes visual inconsistencies
3. EXIF metadata gets scanned
4. Pixel-level forensic checks run
5. Filename patterns are analyzed
6. Hybrid scoring engine calculates final verdict

---

# Tech Stack

- Python 3.11
- Streamlit
- Google Gemini 2.5 Flash
- PyTorch
- ResNet18
- Pillow (PIL)
- NumPy
- Pandas
- HTML/CSS/JavaScript
- Streamlit Cloud

---

# AI Detection Signals

| Signal | Purpose |
|---|---|
| Gemini Vision | Visual reasoning and AI artifact detection |
| Metadata Analysis | Detects editing tools and AI generators |
| Pixel Forensics | Texture, edge, and color pattern analysis |
| Filename Analysis | Detects suspicious naming conventions |
| Signal Fusion Engine | Weighted scoring and confidence estimation |

---

# Installation

## Clone Repository

```bash
git clone https://github.com/codesbydevapriya/LuminaCheck-AI.git
cd LuminaCheck-AI
