#  LuminaCheck AI
### Where Light Reveals Truth

![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red?style=for-the-badge&logo=streamlit)
![Gemini AI](https://img.shields.io/badge/Google_Gemini-AI-green?style=for-the-badge&logo=google)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

> **LuminaCheck AI** is an advanced AI-powered web application that detects whether a digital image is **REAL**, **FAKE**, or **AI-GENERATED** using Google Gemini Vision AI.

---

##  Live Demo
 **[https://luminacheck-ai.streamlit.app](https://luminacheck-ai.streamlit.app)**

---

##  Features

| Feature | Description |
|---------|-------------|
|  AI Detection | Detects REAL, FAKE, or AI-GENERATED images |
|  Confidence Score | 0-100% confidence percentage |
|  Detailed Reasoning | 2-3 specific visual clues |
|  Detection History | Session-based history table |
|  CSV Download | Download detection report |
|  AI Chatbot | Sidebar AI assistant |
|  Dark UI | Professional animated interface |
|  Responsive | Works on mobile & desktop |

---

##  Tech Stack

- **Python** — Core programming language
- **Streamlit** — Web application framework  
- **Google Gemini AI** — Image analysis engine (gemini-2.5-flash)
- **Pillow** — Image processing
- **Pandas** — Data management
- **Streamlit Cloud** — Deployment & hosting

---

##  Run Locally
```bash
# Clone the repository
git clone https://github.com/codesbydevapriya/LuminaCheck-AI.git

# Install dependencies
pip install -r requirements.txt

# Set your API key
export GEMINI_API_KEY="your_api_key_here"

# Run the app
streamlit run app.py
```

---

##  Project Structure
```
LuminaCheck-AI/
├── app.py              # Main application
├── requirements.txt    # Dependencies
└── README.md          # Project documentation
```

---

##  Environment Variables

| Variable | Description |
|----------|-------------|
| `GEMINI_API_KEY` | Google Gemini API Key |

Get your free API key at [aistudio.google.com](https://aistudio.google.com)

---

##  How It Works

1. **Upload** any JPG/JPEG/PNG image
2. **Click** "Analyze Image Now"
3. **AI scans** for forensic clues
4. **Get verdict** — REAL  or FAKE 

---

##  Limitations

- Ultra-realistic AI images may occasionally be misclassified
- Free tier API has rate limits
- Detection accuracy depends on Gemini AI model

---

##  Future Scope

- Custom CNN model for higher accuracy
- Video deepfake detection
- Batch image processing
- Mobile application

---

##  Developer

**Devapriya** — BCA Final Year Student | March 2026

---

##  License

This project is licensed under the MIT License.

---

<p align="center">Made by Devapriya | Powered by Google Gemini AI</p>
