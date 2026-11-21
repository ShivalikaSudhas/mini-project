# app.py – Streamlit frontend for AI‑Based Fake Identity & Document Screening
"""A lightweight web UI built with Streamlit.

Features
- Upload an image of an identity document.
- Run the visual forgery detector (EfficientNet‑B0 backbone).
- Run the OCR model (based on OpenCV + custom logic).
- Show prediction, confidence, extracted text and a Grad‑CAM heat‑map.

The UI is only a thin wrapper around the library code in ``src`` so the heavy
logic stays in the same place as the CLI version.  This file can be launched
with::

    streamlit run frontend/app.py

Make sure the virtual environment has the dependencies from ``requirements.txt``
installed (``pip install -r requirements.txt``).
"""

import os
import pathlib
import numpy as np
import streamlit as st
from PIL import Image

# Import the project modules – they live in ``src`` relative to the repo root.
# Adding the repo root to ``sys.path`` lets the imports work when Streamlit
# executes this script from any working directory.
import sys
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from src.visual_model import get_default_visual_model, predict_visual
from src.ocr_model import extract_text
from src.explainability import visualize_gradcam

st.set_page_config(page_title="Fake ID Screening", layout="centered")
st.title("AI‑Based Fake Identity & Document Screening")
st.caption("Upload a document image – the system predicts if it is forged, extracts the text and shows a Grad‑CAM heat‑map.")

# ---------------------------------------------------------------------------
# Sidebar – model selection / options (future‑proof, currently fixed)
# ---------------------------------------------------------------------------
st.sidebar.header("Settings")
model_path = st.sidebar.text_input("Visual model checkpoint", "D:/visual/models/visual/best_visual_model.pth")

# Load the visual model once – cache the result to avoid re‑loading on every UI
# interaction.  ``@st.cache_resource`` keeps the object alive across runs.
@st.cache_resource
def load_visual_model():
    return get_default_visual_model(checkpoint_path=model_path)

visual_model = load_visual_model()

# ---------------------------------------------------------------------------
# Main UI – image upload & inference
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Choose an image of an ID document", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the image and display it
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    # Run visual detection
    with st.spinner("Running visual forgery detection …"):
        visual_result = predict_visual(img, model=visual_model)
        # ``visual_result`` is a dict with keys: logits, features, proba
        pred_label = "Forged" if visual_result["proba"][0][1] > 0.5 else "Genuine"
        confidence = visual_result["proba"][0].max().item() * 100

    st.subheader("Visual Model Prediction")
    st.write(f"**Label:** {pred_label}")
    st.write(f"**Confidence:** {confidence:.1f}%")

    # Run OCR extraction
    with st.spinner("Extracting text via OCR …"):
        ocr_text = extract_text(np.array(img))
    st.subheader("Extracted Text (OCR)")
    st.code(ocr_text)

    # Show Grad‑CAM heat‑map for the visual model
    with st.spinner("Generating Grad‑CAM visualisation …"):
        cam_img = visualize_gradcam(img, visual_model)
    st.subheader("Grad‑CAM Heat‑map")
    st.image(cam_img, use_column_width=True)

    # Provide download of results as JSON
    import json, io, base64
    result_json = {
        "prediction": pred_label,
        "confidence": confidence,
        "ocr_text": ocr_text,
    }
    json_str = json.dumps(result_json, indent=2)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f"data:file/json;base64,{b64}"
    st.markdown(f"[Download result JSON]({href})")
else:
    st.info("Upload an image to start the analysis.")
