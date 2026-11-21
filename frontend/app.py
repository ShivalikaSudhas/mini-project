# frontend/app.py – Streamlit UI for AI-Based Fake Identity & Document Screening System
"""
Interactive Web Application for Mini-Project Presentation & Evaluation.
Provides:
  - Preset test selection (Aadhaar, PAN, Driving Licence, Passport, Voter ID)
  - Custom drag-and-drop file upload
  - Visual Forensics (EfficientNet-B0 backbone)
  - Text & Layout Forensics (OpenCV morphological text isolation & edge-sharpness variance)
  - Explainable AI (Grad-CAM heatmaps showing forged regions)
  - Architectural & Evaluation Summary for project judges
"""

import os
import sys
import json
import pathlib
from typing import Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.visual_model import get_default_visual_model, predict_visual
from src.ocr_model import run_ocr
from src.explainability import visualize_gradcam

# ── App Styling & Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="AI Fake Identity Screening",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished presentation theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .status-badge-forged {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #F87171;
    }
    .status-badge-genuine {
        background-color: #DCFCE7;
        color: #166534;
        font-weight: 700;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 1.2rem;
        display: inline-block;
        border: 1px solid #4ADE80;
    }
</style>
""", unsafe_allow_html=True)


# ── Preset Demo Documents ────────────────────────────────────────────────────
PRESETS = {
    "PAN Card (Forged)": "demo_samples/pan_forged.jpg",
    "PAN Card (Genuine)": "demo_samples/pan_genuine.jpg",
    "Aadhaar (Forged)": "demo_samples/aadhaar_forged.jpg",
    "Aadhaar (Genuine)": "demo_samples/aadhaar_genuine.jpg",
    "Driving Licence (Forged)": "demo_samples/driving_licence_forged.jpg",
    "Driving Licence (Genuine)": "demo_samples/driving_licence_genuine.jpg",
    "Passport (Forged)": "demo_samples/passport_forged.jpg",
    "Passport (Genuine)": "demo_samples/passport_genuine.jpg",
    "Voter ID (Forged)": "demo_samples/voter_id_forged.jpg",
    "Voter ID (Genuine)": "demo_samples/voter_id_genuine.jpg",
}


# ── Cached Model Loading ─────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading EfficientNet-B0 weights...")
def load_cached_visual_model(ckpt_path: str):
    return get_default_visual_model(checkpoint_path=ckpt_path)


# ── Helper Utilities ─────────────────────────────────────────────────────────
def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def draw_bounding_boxes(img_rgb: np.ndarray, boxes: list) -> np.ndarray:
    """Draw bounding boxes around isolated text fields."""
    annotated = img_rgb.copy()
    for b in boxes:
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (16, 185, 129), 2)
    return annotated


# ── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/id-verified.png", width=64)
    st.title("Screening System")
    st.markdown("**Mini-Project Evaluation Demo**")
    st.markdown("---")

    st.subheader("1. Input Document Source")
    input_mode = st.radio("Choose Input Mode:", ["Presentation Presets", "Upload Custom Image"])

    selected_image: Optional[Image.Image] = None
    image_name = ""

    if input_mode == "Presentation Presets":
        preset_choice = st.selectbox("Select Sample Document:", list(PRESETS.keys()))
        preset_rel_path = PRESETS[preset_choice]
        preset_full_path = repo_root / preset_rel_path
        if preset_full_path.is_file():
            selected_image = Image.open(preset_full_path).convert("RGB")
            image_name = preset_choice
        else:
            st.warning(f"Preset file not found: {preset_rel_path}")
    else:
        uploaded_file = st.file_uploader("Upload ID Document", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name

    st.markdown("---")
    st.subheader("2. Model Settings")
    default_ckpt = str(repo_root / "models/visual/best_visual_model.pth")
    ckpt_path = st.text_input("Checkpoint Path", value=default_ckpt)

    show_overlay = st.checkbox("Overlay OCR Text Bounding Boxes", value=True)

    st.markdown("---")
    st.caption("Developed for Mini-Project: AI-Based Multimodal Fake ID Screening")


# ── Main Content Area ────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🛡️ Multimodal Fake Identity & Document Screening System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Deep Forensic Analysis using EfficientNet-B0 Visual Branch, Morphological Text-Region Analysis & Grad-CAM Explainability</div>', unsafe_allow_html=True)

tab_demo, tab_architecture, tab_eval = st.tabs(["🚀 Live Screening Demo", "📐 System Architecture", "📊 Validation & Anti-Overfitting"])

with tab_demo:
    if selected_image is None:
        st.info("👈 Please select a preset document or upload an image from the sidebar to begin analysis.")
    else:
        # Load model
        model = load_cached_visual_model(ckpt_path)

        # Temporary save for OCR path-based processing
        temp_img_path = repo_root / "temp_eval_input.png"
        selected_image.save(temp_img_path)

        col_img, col_diag = st.columns([1, 1], gap="large")

        # Process Visual Model & OCR
        with st.spinner("Analyzing document visual signatures and layout sharpness..."):
            vis_res = predict_visual(selected_image, model=model)
            ocr_res = run_ocr(str(temp_img_path))
            gradcam_bgr = visualize_gradcam(selected_image, model)

        # Clean up temp file
        if temp_img_path.is_file():
            try:
                temp_img_path.unlink()
            except Exception:
                pass

        # Left Column: Document Viewer
        with col_img:
            st.subheader(f"📄 Target: {image_name}")
            
            img_np = np.array(selected_image)
            if show_overlay and len(ocr_res.get("boxes", [])) > 0:
                display_img = draw_bounding_boxes(img_np, ocr_res["boxes"])
                caption_text = f"Input document with {len(ocr_res['boxes'])} text regions isolated (Green)"
            else:
                display_img = img_np
                caption_text = "Raw Input Document"

            st.image(display_img, caption=caption_text, use_container_width=True)

            # Grad-CAM Heatmap Section
            st.subheader("🔥 Explainable AI (Grad-CAM Saliency)")
            gradcam_rgb = bgr_to_rgb(gradcam_bgr)
            st.image(
                gradcam_rgb,
                caption="Grad-CAM Saliency Map — Warm/Red zones indicate detected digital anomalies or manipulated areas.",
                use_container_width=True
            )

        # Right Column: Diagnostic & Verification Breakdown
        with col_diag:
            st.subheader("🔍 Forensic Screening Diagnosis")

            is_forged = vis_res["prediction"].lower() == "forged"
            pred_text = "FORGED DOCUMENT DETECTED" if is_forged else "GENUINE DOCUMENT VERIFIED"
            badge_class = "status-badge-forged" if is_forged else "status-badge-genuine"

            st.markdown(f'<div class="{badge_class}">{"🚨 " if is_forged else "✅ "}{pred_text}</div>', unsafe_allow_html=True)
            st.write("")

            # Visual Confidence
            conf = vis_res["confidence"] * 100
            st.metric("Primary Visual Confidence", f"{conf:.2f}%")
            st.progress(vis_res["confidence"])

            # Probability Breakdown
            st.markdown("#### ⚖️ Probability Distribution")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.metric("P(Genuine)", f"{vis_res['probability_genuine']*100:.1f}%")
            with p_col2:
                st.metric("P(Forged)", f"{vis_res['probability_forged']*100:.1f}%")

            # OCR Forensic Metrics
            st.markdown("#### 📝 Layout & Edge-Sharpness Analysis")
            boxes = ocr_res.get("boxes", [])
            confs = ocr_res.get("confidences", [])
            mean_sharpness = float(np.mean(confs)) if len(confs) > 0 else 0.0

            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric("Isolated Text Regions", f"{len(boxes)} fields")
            with m_col2:
                st.metric("Mean Text Edge Sharpness", f"{mean_sharpness:.3f}")

            st.caption("Edge sharpness variance detects digital copy-pasting, spliced font textures, and JPEG compression disparities.")

            # Downloadable Verification Report
            st.markdown("#### 💾 Export Audit Report")
            audit_report = {
                "document_name": image_name,
                "final_prediction": vis_res["prediction"].upper(),
                "confidence_score": round(vis_res["confidence"], 4),
                "probability_distribution": {
                    "genuine": vis_res["probability_genuine"],
                    "forged": vis_res["probability_forged"]
                },
                "layout_analysis": {
                    "text_regions_detected": len(boxes),
                    "mean_edge_sharpness": round(mean_sharpness, 4)
                },
                "backbone_architecture": "EfficientNet-B0 (1280-dim embedding)",
                "screening_status": "COMPLETED"
            }

            st.download_button(
                label="📥 Download Formal Audit Report (JSON)",
                data=json.dumps(audit_report, indent=2),
                file_name=f"audit_report_{image_name.replace(' ', '_').lower()}.json",
                mime="application/json"
            )

with tab_architecture:
    st.subheader("System Architecture & Multimodal Pipeline")
    st.markdown("""
    The system uses a **two-stream multimodal forensic pipeline** designed to overcome single-modality forgery vulnerabilities:

    1. **Visual Stream (Deep Convolutional Forensics)**:
       - **Backbone**: `EfficientNet-B0` pretrained on ImageNet and fine-tuned on identity document forensics.
       - **Feature Extraction**: Extracts a 1280-dimensional penultimate embedding capturing high-frequency noise, resampling artifacts, edge discontinuities, and color space inconsistencies.
       - **Classification Head**: Dropout (p=0.3) followed by a linear classification layer with calibrated Softmax probabilities.

    2. **Text & Layout Stream (Forensic Document Analysis)**:
       - **Stroke Isolation**: Adaptive Gaussian thresholding isolates text strokes independently of illumination gradients.
       - **Morphological Structuring**: Rectangular dilation kernels link individual characters into cohesive field regions.
       - **Forensic Splicing Detection**: Computes regional Laplacian edge-gradient sharpness variance across text boundaries to detect digitally spliced text and resolution mismatches.

    3. **Explainability Branch (Grad-CAM)**:
       - Computes gradient activation maps at the final convolutional layer (`conv_head`).
       - Localizes the exact geometric areas (e.g., altered photographs, counterfeit seals, or overwritten text fields) driving the model's determination.
    """)

with tab_eval:
    st.subheader("Validation Strategy: How We Prevent Overfitting")
    st.markdown("""
    When presenting to academic evaluation panels, prevention of dataset memorization is critical. Our project employs a **four-tier anti-overfitting protocol**:

    - **1. Leave-One-Document-Out (LODO) Cross-Validation**:
      - Instead of a naive random split, models are evaluated by holding out entire document classes (e.g. train on Aadhaar, PAN, Voter ID; test on unseen Driving Licences).
      - Demonstrates true generalized feature learning rather than template memorization.

    - **2. Strong Regularization in Classifier Head**:
      - Frozen early convolutional feature extractors preserve robust low-level filters (Gabor-like edges and textures).
      - Dropout rate ($p=0.3$) in the classification head prevents co-adaptation of features.

    - **3. Multimodal Dimensionality Reduction**:
      - Visual features are compressed into a compact 1280-dimensional L2-normalized vector.
      - Combined with compact scalar text-layout features, avoiding high-dimensional parameter explosion.

    - **4. Heavy Augmentation Pipeline**:
      - Random perspective transforms, brightness jitter, Gaussian blur, and JPEG compression simulation during training force the network to rely on invariant forensic cues rather than image quality artifacts.
    """)
