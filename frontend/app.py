# frontend/app.py – Streamlit UI for AI-Based Fake Identity & Document Screening System
"""
Interactive Web Application for Mini-Project Presentation & Evaluation.
Provides:
  - Preset test selection (Aadhaar, PAN, Driving Licence, Passport, Voter ID)
  - Custom drag-and-drop file upload
  - Visual Forensics (EfficientNet-B0 backbone + Grad-CAM explainability) [Shivalika]
  - OCR Stream (PaddleOCR + 11-D Numerical Layout Features + OCR Classifier) [Vidyaaa]
  - Architectural & Evaluation Summary for project judges
"""

import os
import sys
import json
import pathlib
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Ensure project root is in sys.path
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.visual_model import get_default_visual_model, predict_visual
from src.explainability import visualize_gradcam
from ocr_feature_extractor import run_ocr, extract_ocr_features, predict_ocr, OCR_FEATURE_NAMES

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
    .status-badge-genuine {
        background-color: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
    }
    .status-badge-forged {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 1px solid #FECACA;
        padding: 8px 16px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1.1rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        font-size: 1rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ── Preset Sample Catalog ─────────────────────────────────────────────────────
PRESETS = {
    "Aadhaar (Genuine)": "demo_samples/aadhaar_genuine.jpg",
    "Aadhaar (Forged)": "demo_samples/aadhaar_forged.jpg",
    "PAN Card (Genuine)": "demo_samples/pan_genuine.jpg",
    "PAN Card (Forged)": "demo_samples/pan_forged.jpg",
    "Driving Licence (Genuine)": "demo_samples/driving_licence_genuine.jpg",
    "Driving Licence (Forged)": "demo_samples/driving_licence_forged.jpg",
    "Passport (Genuine)": "demo_samples/passport_genuine.jpg",
    "Passport (Forged)": "demo_samples/passport_forged.jpg",
    "Voter ID (Genuine)": "demo_samples/voter_id_genuine.jpg",
    "Voter ID (Forged)": "demo_samples/voter_id_forged.jpg",
}


@st.cache_resource(show_spinner=False)
def load_cached_visual_model(checkpoint_path: str):
    """Load and cache the trained visual forensic model."""
    ckpt = pathlib.Path(checkpoint_path)
    if not ckpt.is_file():
        st.warning(f"Trained checkpoint not found at: {checkpoint_path}. Using architecture with untrained weights.")
        return get_default_visual_model()
    return get_default_visual_model(checkpoint_path=str(ckpt))


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def draw_bounding_boxes(img_rgb: np.ndarray, boxes: list, texts: list = None) -> np.ndarray:
    """Draw bounding boxes and detected text around isolated text fields."""
    annotated = img_rgb.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (16, 185, 129), 2)
        if texts and i < len(texts) and texts[i] and not texts[i].startswith("field_"):
            label = texts[i][:15]
            cv2.putText(annotated, label, (x1, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (16, 185, 129), 1)
    return annotated


# ── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/id-verified.png", width=64)
    st.title("Screening System")
    st.markdown("**Mini-Project Evaluation Demo**")
    st.caption("Visual: Shivalika | OCR: Vidyaaa")
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
    ckpt_path = st.text_input("Visual Checkpoint Path", value=default_ckpt)

    show_overlay = st.checkbox("Overlay OCR Text Bounding Boxes", value=True)

    st.markdown("---")
    st.caption("AI-Based Multimodal Fake ID Screening")


# ── Main Content Area ────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🛡️ Multimodal Fake Identity & Document Screening System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Dual-Stream Forensics: EfficientNet-B0 Visual Branch + PaddleOCR 11-D Layout Engine & Grad-CAM Explainability</div>', unsafe_allow_html=True)

tab_demo, tab_ocr, tab_architecture, tab_eval = st.tabs([
    "🚀 Live Screening Demo",
    "📝 OCR & 11-D Features",
    "📐 System Architecture",
    "📊 Validation & Anti-Overfitting"
])

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
        with st.spinner("Analyzing document visual signatures and PaddleOCR layout features..."):
            vis_res = predict_visual(selected_image, model=model)
            ocr_pred = predict_ocr(str(temp_img_path))
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
            boxes = ocr_pred.get("boxes", [])
            texts = ocr_pred.get("texts", [])
            if show_overlay and len(boxes) > 0:
                display_img = draw_bounding_boxes(img_np, boxes, texts)
                caption_text = f"Input document with {len(boxes)} text regions isolated ({ocr_pred.get('engine', 'OCR')})"
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
            st.subheader("🔍 Dual-Stream Screening Diagnosis")

            # Stream 1: Visual Forensics
            st.markdown("#### 1️⃣ Visual Stream (Shivalika — EfficientNet-B0)")
            is_vis_forged = vis_res["prediction"].lower() == "forged"
            vis_pred_text = "FORGED DOCUMENT DETECTED" if is_vis_forged else "GENUINE DOCUMENT VERIFIED"
            vis_badge = "status-badge-forged" if is_vis_forged else "status-badge-genuine"
            st.markdown(f'<div class="{vis_badge}">{"🚨 " if is_vis_forged else "✅ "}{vis_pred_text}</div>', unsafe_allow_html=True)
            
            p_v1, p_v2 = st.columns(2)
            with p_v1:
                st.metric("Visual Confidence", f"{vis_res['confidence']*100:.1f}%")
            with p_v2:
                st.metric("P(Forged | Visual)", f"{vis_res['probability_forged']*100:.1f}%")

            st.write("")

            # Stream 2: OCR Stream
            st.markdown(f"#### 2️⃣ OCR & Layout Stream (Vidyaaa — {ocr_pred.get('engine', 'PaddleOCR')})")
            is_ocr_forged = ocr_pred["prediction"].lower() == "forged"
            ocr_pred_text = "FORGED LAYOUT / SPLICED TEXT" if is_ocr_forged else "CONSISTENT TEXT & LAYOUT"
            ocr_badge = "status-badge-forged" if is_ocr_forged else "status-badge-genuine"
            st.markdown(f'<div class="{ocr_badge}">{"🚨 " if is_ocr_forged else "✅ "}{ocr_pred_text}</div>', unsafe_allow_html=True)

            p_o1, p_o2 = st.columns(2)
            with p_o1:
                st.metric("OCR Stream Confidence", f"{ocr_pred['confidence']*100:.1f}%")
            with p_o2:
                st.metric("P(Forged | OCR)", f"{ocr_pred['probability_forged']*100:.1f}%")

            st.markdown("---")

            # OCR Summary Metrics
            st.markdown("#### 📝 Extracted Layout Diagnostics")
            feats = ocr_pred.get("features", {})
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                st.metric("Text Boxes", f"{int(feats.get('num_text_boxes', 0))}")
            with m_col2:
                st.metric("Mean Conf", f"{feats.get('mean_confidence', 0.0):.3f}")
            with m_col3:
                st.metric("Low Conf Ratio", f"{feats.get('low_conf_ratio', 0.0)*100:.1f}%")

            # Downloadable Verification Report
            st.markdown("#### 💾 Export Audit Report")
            audit_report = {
                "document_name": image_name,
                "visual_stream": {
                    "prediction": vis_res["prediction"].upper(),
                    "confidence": round(vis_res["confidence"], 4),
                    "probability_genuine": round(vis_res["probability_genuine"], 4),
                    "probability_forged": round(vis_res["probability_forged"], 4),
                    "lead": "Shivalika"
                },
                "ocr_stream": {
                    "prediction": ocr_pred["prediction"].upper(),
                    "confidence": round(ocr_pred["confidence"], 4),
                    "probability_genuine": round(ocr_pred["probability_genuine"], 4),
                    "probability_forged": round(ocr_pred["probability_forged"], 4),
                    "engine": ocr_pred.get("engine", "PaddleOCR"),
                    "features": feats,
                    "lead": "Vidyaaa"
                },
                "backbone_architectures": {
                    "visual": "EfficientNet-B0 (1280-dim embedding)",
                    "ocr": "PaddleOCR + 11-D Numerical Layout MLP (32-16 units)"
                },
                "screening_status": "COMPLETED"
            }

            st.download_button(
                label="📥 Download Formal Dual-Stream Audit Report (JSON)",
                data=json.dumps(audit_report, indent=2),
                file_name=f"audit_report_{image_name.replace(' ', '_').lower()}.json",
                mime="application/json"
            )

with tab_ocr:
    st.subheader("Vidyaaa's OCR Stream: PaddleOCR & 11-D Layout Feature Profiling")
    st.markdown("""
    The OCR stream analyzes identity documents using **PaddleOCR** text recognition coupled with **11-dimensional geometric and confidence profiling**.
    This captures typographic inconsistencies, abnormal font aspect ratios, character density anomalies, and text splicing.
    """)

    if selected_image is not None and 'ocr_pred' in locals():
        st.markdown("### 📊 Extracted 11-D Feature Vector for Current Document")
        feats_data = []
        descriptions = {
            "mean_confidence": "Average OCR confidence across all detected text fields",
            "min_confidence": "Lowest confidence score observed in a single text region",
            "max_confidence": "Highest confidence score observed across document",
            "std_confidence": "Variance of confidence scores (high std = mixed genuine/tampered fields)",
            "low_conf_count": "Count of text boxes with recognition confidence < 0.85",
            "low_conf_ratio": "Fraction of low-confidence text regions",
            "num_text_boxes": "Total number of isolated text regions",
            "text_area_ratio": "Ratio of text surface area to total card surface area",
            "mean_aspect_ratio": "Average width-to-height ratio of text boxes (detects stretched text)",
            "box_height_std": "Standard deviation of text box heights (font size inconsistency)",
            "mean_char_density": "Character count per pixel of text box area"
        }
        for k in OCR_FEATURE_NAMES:
            val = ocr_pred.get("features", {}).get(k, 0.0)
            feats_data.append({
                "Feature Name": k,
                "Extracted Value": val,
                "Forensic Significance": descriptions.get(k, "")
            })
        df_feats = pd.DataFrame(feats_data)
        st.dataframe(df_feats, use_container_width=True)

        st.markdown("### 📝 Extracted Text Fields & Confidences")
        texts_list = ocr_pred.get("texts", [])
        if texts_list:
            df_texts = pd.DataFrame({
                "Index": list(range(1, len(texts_list) + 1)),
                "Extracted Field Text": texts_list
            })
            st.dataframe(df_texts, use_container_width=True)
        else:
            st.info("No text fields isolated.")
    else:
        st.info("Select or upload a document in the sidebar to view its extracted 11-D OCR features.")

with tab_architecture:
    st.subheader("System Architecture & Multimodal Pipeline")
    st.markdown("""
    The screening framework employs a **collaborative dual-stream multimodal architecture**:

    1. **Visual Stream (Deep Convolutional Forensics — Shivalika)**:
       - **Backbone**: `EfficientNet-B0` pretrained on ImageNet and fine-tuned on identity document forensics.
       - **Feature Extraction**: Extracts a 1280-dimensional penultimate embedding capturing high-frequency noise, resampling artifacts, edge discontinuities, and color space inconsistencies.
       - **Classification Head**: Dropout (p=0.3) followed by a linear classification layer with calibrated Softmax probabilities.
       - **Explainability**: Grad-CAM saliency heatmaps highlighting exact pixel regions indicative of forgery.

    2. **OCR & Layout Stream (PaddleOCR Forensics — Vidyaaa)**:
       - **Text Extraction**: Multilingual text isolation with PaddleOCR.
       - **11-D Statistical & Layout Profiling**: Computes 6 confidence metrics and 5 spatial layout/density metrics.
       - **MLP Classifier**: 2-layer Neural Network (32-16 hidden units) trained strictly on layout metrics to detect tampering independently of visual pixels.

    3. **Multi-Disciplinary Synthesis**:
       - Both streams provide complementary perspectives: visual stream identifies pixel-level texture tampering; OCR stream identifies semantic and typographical anomalies.
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
