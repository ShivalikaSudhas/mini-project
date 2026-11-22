# frontend/app.py – Streamlit UI for AI-Based Fake Identity & Document Screening System
"""
Interactive Web Application for Mini-Project Presentation & Evaluation.
Collaborative Multimodal Architecture:
  - Visual Stream: Deep EfficientNet-B0 + Grad-CAM Explainability (Shivalika)
  - OCR Stream: PaddleOCR Text Engine + 11-D Layout Feature Profiling (Vidyaaa)
  - Multimodal Fusion Stream: 27-D Feature Synthesis & Cross-Document LODTO Generalization (Liza)
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
from visual_feature_extractor import VISUAL_FEATURE_NAMES
from verify_new_id import verify_id_multimodal

# ── App Styling & Configuration ───────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Fake ID Screening",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished presentation theme
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.3rem;
    }
    .status-badge-genuine {
        background-color: #ECFDF5;
        color: #065F46;
        border: 2px solid #10B981;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.25rem;
        text-align: center;
        letter-spacing: 0.5px;
    }
    .status-badge-forged {
        background-color: #FEF2F2;
        color: #991B1B;
        border: 2px solid #EF4444;
        padding: 10px 18px;
        border-radius: 8px;
        font-weight: 800;
        font-size: 1.25rem;
        text-align: center;
        letter-spacing: 0.5px;
    }
    .stream-badge-genuine {
        background-color: #F0FDF4;
        color: #166534;
        border: 1px solid #BBF7D0;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
        text-align: center;
    }
    .stream-badge-forged {
        background-color: #FFF1F2;
        color: #9F1239;
        border: 1px solid #FECDD3;
        padding: 6px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.95rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        font-size: 0.95rem;
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
    st.image("https://img.icons8.com/color/96/id-verified.png", width=60)
    st.title("Screening System")
    st.markdown("**AI Mini-Project Evaluation Suite**")
    st.markdown("""
    * **Visual:** Shivalika
    * **OCR:** Vidyaaa
    * **Fusion Lead:** Liza
    """)
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
    st.subheader("2. Settings")
    default_ckpt = str(repo_root / "models/visual/best_visual_model.pth")
    ckpt_path = st.text_input("Visual Checkpoint Path", value=default_ckpt)
    show_overlay = st.checkbox("Overlay OCR Text Bounding Boxes", value=True)

    st.markdown("---")
    st.caption("AI-Based Multimodal Fake ID Screening")


# ── Main Content Area ────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🛡️ Multimodal Fake Identity & Document Screening System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Collaborative Multi-Stream Forensics: EfficientNet-B0 Visual Branch + PaddleOCR Layout Engine + 27-D Multimodal Fusion MLP</div>', unsafe_allow_html=True)

tab_demo, tab_features, tab_architecture, tab_lodto = st.tabs([
    "🚀 Live Screening Demo",
    "🔬 27-D Multimodal Feature Vector",
    "📐 System Architecture & Attribution",
    "📊 LODTO Generalization Study"
])

with tab_demo:
    if selected_image is None:
        st.info("👈 Please select a preset document or upload an image from the sidebar to begin analysis.")
    else:
        # Load visual model
        vis_model = load_cached_visual_model(ckpt_path)

        # Temporary save for path-based extractors
        temp_img_path = repo_root / "temp_eval_input.png"
        selected_image.save(temp_img_path)

        # Run Pipeline
        with st.spinner("Executing Visual Forensics, PaddleOCR Layout Extraction, and 27-D Multimodal Fusion..."):
            vis_res = predict_visual(selected_image, model=vis_model)
            ocr_pred = predict_ocr(str(temp_img_path))
            fusion_res = verify_id_multimodal(str(temp_img_path))
            gradcam_bgr = visualize_gradcam(selected_image, vis_model)

        # Clean up temp file
        if temp_img_path.is_file():
            try:
                temp_img_path.unlink()
            except Exception:
                pass

        col_img, col_diag = st.columns([1, 1], gap="large")

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
            st.subheader("🔍 Multimodal Screening Diagnosis")

            # PRIMARY VERDICT: Multimodal Fusion
            fused_is_forged = fusion_res.get("verdict", "").upper() == "FORGED"
            fusion_badge = "status-badge-forged" if fused_is_forged else "status-badge-genuine"
            verdict_text = "FORGED DOCUMENT DETECTED" if fused_is_forged else "GENUINE DOCUMENT VERIFIED"

            st.markdown(f'<div class="{fusion_badge}">{"🚨 " if fused_is_forged else "✅ "}FINAL VERDICT: {verdict_text}</div>', unsafe_allow_html=True)
            st.caption(f"**Fusion Decision Engine:** 27-D Multimodal MLP Classifier | Certainty: **{fusion_res.get('confidence_pct', 0.0)}%**")

            st.write("")

            # SUB-STREAM BREAKDOWN
            st.markdown("#### ⚖️ Individual Stream Breakdown")
            s_col1, s_col2 = st.columns(2)

            with s_col1:
                st.markdown("**1️⃣ Visual Stream (Shivalika)**")
                is_vis_forged = vis_res["prediction"].lower() == "forged"
                vis_badge = "stream-badge-forged" if is_vis_forged else "stream-badge-genuine"
                st.markdown(f'<div class="{vis_badge}">{"🚨 Forged" if is_vis_forged else "✅ Genuine"} ({vis_res["confidence"]*100:.1f}%)</div>', unsafe_allow_html=True)
                st.caption(f"P(Forged): {vis_res['probability_forged']*100:.1f}%")

            with s_col2:
                st.markdown(f"**2️⃣ OCR Stream (Vidyaaa)**")
                is_ocr_forged = ocr_pred["prediction"].lower() == "forged"
                ocr_badge = "stream-badge-forged" if is_ocr_forged else "stream-badge-genuine"
                st.markdown(f'<div class="{ocr_badge}">{"🚨 Forged" if is_ocr_forged else "✅ Genuine"} ({ocr_pred["confidence"]*100:.1f}%)</div>', unsafe_allow_html=True)
                st.caption(f"P(Forged): {ocr_pred['probability_forged']*100:.1f}%")

            st.markdown("---")

            # FORENSIC DIAGNOSTIC CLUES
            st.markdown("#### 🔬 Automated Forensic Clues")
            for clue in fusion_res.get("diagnostics", []):
                st.markdown(f"• **{clue}**")

            # Quick Metric Highlights
            m1, m2, m3 = st.columns(3)
            with m1:
                st.metric("Text Regions", f"{fusion_res.get('text_boxes_detected', 0)}")
            with m2:
                st.metric("Mean OCR Conf", f"{fusion_res.get('mean_ocr_conf', 0.0):.3f}")
            with m3:
                st.metric("Tampering Prob", f"{fusion_res.get('forged_probability_pct', 0.0)}%")

            # Downloadable Verification Report
            st.markdown("#### 💾 Export Audit Report")
            audit_report = {
                "document_name": image_name,
                "final_multimodal_verdict": fusion_res.get("verdict"),
                "multimodal_certainty_pct": fusion_res.get("confidence_pct"),
                "forged_probability_pct": fusion_res.get("forged_probability_pct"),
                "stream_contributions": {
                    "visual_stream": {
                        "lead": "Shivalika",
                        "model": "EfficientNet-B0",
                        "prediction": vis_res["prediction"].upper(),
                        "confidence": round(vis_res["confidence"], 4)
                    },
                    "ocr_stream": {
                        "lead": "Vidyaaa",
                        "model": "PaddleOCR + 11-D Layout Features",
                        "prediction": ocr_pred["prediction"].upper(),
                        "confidence": round(ocr_pred["confidence"], 4)
                    },
                    "multimodal_fusion": {
                        "lead": "Liza",
                        "model": "27-D Feature MLP (64-32 units)",
                        "verdict": fusion_res.get("verdict"),
                        "certainty": fusion_res.get("confidence_pct")
                    }
                },
                "diagnostics": fusion_res.get("diagnostics", [])
            }

            st.download_button(
                label="📥 Download Full Multimodal Audit Report (JSON)",
                data=json.dumps(audit_report, indent=2),
                file_name=f"multimodal_audit_{image_name.replace(' ', '_').lower()}.json",
                mime="application/json"
            )

with tab_features:
    st.subheader("🔬 27-Dimensional Multimodal Feature Vector Breakdown")
    st.markdown("""
    The multimodal screening engine synthesizes **16 Visual Forensic Features** and **11 OCR Layout Features** into a unified 27-D vector representation:
    """)

    if selected_image is not None and 'fusion_res' in locals():
        fdict = fusion_res.get("features_dict", {})

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.markdown("### 📷 Visual Forensic Features (16-D)")
            vis_rows = []
            vis_desc = {
                "cnn_feat_1": "MobileNet-V3 deep spatial latent chunk 1",
                "cnn_feat_2": "MobileNet-V3 deep spatial latent chunk 2",
                "cnn_feat_3": "MobileNet-V3 deep spatial latent chunk 3",
                "cnn_feat_4": "MobileNet-V3 deep spatial latent chunk 4",
                "cnn_feat_5": "MobileNet-V3 deep spatial latent chunk 5",
                "cnn_feat_6": "MobileNet-V3 deep spatial latent chunk 6",
                "cnn_feat_7": "MobileNet-V3 deep spatial latent chunk 7",
                "cnn_feat_8": "MobileNet-V3 deep spatial latent chunk 8",
                "laplacian_var": "Laplacian variance (flags smoothing/blurring around tamper points)",
                "laplacian_mean": "Mean Laplacian absolute response (edge sharpness)",
                "gradient_mean": "Mean Sobel gradient magnitude across both axes",
                "gradient_std": "Standard deviation of Sobel gradient magnitude (edge consistency)",
                "high_freq_ratio": "2D Fast Fourier Transform high-frequency cutoff ratio (flags seams)",
                "ela_mean": "Error Level Analysis (ELA) mean JPEG compression residual",
                "ela_std": "ELA standard deviation of compression residuals",
                "color_std_ratio": "Ratio of blue to red channel standard deviations (color covariance)"
            }
            for k in VISUAL_FEATURE_NAMES:
                vis_rows.append({
                    "Feature Name": k,
                    "Value": round(fdict.get(k, 0.0), 4),
                    "Forensic Intuition": vis_desc.get(k, "")
                })
            st.dataframe(pd.DataFrame(vis_rows), use_container_width=True)

        with col_f2:
            st.markdown("### 📝 OCR & Layout Features (11-D)")
            ocr_rows = []
            ocr_desc = {
                "mean_confidence": "Average PaddleOCR confidence across detected fields",
                "min_confidence": "Lowest confidence score observed in a single text region",
                "max_confidence": "Highest confidence score observed across document",
                "std_confidence": "Variance of confidence scores (flags mixed genuine/forged fields)",
                "low_conf_count": "Count of text boxes with recognition confidence < 0.85",
                "low_conf_ratio": "Fraction of low-confidence text regions",
                "num_text_boxes": "Total number of isolated text regions",
                "text_area_ratio": "Ratio of text surface area to total card surface area",
                "mean_aspect_ratio": "Average width-to-height ratio of text boxes (detects stretched text)",
                "box_height_std": "Standard deviation of text box heights (font size inconsistency)",
                "mean_char_density": "Character count per pixel of text box area"
            }
            for k in OCR_FEATURE_NAMES:
                ocr_rows.append({
                    "Feature Name": k,
                    "Value": round(fdict.get(k, 0.0), 4),
                    "Forensic Intuition": ocr_desc.get(k, "")
                })
            st.dataframe(pd.DataFrame(ocr_rows), use_container_width=True)
    else:
        st.info("Select or upload a document in the sidebar to inspect its extracted 27-D multimodal feature vector.")

with tab_architecture:
    st.subheader("📐 System Architecture & Teammate Attribution")
    st.markdown(r"""
    This project integrates the specialized contributions of all three team members into a robust screening pipeline:

    | Stream | Contributor | Core Technology | Key Output |
    |--------|-------------|-----------------|------------|
    | **Visual Stream** | **Shivalika** | PyTorch EfficientNet-B0 + Grad-CAM Saliency | 1280-D spatial embedding, visual probability score, explainable heatmap |
    | **OCR Stream** | **Vidyaaa** | PaddleOCR (PP-OCRv4) + 11-D Statistical Layout | 11-D layout feature vector, OCR baseline classifier |
    | **Multimodal Fusion** | **Liza (Fusion Lead)** | 27-D Synthesis + MLP Neural Network (64-32) | Final multimodal screening verdict, LODTO generalization study |

    ---

    ### Pipeline Architecture Flow:
    1. **Dual Input Acquisition**: Document image passes concurrently to the Visual CNN and PaddleOCR text engine.
    2. **Feature Extraction**:
       - Shivalika's pipeline computes 1280-D deep convolutional features and Grad-CAM activation maps.
       - Vidyaaa's pipeline extracts text bounding boxes and computes the 11-D OCR layout feature vector.
       - Liza's visual extractor computes 8 spatial latents + 8 forensic texture metrics (16-D).
    3. **Multimodal Synthesis**: The 16-D visual and 11-D OCR vectors concatenate into a 27-D vector $\mathbf{x}_{\text{fused}} \in \mathbb{R}^{27}$.
    4. **Standardization & Classification**: Features pass through a fitted `StandardScaler` and into the 2-layer MLP Classifier to render the decisive verdict.
    """)

with tab_lodto:
    st.subheader("📊 5-Fold Leave-One-Document-Type-Out (LODTO) Study")
    st.markdown("""
    **Conducted by Liza (Fusion Lead)** to prove that the multimodal model generalizes to completely unseen document templates rather than memorizing training layouts.
    In each fold, the classifier was trained on 4 document types and evaluated on a hold-out 5th document type.
    """)

    # Display LODTO report figures
    rep_dir = repo_root / "reports"
    img_gap = rep_dir / "lodto_generalization_gap.png"
    img_folds = rep_dir / "lodto_fold_breakdown.png"
    img_roc = rep_dir / "roc_curves_comparison.png"

    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if img_gap.is_file():
            st.image(str(img_gap), caption="Figure 1: Multimodal Generalization Gap vs Single-Modality Baselines", use_container_width=True)
    with col_p2:
        if img_roc.is_file():
            st.image(str(img_roc), caption="Figure 2: Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)

    if img_folds.is_file():
        st.image(str(img_folds), caption="Figure 3: LODTO 5-Fold Performance Breakdown Across Unseen Card Types", use_container_width=True)

    # Display Empirical Results Table
    lodto_csv = repo_root / "lodto_experiment_results.csv"
    if lodto_csv.is_file():
        st.markdown("### 📋 Empirical Fold-by-Fold Cross-Document Results")
        df_lodto = pd.read_csv(lodto_csv)
        st.dataframe(df_lodto, use_container_width=True)
