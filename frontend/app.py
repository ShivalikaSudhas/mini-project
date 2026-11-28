# frontend/app.py – Editorial Research Interface for Identity Document Forensics
"""
AI-Based Fake Identity & Document Screening System
Editorial Magazine & Research Publication Layout

Collaborative Forensic Architecture:
  • Visual Forensics: Deep EfficientNet-B0 + Grad-CAM Saliency (Shivalika)
  • OCR Diagnostics: 11-Dimensional Document Layout Profiling (Vidyaaa)
  • Multimodal Fusion: 27-D Feature Synthesis & Cross-Document LODTO Study (Liza)
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

# Ensure repository root is on sys.path
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.visual_model import get_default_visual_model, predict_visual
from src.explainability import (
    visualize_gradcam,
    visualize_gradcam_with_heatmap,
    generate_xai_explanation
)
from ocr_feature_extractor import run_ocr, extract_ocr_features, predict_ocr, OCR_FEATURE_NAMES
from visual_feature_extractor import VISUAL_FEATURE_NAMES
from verify_new_id import verify_id_multimodal

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FORGEX",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Editorial Theme & Custom Design System ────────────────────────────────────
# Color Architecture (Soft Editorial Lavender Publication):
#   #F1F2FA – Lavender Canvas / Page Background (60%)
#   #FEFEFE – White Paper Cards floating above canvas (25%)
#   #D2D6E4 – Elevated Lavender / Sidebar / Focal Panels (10%)
#   #9498B2 – Accent Surface, Borders & Subtle Indicators (5%)
#   #525355 – Secondary Body & Metadata
#   #141415 – Primary Editorial Typography
EDITORIAL_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global Canvas Reset — Soft Lavender Atmosphere */
html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stMain"] {
    background-color: #F1F2FA !important;
    color: #141415 !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    -webkit-font-smoothing: antialiased;
}

[data-testid="stMainBlockContainer"] {
    max-width: 1380px !important;
    padding-top: 1.5rem !important;
    padding-bottom: 4rem !important;
    padding-left: 2.5rem !important;
    padding-right: 2.5rem !important;
}

/* Sidebar — #D2D6E4 Magazine Navigation Rail */
[data-testid="stSidebar"] {
    background-color: #D2D6E4 !important;
    border-right: none !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label {
    color: #525355 !important;
    font-size: 0.9rem !important;
}

/* Editorial Typography */
.editorial-kicker {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #9498B2;
    margin-bottom: 0.4rem;
}

.editorial-headline {
    font-family: 'Newsreader', serif;
    font-size: 3.1rem;
    font-weight: 500;
    line-height: 1.15;
    letter-spacing: -0.025em;
    color: #141415;
    margin-bottom: 0.6rem;
}

.editorial-subdeck {
    font-size: 1.05rem;
    font-weight: 400;
    line-height: 1.55;
    color: #525355;
    max-width: 860px;
    margin-bottom: 2.2rem;
}

.editorial-divider {
    height: 1px;
    background: linear-gradient(90deg, rgba(148, 152, 178, 0.45) 0%, rgba(148, 152, 178, 0.08) 100%);
    margin: 2rem 0;
    border: none;
}

/* Navigation Tabs – Underline Editorial Strip */
.stTabs [data-baseweb="tab-list"] {
    gap: 2.5rem !important;
    border-bottom: 1px solid rgba(148, 152, 178, 0.25) !important;
    padding-bottom: 0px !important;
    margin-bottom: 2rem !important;
    background: transparent !important;
}

/* Inactive tabs – Crisp legible charcoal (#525355) */
.stTabs [data-baseweb="tab"],
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] span,
.stTabs button {
    background: transparent !important;
    border: none !important;
    color: #525355 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.01em !important;
    padding: 0.75rem 0.25rem !important;
    transition: all 0.2s ease !important;
    text-decoration: none !important;
}

.stTabs [data-baseweb="tab"]:hover,
.stTabs [data-baseweb="tab"]:hover p,
.stTabs [data-baseweb="tab"]:hover span,
.stTabs button:hover {
    color: #141415 !important;
}

/* Active tab – High-contrast dark typography (#141415) with soft slate-lavender underline (#9498B2) */
.stTabs [aria-selected="true"],
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span,
.stTabs button[aria-selected="true"] {
    color: #141415 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #9498B2 !important;
    background: transparent !important;
}

/* Tab highlight bar – Completely override Streamlit red/coral highlight */
.stTabs [data-baseweb="tab-highlight"],
.stTabs div[data-testid="stTabHighlight"] {
    background-color: #9498B2 !important;
    height: 2px !important;
}

/* Focus and active state hygiene */
.stTabs button:focus,
.stTabs button:active {
    box-shadow: none !important;
    color: #141415 !important;
}

/* Hero Document Visual Container */
.doc-hero-wrapper {
    margin-bottom: 1.5rem;
}

.doc-hero-kicker {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #9498B2;
    margin-bottom: 0.25rem;
}

.doc-hero-title {
    font-family: 'Newsreader', serif;
    font-size: 1.65rem;
    font-weight: 600;
    color: #141415;
    letter-spacing: -0.015em;
    margin-bottom: 1rem;
}

.editorial-img-frame {
    background: #FEFEFE;
    padding: 12px;
    border-radius: 14px;
    box-shadow: 0 2px 12px rgba(20, 20, 21, 0.06);
    display: inline-block;
    width: 100%;
}

.editorial-caption {
    font-size: 0.8rem;
    color: #525355;
    font-style: italic;
    margin-top: 0.5rem;
    text-align: left;
}

/* Editorial Cards & Panels — White paper floating on lavender canvas */
.editorial-card {
    background: #FEFEFE;
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 2px 12px rgba(20, 20, 21, 0.06);
    margin-bottom: 1.5rem;
    border: 1px solid rgba(148, 152, 178, 0.12);
}

.editorial-card-accent {
    background: #D2D6E4;
    border-radius: 14px;
    padding: 24px;
    box-shadow: 0 2px 10px rgba(20, 20, 21, 0.05);
    margin-bottom: 1.5rem;
}

/* Analytical Verdict Panel — White paper card */
.verdict-panel {
    background-color: #FEFEFE;
    border-radius: 14px;
    padding: 28px 24px;
    box-shadow: 0 2px 14px rgba(20, 20, 21, 0.06);
    margin-bottom: 1.75rem;
    border: 1px solid rgba(148, 152, 178, 0.12);
}

.verdict-kicker {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #9498B2;
    margin-bottom: 0.6rem;
}

.verdict-title {
    font-family: 'Newsreader', serif;
    font-size: 2.2rem;
    font-weight: 600;
    line-height: 1.1;
    color: #141415;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
}

.verdict-meta {
    font-size: 0.95rem;
    font-weight: 500;
    color: #525355;
    margin-bottom: 1.1rem;
}

.verdict-summary {
    font-size: 0.88rem;
    line-height: 1.55;
    color: #525355;
    border-top: 1px solid rgba(148, 152, 178, 0.3);
    padding-top: 0.9rem;
}

/* Stream Metric Rows */
.stream-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    padding: 10px 0;
    border-bottom: 1px solid rgba(148, 152, 178, 0.18);
}

.stream-name {
    font-size: 0.88rem;
    font-weight: 600;
    color: #141415;
}

.stream-author {
    font-size: 0.75rem;
    color: #9498B2;
    font-weight: 400;
    margin-left: 4px;
}

.stream-val {
    font-size: 0.9rem;
    font-weight: 600;
    color: #141415;
    font-family: 'JetBrains Mono', monospace;
}

/* Forensic Clue Block */
.forensic-note-item {
    font-size: 0.84rem;
    color: #525355;
    line-height: 1.55;
    margin-bottom: 0.6rem;
    padding-left: 1rem;
    position: relative;
}

.forensic-note-item::before {
    content: "•";
    position: absolute;
    left: 0;
    color: #9498B2;
    font-size: 1.1rem;
    line-height: 1;
}

/* Editorial Button Override */
div.stDownloadButton > button,
div.stButton > button {
    background-color: #9498B2 !important;
    color: #FEFEFE !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.65rem 1.25rem !important;
    box-shadow: 0 2px 8px rgba(148, 152, 178, 0.25) !important;
    transition: all 0.2s ease !important;
}

div.stDownloadButton > button:hover,
div.stButton > button:hover {
    background-color: #141415 !important;
    color: #FEFEFE !important;
    box-shadow: 0 4px 12px rgba(20, 20, 21, 0.15) !important;
}

/* Expander Overrides */
.streamlit-expanderHeader {
    background-color: #FEFEFE !important;
    border-radius: 8px !important;
    color: #141415 !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    border: none !important;
}

[data-testid="stExpander"] {
    border: 1px solid rgba(148, 152, 178, 0.2) !important;
    border-radius: 10px !important;
    background-color: #FEFEFE !important;
    box-shadow: none !important;
}

/* Dataframe clean editorial skin */
[data-testid="stDataFrame"] {
    border-radius: 10px !important;
    overflow: hidden !important;
    border: 1px solid rgba(148, 152, 178, 0.2) !important;
}
</style>
"""

st.markdown(EDITORIAL_CSS, unsafe_allow_html=True)


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
    ckpt = pathlib.Path(checkpoint_path).resolve()
    if not ckpt.is_file():
        fallback = repo_root / "models" / "visual" / "best_visual_model.pth"
        if fallback.is_file():
            ckpt = fallback
        else:
            return get_default_visual_model()
    return get_default_visual_model(checkpoint_path=str(ckpt))


def bgr_to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def draw_bounding_boxes(img_rgb: np.ndarray, boxes: list, texts: list = None) -> np.ndarray:
    """Draw refined, minimal bounding boxes around detected typography fields."""
    annotated = img_rgb.copy()
    for i, b in enumerate(boxes):
        x1, y1, x2, y2 = map(int, b)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (148, 152, 178), 1)
        if texts and i < len(texts) and texts[i] and not texts[i].startswith("field_"):
            label = texts[i][:18]
            cv2.putText(
                annotated, label, (x1, max(12, y1 - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (82, 83, 85), 1, cv2.LINE_AA
            )
    return annotated


# ── Editorial Sidebar (Controls & Curated Presets) ─────────────────────────────
with st.sidebar:
    st.markdown('<div class="editorial-kicker">Screening Suite</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family: \'Newsreader\', serif; font-size: 1.45rem; font-weight: 600; color: #141415; margin-bottom: 0.2rem;">Investigation Console</div>', unsafe_allow_html=True)
    st.caption("AI-Based Multimodal Identity & Document Screening System")
    st.markdown('<hr style="border: none; height: 1px; background: rgba(0,0,0,0); margin: 1.2rem 0;">', unsafe_allow_html=True)

    st.markdown('<div style="font-size: 0.8rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #141415; margin-bottom: 0.4rem;">Document Source</div>', unsafe_allow_html=True)
    input_mode = st.radio(
        "Source Mode",
        ["Curated Presentation Presets", "Upload Custom Document"],
        label_visibility="collapsed"
    )

    selected_image: Optional[Image.Image] = None
    image_name = ""

    if input_mode == "Curated Presentation Presets":
        preset_choice = st.selectbox(
            "Select Benchmark Sample:",
            list(PRESETS.keys()),
            index=0
        )
        preset_rel_path = PRESETS[preset_choice]
        preset_full_path = repo_root / preset_rel_path
        if preset_full_path.is_file():
            selected_image = Image.open(preset_full_path).convert("RGB")
            image_name = preset_choice
        else:
            st.warning(f"File not found: {preset_rel_path}")
    else:
        uploaded_file = st.file_uploader(
            "Upload ID Document Image",
            type=["jpg", "jpeg", "png"],
            help="High-resolution image (Aadhaar, PAN, DL, Passport, Voter ID)"
        )
        if uploaded_file is not None:
            selected_image = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name

    st.markdown('<hr style="border: none; height: 1px; background: rgba(148,152,178,0.25); margin: 1.4rem 0;">', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 0.8rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #9498B2; margin-bottom: 0.4rem;">Detection Calibration</div>', unsafe_allow_html=True)
    show_overlay = st.checkbox("Overlay OCR Typography Bounding Boxes", value=True)
    calib_threshold = st.slider(
        "Tamper Decision Threshold",
        min_value=0.40,
        max_value=0.85,
        value=0.60,
        step=0.05,
        help="Calibrated decision threshold for classifying documents as FORGED vs GENUINE on synthetic samples."
    )

    default_ckpt = str(repo_root / "models" / "visual" / "best_visual_model.pth")
    ckpt_path = default_ckpt

    st.markdown('<hr style="border: none; height: 1px; background: rgba(148,152,178,0.25); margin: 1.4rem 0;">', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.78rem; line-height: 1.6; color: #525355;">
        <span style="font-weight: 600; color: #141415;">Research Attribution</span><br>
        • Visual Branch & Integration: <b>Shivalika</b><br>
        • OCR Layout Profiling: <b>Vidyashree</b><br>
        • Multimodal Fusion & LODTO: <b>Liza</b>
    </div>
    """, unsafe_allow_html=True)


# ── Editorial Masthead (Header) ───────────────────────────────────────────────
st.markdown('<div class="editorial-kicker">Forensic Document Examination · Research Benchmark</div>', unsafe_allow_html=True)
st.markdown('<div class="editorial-headline">Multimodal Identity & Document Screening</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="editorial-subdeck">'
    'An investigation into deep visual spatial embeddings, morphological typography consistency, '
    'and multimodal feature synthesis for cross-document identity verification across 13 Indian languages.'
    '</div>',
    unsafe_allow_html=True
)


# ── Editorial Navigation Bar ──────────────────────────────────────────────────
tab_demo, tab_features, tab_architecture, tab_lodto, tab_dashboard = st.tabs([
    "Live Screening Narrative",
    "27-D Feature Synthesis",
    "Architecture & Teammates",
    "LODTO Generalization Study",
    "Training Results Dashboard"
])


# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: LIVE SCREENING DEMO (EDITORIAL ASYMMETRICAL 70/30 LAYOUT)
# ═════════════════════════════════════════════════════════════════════════════
with tab_demo:
    if selected_image is None:
        st.info("Please select a document preset from the left panel or upload an ID image to begin analysis.")
    else:
        vis_model = load_cached_visual_model(ckpt_path)

        temp_img_path = repo_root / "temp_eval_input.png"
        selected_image.save(temp_img_path)

        with st.spinner("Extracting visual latents, OCR morphology, and multimodal consensus..."):
            vis_res = predict_visual(selected_image, model=vis_model)
            ocr_pred = predict_ocr(str(temp_img_path))
            fusion_res = verify_id_multimodal(str(temp_img_path))
            gradcam_bgr, raw_heatmap = visualize_gradcam_with_heatmap(selected_image, vis_model)

            p_vis_f = float(vis_res.get("probability_forged", 0.5))
            p_ocr_f = float(ocr_pred.get("probability_forged", 0.5))
            p_mlp_f = float(fusion_res.get("forged_probability_pct", 50.0)) / 100.0

            p_combined_forged = (0.40 * p_vis_f) + (0.30 * p_ocr_f) + (0.30 * p_mlp_f)
            fused_is_forged = p_combined_forged >= calib_threshold
            final_verdict_str = "FORGED" if fused_is_forged else "GENUINE"
            certainty_pct = round((p_combined_forged if fused_is_forged else (1.0 - p_combined_forged)) * 100, 1)

            xai_info = generate_xai_explanation(
                raw_heatmap,
                vis_res=vis_res,
                ocr_res=ocr_pred,
                final_verdict=final_verdict_str
            )

        if temp_img_path.is_file():
            try:
                temp_img_path.unlink()
            except Exception:
                pass

        # Asymmetrical Layout: Left 68% (Narrative), Right 32% (Editorial Analysis)
        col_narrative, col_sidebar = st.columns([0.68, 0.32], gap="large")

        # ── LEFT COLUMN (68%): Primary Document & Explainability Narrative ────
        with col_narrative:
            st.markdown(f"""
            <div class="doc-hero-wrapper">
                <div class="doc-hero-kicker">Target Document</div>
                <div class="doc-hero-title">{image_name}</div>
            </div>
            """, unsafe_allow_html=True)

            img_np = np.array(selected_image)
            boxes = ocr_pred.get("boxes", [])
            texts = ocr_pred.get("texts", [])
            if show_overlay and len(boxes) > 0:
                display_img = draw_bounding_boxes(img_np, boxes, texts)
                caption_text = f"Figure 1: Isolated typographic regions ({len(boxes)} fields mapped via {ocr_pred.get('engine', 'OCR')})"
            else:
                display_img = img_np
                caption_text = "Figure 1: Unannotated source document canvas"

            st.image(display_img, caption=caption_text, use_container_width=True)

            st.markdown('<div class="editorial-divider"></div>', unsafe_allow_html=True)

            # Grad-CAM Heatmap Hero Block
            st.markdown("""
            <div style="margin-top: 1.5rem; margin-bottom: 0.8rem;">
                <div class="doc-hero-kicker">Explainable AI & Forensic Saliency</div>
                <div style="font-family: 'Newsreader', serif; font-size: 1.45rem; font-weight: 600; color: #141415;">
                    Grad-CAM Activation Saliency
                </div>
                <div style="font-size: 0.88rem; color: #525355; margin-top: 0.3rem;">
                    Back-propagated feature activations from the penultimate EfficientNet-B0 MBConv layer, highlighting regions of anomalous edge energy.
                </div>
            </div>
            """, unsafe_allow_html=True)

            gradcam_rgb = bgr_to_rgb(gradcam_bgr)
            st.image(
                gradcam_rgb,
                caption="Figure 2: Grad-CAM attention heatmap overlay — warmer hues denote elevated neural attribution",
                use_container_width=True
            )

            # Editorial Forensic Narrative Text Box
            st.markdown(f"""
            <div class="editorial-card" style="margin-top: 1.5rem;">
                <div class="doc-hero-kicker">Forensic Attribution & Interpretability (XAI)</div>
                <div style="font-size: 1.05rem; font-weight: 600; color: #141415; margin-bottom: 0.4rem;">
                    Focus Area: {xai_info['hotspot_region']} <span style="font-weight: 400; color: #9498B2; font-size: 0.85rem;">({xai_info['hotspot_location']} quadrant · Peak Saliency {xai_info['peak_intensity']*100:.1f}%)</span>
                </div>
                <div style="font-size: 0.88rem; line-height: 1.6; color: #525355; margin-bottom: 1rem;">
                    {xai_info['zone_description']}
                </div>
                <div style="border-top: 1px solid rgba(148, 152, 178, 0.25); padding-top: 0.85rem;">
                    <div style="font-size: 0.82rem; font-weight: 600; color: #141415; margin-bottom: 0.3rem; text-transform: uppercase; letter-spacing: 0.05em;">
                        Observations & Anomaly Triggers
                    </div>
            """, unsafe_allow_html=True)

            for trg in xai_info["triggers"]:
                st.markdown(f'<div class="forensic-note-item">{trg}</div>', unsafe_allow_html=True)

            st.markdown(f"""
                </div>
                <div style="margin-top: 0.85rem; font-size: 0.84rem; line-height: 1.55; color: #525355; background: rgba(254, 254, 254, 0.7); padding: 12px 16px; border-radius: 8px;">
                    <span style="font-weight: 600; color: #141415;">Resolution:</span> {xai_info['why_resolved']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── RIGHT COLUMN (32%): Editorial Magazine Sidebar Analysis ───────────
        with col_sidebar:
            # 1. Editorial Insight Verdict Panel
            verdict_deck = (
                "Supporting evidence detected across Visual, OCR and Attention streams."
                if fused_is_forged
                else "Document conforms to authentic structural, typographic, and visual tolerances."
            )

            st.markdown(f"""
            <div class="verdict-panel">
                <div class="verdict-kicker">Forgery Assessment</div>
                <div class="verdict-title">{final_verdict_str}</div>
                <div class="verdict-meta">{certainty_pct}% Confidence · Multimodal Consensus</div>
                <div class="verdict-summary">{verdict_deck}</div>
            </div>
            """, unsafe_allow_html=True)

            # 2. Individual Model Stream Outputs (Typography Metric Blocks)
            st.markdown('<div class="doc-hero-kicker" style="margin-bottom: 0.5rem;">Stream Breakdown</div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="editorial-card" style="padding: 16px 20px; margin-bottom: 1.2rem;">
            """, unsafe_allow_html=True)

            vis_state = vis_res["prediction"].capitalize()
            vis_conf = f"{vis_res['confidence']*100:.1f}%"
            st.markdown(f"""
                <div class="stream-row">
                    <div>
                        <span class="stream-name">Visual CNN</span>
                        <span class="stream-author">(Shivalika)</span>
                    </div>
                    <div class="stream-val">{vis_state} · {vis_conf}</div>
                </div>
            """, unsafe_allow_html=True)

            ocr_state = ocr_pred["prediction"].capitalize()
            ocr_conf = f"{ocr_pred['confidence']*100:.1f}%"
            st.markdown(f"""
                <div class="stream-row">
                    <div>
                        <span class="stream-name">OCR Typography</span>
                        <span class="stream-author">(Vidyaaa)</span>
                    </div>
                    <div class="stream-val">{ocr_state} · {ocr_conf}</div>
                </div>
            """, unsafe_allow_html=True)

            fused_state = "Forged" if fusion_res.get("verdict") == "FORGED" else "Genuine"
            fused_conf = f"{fusion_res.get('confidence_pct', 0.0):.1f}%"
            st.markdown(f"""
                <div class="stream-row" style="border-bottom: none;">
                    <div>
                        <span class="stream-name">27-D MLP Fusion</span>
                        <span class="stream-author">(Liza)</span>
                    </div>
                    <div class="stream-val">{fused_state} · {fused_conf}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 3. Micro Metrics Strip
            st.markdown('<div class="doc-hero-kicker" style="margin-bottom: 0.5rem;">Document Metrics</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="editorial-card" style="padding: 16px 20px; margin-bottom: 1.2rem;">
                <div class="stream-row">
                    <span class="stream-name" style="font-weight: 500; color: #525355;">Isolated Text Regions</span>
                    <span class="stream-val">{fusion_res.get('text_boxes_detected', 0)}</span>
                </div>
                <div class="stream-row">
                    <span class="stream-name" style="font-weight: 500; color: #525355;">Mean Character Confidence</span>
                    <span class="stream-val">{fusion_res.get('mean_ocr_conf', 0.0):.3f}</span>
                </div>
                <div class="stream-row" style="border-bottom: none;">
                    <span class="stream-name" style="font-weight: 500; color: #525355;">Computed Tamper Risk</span>
                    <span class="stream-val">{fusion_res.get('forged_probability_pct', 0.0):.1f}%</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # 4. Forensic Notes
            st.markdown('<div class="doc-hero-kicker" style="margin-bottom: 0.5rem;">Analytical Notes</div>', unsafe_allow_html=True)
            st.markdown('<div class="editorial-card" style="padding: 18px 20px;">', unsafe_allow_html=True)
            for clue in fusion_res.get("diagnostics", []):
                st.markdown(f'<div class="forensic-note-item">{clue}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # 5. Export Audit Report
            audit_report = {
                "document_name": image_name,
                "final_multimodal_verdict": final_verdict_str,
                "consensus_certainty_pct": certainty_pct,
                "computed_tamper_probability_pct": fusion_res.get("forged_probability_pct"),
                "individual_streams": {
                    "visual_stream": {
                        "contributor": "Shivalika",
                        "architecture": "EfficientNet-B0 + Grad-CAM",
                        "verdict": vis_res["prediction"].upper(),
                        "confidence": round(vis_res["confidence"], 4)
                    },
                    "ocr_stream": {
                        "contributor": "Vidyaaa",
                        "architecture": "PaddleOCR + 11-D Spatial Profiling",
                        "verdict": ocr_pred["prediction"].upper(),
                        "confidence": round(ocr_pred["confidence"], 4)
                    },
                    "multimodal_fusion": {
                        "contributor": "Liza",
                        "architecture": "27-D Concatenation + 2-Layer MLP",
                        "verdict": fusion_res.get("verdict"),
                        "certainty": fusion_res.get("confidence_pct")
                    }
                },
                "explainable_ai": {
                    "hotspot_region": xai_info.get("hotspot_region"),
                    "quadrant": xai_info.get("hotspot_location"),
                    "peak_saliency": xai_info.get("peak_intensity"),
                    "saliency_spread_pct": xai_info.get("saliency_spread_pct"),
                    "triggers": xai_info.get("triggers"),
                    "resolution": xai_info.get("why_resolved")
                }
            }

            st.download_button(
                label="Export Audit Dossier (JSON)",
                data=json.dumps(audit_report, indent=2),
                file_name=f"forensic_audit_{image_name.replace(' ', '_').lower()}.json",
                mime="application/json"
            )


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: 27-D MULTIMODAL FEATURE VECTOR
# ═════════════════════════════════════════════════════════════════════════════
with tab_features:
    st.markdown('<div class="editorial-kicker">Feature Space Architecture</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-headline" style="font-size: 2.2rem;">27-Dimensional Multimodal Vector</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="editorial-subdeck">'
        'A unified numerical embedding synthesizing 16 visual forensic descriptors (spatial CNN latents, '
        'Laplacian edge variance, JPEG compression residuals) and 11 OCR layout indicators.'
        '</div>',
        unsafe_allow_html=True
    )

    if selected_image is not None and 'fusion_res' in locals():
        fdict = fusion_res.get("features_dict", {})

        col_f1, col_f2 = st.columns(2, gap="large")

        with col_f1:
            st.markdown('<div class="doc-hero-title" style="font-size: 1.3rem;">Visual Descriptors (16-D)</div>', unsafe_allow_html=True)
            vis_rows = []
            vis_desc = {
                "cnn_feat_1": "EfficientNet deep spatial latent bin 1",
                "cnn_feat_2": "EfficientNet deep spatial latent bin 2",
                "cnn_feat_3": "EfficientNet deep spatial latent bin 3",
                "cnn_feat_4": "EfficientNet deep spatial latent bin 4",
                "cnn_feat_5": "EfficientNet deep spatial latent bin 5",
                "cnn_feat_6": "EfficientNet deep spatial latent bin 6",
                "cnn_feat_7": "EfficientNet deep spatial latent bin 7",
                "cnn_feat_8": "EfficientNet deep spatial latent bin 8",
                "laplacian_var": "Laplacian variance (detects boundary smoothing & blur)",
                "laplacian_mean": "Mean Laplacian magnitude (edge sharpness index)",
                "gradient_mean": "Mean Sobel gradient magnitude across both axes",
                "gradient_std": "Standard deviation of Sobel gradient (edge continuity)",
                "high_freq_ratio": "Fast Fourier Transform high-frequency ratio (splicing seams)",
                "ela_mean": "Error Level Analysis mean JPEG residual",
                "ela_std": "Error Level Analysis standard deviation of residuals",
                "color_std_ratio": "Blue-to-red channel variance ratio (spectral uniformity)"
            }
            for k in VISUAL_FEATURE_NAMES:
                vis_rows.append({
                    "Descriptor": k,
                    "Magnitude": round(float(fdict.get(k, 0.0)), 4),
                    "Forensic Intuition": vis_desc.get(k, "")
                })
            st.dataframe(pd.DataFrame(vis_rows), use_container_width=True, hide_index=True)

        with col_f2:
            st.markdown('<div class="doc-hero-title" style="font-size: 1.3rem;">OCR Typography Descriptors (11-D)</div>', unsafe_allow_html=True)
            ocr_rows = []
            ocr_desc = {
                "mean_confidence": "Mean character recognition confidence across fields",
                "min_confidence": "Lowest single-field confidence observed",
                "max_confidence": "Highest single-field confidence observed",
                "std_confidence": "Confidence variance (flags spliced vs native text)",
                "low_conf_count": "Count of text boxes below 0.85 confidence",
                "low_conf_ratio": "Proportion of low-confidence text regions",
                "num_text_boxes": "Total isolated typographic regions",
                "text_area_ratio": "Text area relative to complete card canvas area",
                "mean_aspect_ratio": "Average text box width-to-height ratio",
                "box_height_std": "Standard deviation of line heights (font inconsistency)",
                "mean_char_density": "Character count per square unit of text box area"
            }
            for k in OCR_FEATURE_NAMES:
                ocr_rows.append({
                    "Descriptor": k,
                    "Magnitude": round(float(fdict.get(k, 0.0)), 4),
                    "Forensic Intuition": ocr_desc.get(k, "")
                })
            st.dataframe(pd.DataFrame(ocr_rows), use_container_width=True, hide_index=True)
    else:
        st.info("Select or upload a document in the sidebar to inspect its extracted 27-D multimodal feature vector.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: SYSTEM ARCHITECTURE & ATTRIBUTION
# ═════════════════════════════════════════════════════════════════════════════
with tab_architecture:
    st.markdown('<div class="editorial-kicker">Methodology & Attribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-headline" style="font-size: 2.2rem;">System Architecture</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="editorial-subdeck">'
        'Collaborative research pipeline integrating computer vision, optical character recognition, '
        'and multimodal machine learning across three specialized streams.'
        '</div>',
        unsafe_allow_html=True
    )

    col_arch1, col_arch2 = st.columns([0.60, 0.40], gap="large")

    with col_arch1:
        st.markdown(r"""
        ### Multi-Stream Investigation Pipeline

        1. **Dual Input Ingestion**  
           The candidate document is preprocessed into normalized $(224 \times 224)$ tensors for deep convolutional feature extraction while concurrently passing through an adaptive morphological binarization engine.

        2. **Visual Feature Extraction (Shivalika)**  
           A fine-tuned **EfficientNet-B0** vision backbone extracts penultimate 1280-D spatial embeddings. Contrastive Grad-CAM back-propagates gradients through MBConv blocks to isolate suspicious high-frequency anomalies.

        3. **OCR Typography Profiling (Vidyaaa)**  
           Extracts character recognition confidences, font height variance, aspect ratios, and spatial density metrics across all isolated text boxes, formulating an 11-D typographic profile.

        4. **Multimodal Fusion & Ensemble (Liza)**  
           Concatenates visual and OCR metrics into a 27-D vector $\mathbf{x}_{\text{fused}} \in \mathbb{R}^{27}$, standardized via a learned reference distribution, and evaluated via a 2-layer neural network with weighted consensus arbitration.
        """)

    with col_arch2:
        st.markdown("""
        <div class="editorial-card">
            <div class="doc-hero-kicker">Teammate Roles</div>
            <div style="margin-bottom: 1.2rem;">
                <div style="font-weight: 600; color: #141415;">Shivalika — Visual Lead</div>
                <div style="font-size: 0.85rem; color: #525355;">PyTorch EfficientNet-B0 architecture, Grad-CAM XAI saliency map generator, Streamlit presentation platform.</div>
            </div>
            <div style="margin-bottom: 1.2rem;">
                <div style="font-weight: 600; color: #141415;">Vidyaaa — OCR Lead</div>
                <div style="font-size: 0.85rem; color: #525355;">PaddleOCR text recognition engine, 11-D document layout metrics, morphological fallback analyzer.</div>
            </div>
            <div>
                <div style="font-weight: 600; color: #141415;">Liza — Fusion Lead</div>
                <div style="font-size: 0.85rem; color: #525355;">27-D Multimodal feature fusion MLP, leave-one-document-type-out (LODTO) cross-document generalization study.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: LODTO GENERALIZATION STUDY
# ═════════════════════════════════════════════════════════════════════════════
with tab_lodto:
    st.markdown('<div class="editorial-kicker">Empirical Generalization Study</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-headline" style="font-size: 2.2rem;">Leave-One-Document-Type-Out (LODTO)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="editorial-subdeck">'
        'Assessing cross-document generalization resilience across 5 Indian identity templates '
        '(Aadhaar, Driving Licence, PAN Card, Passport, Voter ID). In each rotation, the model trains '
        'on 4 known document categories and is evaluated strictly on an unseen 5th category.'
        '</div>',
        unsafe_allow_html=True
    )

    rep_dir = repo_root / "reports"
    img_gap = rep_dir / "lodto_generalization_gap.png"
    img_folds = rep_dir / "lodto_fold_breakdown.png"
    img_roc = rep_dir / "roc_curves_comparison.png"

    col_l1, col_l2 = st.columns(2, gap="large")
    with col_l1:
        if img_gap.is_file():
            st.image(str(img_gap), caption="Figure 1: Generalization Gap across Seen vs. Unseen Templates", use_container_width=True)
    with col_l2:
        if img_roc.is_file():
            st.image(str(img_roc), caption="Figure 2: Receiver Operating Characteristic (ROC) Comparison", use_container_width=True)

    if img_folds.is_file():
        st.image(str(img_folds), caption="Figure 3: Fold-by-Fold Performance Breakdown Across Indian Document Categories", use_container_width=True)

    lodto_csv = repo_root / "lodto_experiment_results.csv"
    if lodto_csv.is_file():
        st.markdown('<div class="editorial-divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="doc-hero-title" style="font-size: 1.3rem;">Empirical Benchmark Data</div>', unsafe_allow_html=True)
        df_lodto = pd.read_csv(lodto_csv)
        st.dataframe(df_lodto, use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 5: TRAINING RESULTS DASHBOARD (SOURCE: GOOGLE COLAB CSV OUTPUTS)
# ═════════════════════════════════════════════════════════════════════════════
with tab_dashboard:
    st.markdown('<div class="editorial-kicker">Measured Experiment Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="editorial-headline" style="font-size: 2.2rem;">Training Results Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        "These values are loaded from the Google Colab experiment CSV files committed "
        "under `results/retrained_current_dataset/`. LODTO tests each document type as "
        "unseen after training on the other four types."
    )

    result_dir = repo_root / "results" / "retrained_current_dataset"
    lodto_csv = result_dir / "lodto_results.csv"
    comparison_csv = result_dir / "model_comparison.csv"

    if lodto_csv.is_file() and comparison_csv.is_file():
        lodto_df = pd.read_csv(lodto_csv)
        comparison_df = pd.read_csv(comparison_csv)

        mean_acc = lodto_df["Accuracy"].mean() * 100
        mean_f1 = lodto_df["F1-score"].mean() * 100
        best_doc = lodto_df.loc[lodto_df["Accuracy"].idxmax(), "Held-out document type"]

        c1, c2, c3 = st.columns(3)
        c1.metric("LODTO document types", int(lodto_df["Held-out document type"].nunique()))
        c2.metric("Mean unseen accuracy", f"{mean_acc:.2f}%")
        c3.metric("Mean unseen F1", f"{mean_f1:.2f}%")

        st.markdown("### LODTO unseen-document results")
        st.dataframe(lodto_df, use_container_width=True, hide_index=True)
        st.caption(f"Highest unseen accuracy in this run: {best_doc}.")

        st.markdown("### CV, OCR and fusion comparison")
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        fusion = comparison_df[comparison_df["Model"].astype(str).str.contains("fusion", case=False)]
        if not fusion.empty:
            row = fusion.iloc[0]
            st.success(
                f"CV + OCR fusion test accuracy: {row['Accuracy']*100:.2f}% "
                f"| F1-score: {row['F1-score']*100:.2f}% "
                f"| AUROC: {row['AUROC']*100:.2f}%"
            )

        graph_dir = result_dir / "graphs"
        for filename, caption in [
            ("lodto_unseen_accuracy.png", "Figure: LODTO unseen-document accuracy by document type"),
            ("lodto_unseen_f1.png", "Figure: LODTO unseen-document F1-score by document type"),
            ("lodto_unseen_auroc.png", "Figure: LODTO unseen-document AUROC by document type"),
            ("model_comparison_accuracy.png", "Figure: Accuracy comparison of CV, OCR and CV + OCR fusion"),
        ]:
            image_path = graph_dir / filename
            if image_path.is_file():
                st.image(str(image_path), caption=caption, use_container_width=True)

        st.caption(
            "Inference speed is not reported because no measured inference-time results "
            "were included in the Google Colab output files."
        )
    else:
        st.warning(
            "Google Colab result files not found. Ensure lodto_results.csv and "
            "model_comparison.csv are inside results/retrained_current_dataset/."
        )
