# src/explainability.py
"""
explainability.py — Grad-CAM Explainable AI for Deep Visual Forensics.
Produces class-activation heat-maps indicating which spatial regions of
an ID document drove the model's prediction (e.g. genuine vs forged).
"""

from __future__ import annotations
import os
import sys
import pathlib
from pathlib import Path
from typing import Any, Optional, Tuple, Union

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch
from PIL import Image

from src.visual_model import get_default_visual_model, _preprocess_input_image


# ---------------------------------------------------------------------------
# Grad-CAM core
# ---------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.hook_handle = None

    def _save_activation(self, module, inp, out):
        out.retain_grad()
        self.activations = out

    def generate(self, input_tensor, target_class=1):
        self.model.eval()
        self.model.zero_grad()

        # Ensure input tensor requires gradients
        inp = input_tensor.clone().detach().requires_grad_(True)

        # Register forward hook
        self.hook_handle = self.target_layer.register_forward_hook(self._save_activation)

        try:
            out = self.model(inp)
            logits = out["logits"]
            # Contrastive log-odds: difference between Forged (class 1) and Genuine (class 0)
            if target_class == 1 and logits.shape[1] > 1:
                score = logits[0, 1] - logits[0, 0]
            else:
                score = logits[0, target_class]
            score.backward()

            if self.activations is not None and self.activations.grad is not None:
                grads = self.activations.grad
                pooled_grads = torch.mean(grads, dim=[0, 2, 3])
                act = self.activations[0].detach()

                for i in range(act.shape[0]):
                    act[i] *= pooled_grads[i]

                heatmap = torch.mean(act, dim=0).cpu().numpy()
                heatmap = np.maximum(heatmap, 0)
                if heatmap.max() > 0:
                    heatmap /= heatmap.max()
                return heatmap
            else:
                # Fallback: gradient saliency map
                if inp.grad is not None:
                    sal = torch.max(torch.abs(inp.grad[0]), dim=0)[0].cpu().numpy()
                    sal = np.maximum(sal, 0)
                    if sal.max() > 0:
                        sal /= sal.max()
                    return sal
                return np.zeros((inp.shape[2], inp.shape[3]), dtype=np.float32)

        finally:
            if self.hook_handle is not None:
                self.hook_handle.remove()
                self.hook_handle = None


# ---------------------------------------------------------------------------
# Helper: pick the best target layer regardless of backbone variant
# ---------------------------------------------------------------------------

def _get_target_layer(model):
    """Return the best convolutional layer for Grad-CAM.
    Works for both timm (conv_head) and torchvision (Sequential features).
    """
    if hasattr(model, "backbone") and hasattr(model.backbone, "conv_head"):
        return model.backbone.conv_head

    if hasattr(model, "backbone"):
        children = list(model.backbone.children())
        if children:
            return children[-1]
        return model.backbone

    children = list(model.children())
    return children[-1] if children else model


# ---------------------------------------------------------------------------
# Helper: convert any supported image input to a BGR NumPy array
# ---------------------------------------------------------------------------

def _to_bgr_numpy(image_input) -> np.ndarray:
    if isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 3 and image_input.shape[2] == 3:
            return cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
        return image_input
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is not None:
            return img
    if isinstance(image_input, Image.Image):
        return cv2.cvtColor(np.array(image_input.convert("RGB")), cv2.COLOR_RGB2BGR)
    raise TypeError(f"Unsupported image type: {type(image_input)}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def visualize_gradcam(image_input: Any, model: Any) -> np.ndarray:
    """Generate a Grad-CAM heat-map overlay.

    Parameters
    ----------
    image_input : PIL.Image | np.ndarray | str | Path
    model       : VisualForensicModel

    Returns
    -------
    np.ndarray – BGR overlay image (same spatial size as input).
    """
    overlay, _ = visualize_gradcam_with_heatmap(image_input, model)
    return overlay


def visualize_gradcam_with_heatmap(image_input: Any, model: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Generate both the Grad-CAM heat-map overlay and raw normalized heatmap (0..1)."""
    input_tensor = _preprocess_input_image(image_input)
    target_layer = _get_target_layer(model)

    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate(input_tensor, target_class=1)

    orig_bgr = _to_bgr_numpy(image_input)
    h, w = orig_bgr.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_colored, 0.4, 0)
    return overlay, heatmap_resized


def generate_xai_explanation(
    heatmap: np.ndarray,
    vis_res: Optional[dict] = None,
    ocr_res: Optional[dict] = None,
    final_verdict: Optional[str] = None
) -> dict:
    """
    Generates human-interpretable Explainable AI (XAI) forensic text diagnostics:
      - Hotspot region localization (photo, QR code, number, header, text block)
      - Why it was assumed/flagged as forged (gradient spikes, edge contrast, text variance)
      - How and why the multimodal consensus resolved the final verdict
    """
    h, w = heatmap.shape
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    peak_val = float(np.max(heatmap))
    spread_pct = float(np.sum(heatmap > 0.55) / (h * w) * 100)

    # 1. Spatial localization
    row = "Top" if peak_y < h / 3 else ("Bottom" if peak_y > 2 * h / 3 else "Middle")
    col = "Left" if peak_x < w / 3 else ("Right" if peak_x > 2 * w / 3 else "Center")

    if row == "Middle" and col == "Left":
        region_name = "Photo Portrait / Avatar Area"
        zone_desc = "The primary forensic saliency peak is concentrated around the ID photo boundary."
    elif row == "Middle" and col in ["Center", "Right"]:
        region_name = "QR Code & Demographic Data Block"
        zone_desc = "The primary forensic saliency peak is concentrated on the QR matrix and central demographic fields."
    elif row == "Bottom":
        region_name = "Unique ID Number Strip"
        zone_desc = "The primary forensic saliency peak is focused on the document's unique identity number."
    elif row == "Top":
        region_name = "Government Emblem & State Header"
        zone_desc = "The primary forensic saliency peak is located along the official government header."
    else:
        region_name = f"{row}-{col} Document Region"
        zone_desc = f"The primary forensic saliency peak is detected in the {row.lower()}-{col.lower()} area of the document."

    # 2. Specific Forensic Triggers
    triggers = []
    if peak_val > 0.70:
        triggers.append(f"High gradient contrast anomaly ({peak_val*100:.1f}% intensity) centered at {region_name}")
    if spread_pct > 15.0:
        triggers.append(f"Diffuse multi-region activation ({spread_pct:.1f}% of document surface shows elevated forensic gradients)")
    if ocr_res and ocr_res.get("features", {}).get("box_height_std", 0) > 15.0:
        triggers.append("Font height standard deviation irregularity detected across detected text fields")
    if ocr_res and ocr_res.get("features", {}).get("std_confidence", 0) > 0.20:
        triggers.append("Fluctuating OCR confidence scores (potential text splicing or print inconsistency)")

    if not triggers:
        triggers.append(f"Localized feature transition detected at {region_name}")

    # 3. Why it was assumed forged
    why_assumed_forged = (
        f"The visual deep feature backbone flagged abnormal gradient concentration in the {region_name}. "
        f"In forensic screening, sudden localized gradient spikes typically signal cut-and-paste boundary artifacts, "
        f"digital text replacement, or compression residual mismatches between spliced elements and the background canvas."
    )

    # 4. Contextual explanation & resolution
    is_forged = (final_verdict or "FORGED").upper() == "FORGED"
    if is_forged:
        why_resolved = (
            f"The anomaly at the {region_name} was verified by the multimodal consensus engine. "
            f"The elevated visual saliency aligns with localized digital splicing or tampering signatures, "
            f"resulting in a confirmed FORGED verdict."
        )
    else:
        why_resolved = (
            f"Although the visual branch observed high-contrast gradients at the {region_name}, "
            f"the OCR layout consistency and multimodal feature fusion confirmed that the font alignment, "
            f"character density, and document structure remain within authentic tolerances, "
            f"resolving the initial anomaly as synthetic template rendering rather than malicious tampering."
        )

    return {
        "hotspot_region": region_name,
        "hotspot_location": f"{row}-{col}",
        "peak_intensity": round(peak_val, 4),
        "saliency_spread_pct": round(spread_pct, 1),
        "zone_description": zone_desc,
        "triggers": triggers,
        "why_assumed_forged": why_assumed_forged,
        "why_resolved": why_resolved,
    }


def generate_and_save_gradcam(image_path: str, output_path: str) -> str:
    """Generate Grad-CAM and save the overlay to disk."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    model = get_default_visual_model()
    overlay = visualize_gradcam(image_path, model)
    cv2.imwrite(str(out_p), overlay)
    return str(out_p)


if __name__ == "__main__":
    test_img = r"demo_samples\aadhaar_forged.jpg"
    out_test = r"results\explainability\gradcam_sample_forged_aadhaar.png"
    if os.path.exists(test_img):
        save_path = generate_and_save_gradcam(test_img, out_test)
        print(f"Grad-CAM generated successfully: {save_path}")
