"""
src/explainability.py — Real Grad-CAM Heatmap Generation for Forensic Inspection
Hooks into the EfficientNet-B0 penultimate conv layer to produce
saliency maps showing exact image regions driving forgery detection decisions.

Public API
----------
visualize_gradcam(image_input, model) -> np.ndarray   (BGR overlay)
generate_and_save_gradcam(image_path, output_path) -> str
"""

import os
import sys
from pathlib import Path
from typing import Any

# Ensure repo root is importable
project_root = Path(__file__).resolve().parent.parent
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
        self.gradients = None
        self.activations = None

        # Register forward and backward hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class=1):
        self.model.eval()
        self.model.zero_grad()

        # Ensure target layer parameters require gradients so backward hook fires
        orig_grad_states = []
        for param in self.target_layer.parameters():
            orig_grad_states.append(param.requires_grad)
            param.requires_grad = True

        out = self.model(input_tensor)
        logits = out["logits"]
        score = logits[0, target_class]
        score.backward()

        # Restore original requires_grad states
        for param, state in zip(self.target_layer.parameters(), orig_grad_states):
            param.requires_grad = state

        if self.gradients is None or self.activations is None:
            # Fallback for frozen/constant activations
            return np.zeros((input_tensor.shape[2], input_tensor.shape[3]), dtype=np.float32)

        # Global average pool gradients
        pooled_grads = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        # Weight the channels by gradients
        for i in range(activations.shape[0]):
            activations[i] *= pooled_grads[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap


# ---------------------------------------------------------------------------
# Helper: pick the best target layer regardless of backbone variant
# ---------------------------------------------------------------------------

def _get_target_layer(model):
    """Return the best convolutional layer for Grad-CAM.
    Works for both timm (conv_head) and torchvision (Sequential features).
    """
    # timm-style backbone
    if hasattr(model.backbone, "conv_head"):
        return model.backbone.conv_head

    # torchvision EfficientNet – backbone is nn.Sequential (the .features block)
    children = list(model.backbone.children())
    if children:
        return children[-1]

    # Absolute fallback
    return model.backbone


# ---------------------------------------------------------------------------
# Helper: convert any supported image input to a BGR NumPy array
# ---------------------------------------------------------------------------

def _to_bgr_numpy(image_input) -> np.ndarray:
    if isinstance(image_input, np.ndarray):
        return cv2.cvtColor(image_input, cv2.COLOR_RGB2BGR)
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is not None:
            return img
    if isinstance(image_input, Image.Image):
        return cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
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
    input_tensor = _preprocess_input_image(image_input)
    target_layer = _get_target_layer(model)

    grad_cam = GradCAM(model, target_layer)
    heatmap = grad_cam.generate(input_tensor, target_class=1)

    orig_bgr = _to_bgr_numpy(image_input)
    h, w = orig_bgr.shape[:2]

    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(orig_bgr, 0.6, heatmap_colored, 0.4, 0)
    return overlay


def generate_and_save_gradcam(image_path: str, output_path: str) -> str:
    """Generate Grad-CAM and save the overlay to disk."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    model = get_default_visual_model()
    overlay = visualize_gradcam(image_path, model)
    cv2.imwrite(str(out_p), overlay)
    return str(out_p)


if __name__ == "__main__":
    test_img = r"D:\visual\indian_multilingual_id_dataset_ALL\indian_multilingual_id_dataset\assamese\aadhaar\forged\0000.jpg"
    out_test = r"D:\visual\results\explainability\gradcam_sample_forged_aadhaar.png"
    save_path = generate_and_save_gradcam(test_img, out_test)
    print(f"Grad-CAM generated successfully: {save_path}")
