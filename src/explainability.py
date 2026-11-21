"""
src/explainability.py — Real Grad-CAM Heatmap Generation for Forensic Inspection
Hooks into the EfficientNet-B0 penultimate conv_head layer to produce
saliency maps showing exact image regions driving forgery detection decisions.
"""

import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from src.visual_model import get_default_visual_model, _preprocess_input_image

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
        
        out = self.model(input_tensor)
        logits = out["logits"]
        score = logits[0, target_class]
        score.backward()
        
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

def generate_and_save_gradcam(image_path: str, output_path: str) -> str:
    p = Path(image_path)
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    
    model = get_default_visual_model()
    target_layer = model.backbone.conv_head
    grad_cam = GradCAM(model, target_layer)
    
    input_tensor = _preprocess_input_image(str(p))
    heatmap = grad_cam.generate(input_tensor, target_class=1)
    
    # Load original image and overlay heatmap
    orig_img = cv2.imread(str(p))
    h, w = orig_img.shape[:2]
    
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    
    overlay = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)
    cv2.imwrite(str(out_p), overlay)
    return str(out_p)

if __name__ == "__main__":
    test_img = r"D:\visual\indian_multilingual_id_dataset_ALL\indian_multilingual_id_dataset\assamese\aadhaar\forged\0000.jpg"
    out_test = r"D:\visual\results\explainability\gradcam_sample_forged_aadhaar.png"
    save_path = generate_and_save_gradcam(test_img, out_test)
    print(f"Grad-CAM generated successfully: {save_path}")
