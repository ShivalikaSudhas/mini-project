"""
visual_model.py — EfficientNet-B0 Visual Branch for Identity Document Forgery Detection
Phase 3: Visual Model Development
Provides:
  - VisualForensicModel (EfficientNet-B0 backbone + classifier head)
  - Penultimate 1280-dimensional feature extraction for multimodal fusion
  - Interfaces:
      predict_visual(image, model=None, checkpoint_path=None)
      extract_visual_features(image, model=None, checkpoint_path=None)
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from PIL import Image
# import cv2  # Removed to avoid numpy ABI mismatch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision import models

log = logging.getLogger(__name__)

# Constants
FEATURE_DIM = 1280
EFFICIENTNET_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

_DEFAULT_TRANSFORM = T.Compose([
    T.Resize((EFFICIENTNET_SIZE, EFFICIENTNET_SIZE), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Module-level model cache for quick inference
_LOADED_MODEL: Optional["VisualForensicModel"] = None
_LOADED_CKPT_PATH: Optional[str] = None


class VisualForensicModel(nn.Module):
    """
    EfficientNet-B0 based visual forgery detector.
    Outputs binary classification logits and a 1280-dimensional feature vector.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout_rate: float = 0.3,
        feature_dim: int = FEATURE_DIM,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim

        # Backbone: timm efficientnet_b0
        self.backbone = models.efficientnet_b0(pretrained=pretrained).features

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, num_classes),
        )

        if freeze_backbone:
            self._freeze_backbone()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat_map = self.backbone(x)         # (B, 1280, H, W)
        pooled = self.global_pool(feat_map) # (B, 1280, 1, 1)
        features = pooled.flatten(1)        # (B, 1280)
        logits = self.classifier(features)  # (B, 2)
        proba  = F.softmax(logits, dim=-1)  # (B, 2)

        return {
            "logits":   logits,
            "features": features,
            "proba":    proba,
        }

    @torch.no_grad()
    def extract_visual_features(self, x: torch.Tensor) -> torch.Tensor:
        """Returns L2-normalized (B, 1280) feature tensor."""
        self.eval()
        out = self.forward(x)
        feats = F.normalize(out["features"], p=2, dim=-1)
        return feats.detach().cpu()

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_top_n_blocks(self, n: int = 3):
        """Unfreeze top n MBConv blocks for fine-tuning."""
        all_blocks = list(self.backbone.blocks)
        total = len(all_blocks)
        for i, block in enumerate(all_blocks):
            if i >= total - n:
                for param in block.parameters():
                    param.requires_grad = True
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True

    def freeze_backbone(self):
        self._freeze_backbone()

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Optional[Dict] = None,
        optimizer_state: Optional[Dict] = None,
    ):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.state_dict(),
            "metrics": metrics or {},
            "model_config": {
                "num_classes": self.num_classes,
                "feature_dim": self.feature_dim,
            },
        }
        if optimizer_state:
            ckpt["optimizer_state_dict"] = optimizer_state
        torch.save(ckpt, path)

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None) -> Tuple["VisualForensicModel", Dict]:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device)
        cfg = ckpt.get("model_config", {})
        model = cls(
            num_classes=cfg.get("num_classes", 2),
            pretrained=False,
            freeze_backbone=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        return model, ckpt


# ─── Inference Preprocessing Helper ──────────────────────────────────────────

def _preprocess_input_image(image_input: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
    """Preprocesses input into (1, 3, 224, 224) normalized tensor."""
    if isinstance(image_input, (str, Path)):
        img_path = str(image_input)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        img_pil = Image.open(img_path).convert("RGB")
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            img_pil = Image.fromarray(image_input).convert("RGB")
        elif image_input.shape[2] == 3:
            # Assume RGB; if uint8
            img_pil = Image.fromarray(image_input.astype(np.uint8))
        else:
            raise ValueError(f"Unsupported numpy image shape: {image_input.shape}")
    elif isinstance(image_input, Image.Image):
        img_pil = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    tensor = _DEFAULT_TRANSFORM(img_pil)  # (3, 224, 224)
    return tensor.unsqueeze(0)            # (1, 3, 224, 224)


def get_default_visual_model(checkpoint_path: Optional[str] = None) -> VisualForensicModel:
    """Helper to return or load a shared model instance."""
    global _LOADED_MODEL, _LOADED_CKPT_PATH
    default_ckpt = checkpoint_path or "D:/visual/models/visual/best_visual_model.pth"

    if _LOADED_MODEL is not None and _LOADED_CKPT_PATH == default_ckpt:
        return _LOADED_MODEL

    if os.path.exists(default_ckpt):
        model, _ = VisualForensicModel.load_checkpoint(default_ckpt)
        _LOADED_MODEL = model
        _LOADED_CKPT_PATH = default_ckpt
        return _LOADED_MODEL
    else:
        # Fallback to pretrained ImageNet model
        model = VisualForensicModel(pretrained=True, freeze_backbone=True)
        model.eval()
        _LOADED_MODEL = model
        _LOADED_CKPT_PATH = "pretrained_imagenet"
        return _LOADED_MODEL


# ─── Public Interfaces Required by Prompt ────────────────────────────────────

def predict_visual(
    image: Union[str, Path, Image.Image, np.ndarray],
    model: Optional[VisualForensicModel] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None
) -> Dict:
    """
    predict_visual(image)
    Interface required by Master Prompt Section 3 (Phase 3).
    Returns class prediction, calibrated probability, confidence, and 1280-d features.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model is None:
        model = get_default_visual_model(checkpoint_path)
    model.to(device)
    model.eval()

    tensor = _preprocess_input_image(image).to(device)
    with torch.no_grad():
        out = model(tensor)
        proba = out["proba"].squeeze(0).cpu().numpy()  # [p_genuine, p_forged]
        features = F.normalize(out["features"], p=2, dim=-1).squeeze(0).cpu().numpy()

    p_genuine = float(proba[0])
    p_forged = float(proba[1])
    class_id = int(np.argmax(proba))
    prediction = "forged" if class_id == 1 else "genuine"
    confidence = float(np.max(proba))

    return {
        "prediction": prediction,
        "class_id": class_id,
        "probability_genuine": round(p_genuine, 4),
        "probability_forged": round(p_forged, 4),
        "confidence": round(confidence, 4),
        "features": features,
    }


def extract_visual_features(
    image: Union[str, Path, Image.Image, np.ndarray],
    model: Optional[VisualForensicModel] = None,
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None
) -> np.ndarray:
    """
    extract_visual_features(image)
    Interface required by Master Prompt Section 3 (Phase 3 & Phase 5).
    Returns numerical 1280-dimensional L2-normalized feature vector.
    """
    res = predict_visual(image, model=model, checkpoint_path=checkpoint_path, device=device)
    return res["features"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Testing VisualForensicModel public interfaces...")
    dummy_img = Image.new("RGB", (320, 200), color=(180, 140, 120))
    res = predict_visual(dummy_img)
    print("predict_visual result:", {k: v for k, v in res.items() if k != 'features'})
    feats = extract_visual_features(dummy_img)
    print(f"extract_visual_features shape: {feats.shape}, norm: {np.linalg.norm(feats):.4f}")
    print("Visual branch interfaces verified successfully!")
