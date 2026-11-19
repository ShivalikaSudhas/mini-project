"""
visual_model.py — SIH26188 Indian ID Forgery Detection
=======================================================
EfficientNet-B0 backbone (ImageNet pretrained) adapted for:
  1. Binary classification  : genuine (0) vs forged (1)
  2. Feature extraction     : 1280-dim penultimate-layer vector for fusion

Design choices:
  - Uses `timm` (PyTorch Image Models) for reliable pretrained weights
  - Frozen backbone by default; call unfreeze_top_n_blocks() for fine-tuning
  - Dropout before classifier head to reduce overfitting on small datasets
  - Returns both logits and 1280-d features in every forward pass

Author : Visual Lead / Integration Lead
Project: SIH26188
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

# EfficientNet-B0 penultimate feature dimension (timm default)
FEATURE_DIM = 1280


class VisualForensicModel(nn.Module):
    """
    EfficientNet-B0 based visual forgery detector.

    Parameters
    ----------
    num_classes     : 2  (genuine / forged) — kept as param for flexibility
    pretrained      : load ImageNet weights via timm
    freeze_backbone : freeze all backbone layers (recommended for small datasets)
    dropout_rate    : dropout before the final linear layer
    feature_dim     : output dimension of the backbone (1280 for EfficientNet-B0)
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

        # ── Load EfficientNet-B0 backbone (features only, no head) ───────────
        try:
            import timm
        except ImportError:
            raise ImportError("Install timm: pip install timm")

        # global_pool='' disables timm's built-in pooling; we pool ourselves
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=pretrained,
            num_classes=0,        # remove built-in classifier
            global_pool="",       # disable built-in pooling
        )
        log.info("Loaded EfficientNet-B0 (pretrained=%s)", pretrained)

        # ── Global average pooling ────────────────────────────────────────────
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # (B, 1280, 1, 1) → (B, 1280)

        # ── Classification head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(feature_dim, num_classes),
        )

        # ── Freeze backbone if requested ──────────────────────────────────────
        if freeze_backbone:
            self._freeze_backbone()

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, 3, 224, 224) float32 tensor (ImageNet-normalised)

        Returns
        -------
        dict with keys:
          "logits"   : (B, 2)    raw class scores
          "features" : (B, 1280) penultimate-layer embedding for fusion
          "proba"    : (B, 2)    softmax probabilities
        """
        # ── Backbone feature map ─────────────────────────────────────────────
        feat_map = self.backbone(x)           # (B, 1280, H, W)

        # ── Global average pool ──────────────────────────────────────────────
        pooled = self.global_pool(feat_map)   # (B, 1280, 1, 1)
        features = pooled.flatten(1)          # (B, 1280)

        # ── Classifier ───────────────────────────────────────────────────────
        logits = self.classifier(features)    # (B, 2)
        proba  = F.softmax(logits, dim=-1)    # (B, 2)

        return {
            "logits":   logits,
            "features": features,
            "proba":    proba,
        }

    # ── Feature-only extraction (for fusion module) ───────────────────────────

    @torch.no_grad()
    def extract_visual_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a detached (B, 1280) feature tensor.
        Use this interface in the fusion module — do NOT call forward().

        Interface contract (agreed with OCR / fusion teammates):
          Input : (B, 3, 224, 224) float32 — ImageNet normalised
          Output: (B, 1280)        float32 — L2-normalised feature vector
        """
        self.eval()
        out = self.forward(x)
        feats = out["features"]
        # L2-normalise so fusion is scale-independent
        feats = F.normalize(feats, p=2, dim=-1)
        return feats.detach().cpu()

    # ── Backbone freezing / unfreezing ────────────────────────────────────────

    def _freeze_backbone(self):
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        log.info("Backbone frozen — only classifier head is trainable.")

    def unfreeze_top_n_blocks(self, n: int = 3):
        """
        Unfreeze the last n MBConv blocks of EfficientNet-B0 for fine-tuning.
        Call this after the initial frozen-head training phase converges.

        EfficientNet-B0 has 9 MBConv blocks (blocks 0–8).
        Unfreezing the last 3 (blocks 6–8) is a safe starting point.
        """
        # timm names the blocks as backbone.blocks[i]
        all_blocks = list(self.backbone.blocks)
        total = len(all_blocks)
        for i, block in enumerate(all_blocks):
            if i >= total - n:
                for param in block.parameters():
                    param.requires_grad = True
        # Also unfreeze the final conv layer
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True
        log.info("Unfrozen last %d blocks + conv_head for fine-tuning.", n)

    def freeze_backbone(self):
        """Re-freeze entire backbone (undo unfreeze_top_n_blocks)."""
        self._freeze_backbone()

    # ── Parameter summary ─────────────────────────────────────────────────────

    def count_parameters(self) -> Dict[str, int]:
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        log.info("Parameters — total: %d  trainable: %d  frozen: %d",
                 total, trainable, total - trainable)
        return {"total": total, "trainable": trainable,
                "frozen": total - trainable}

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def save_checkpoint(
        self,
        path: str,
        epoch: int,
        metrics: Optional[Dict] = None,
        optimizer_state: Optional[Dict] = None,
    ):
        """Save model weights + training metadata."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        ckpt = {
            "epoch":            epoch,
            "model_state_dict": self.state_dict(),
            "metrics":          metrics or {},
            "model_config": {
                "num_classes":     self.num_classes,
                "feature_dim":     self.feature_dim,
            },
        }
        if optimizer_state:
            ckpt["optimizer_state_dict"] = optimizer_state
        torch.save(ckpt, path)
        log.info("Checkpoint saved → %s  (epoch %d)", path, epoch)

    @classmethod
    def load_checkpoint(cls, path: str, device: Optional[str] = None) -> Tuple["VisualForensicModel", Dict]:
        """Load model from checkpoint."""
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=device)
        cfg  = ckpt.get("model_config", {})
        model = cls(
            num_classes=cfg.get("num_classes", 2),
            pretrained=False,     # weights are in the checkpoint
            freeze_backbone=False,
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        log.info("Checkpoint loaded ← %s  (epoch %d)",
                 path, ckpt.get("epoch", -1))
        return model, ckpt


# ─── Standalone test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Running on: %s", device)

    model = VisualForensicModel(pretrained=True, freeze_backbone=True).to(device)
    model.count_parameters()

    # Fake batch
    x = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        out = model(x)

    log.info("logits shape   : %s", out["logits"].shape)
    log.info("features shape : %s", out["features"].shape)
    log.info("proba shape    : %s", out["proba"].shape)

    # Test feature extraction interface
    feats = model.extract_visual_features(x)
    log.info("extract_visual_features → %s  norm~%.4f",
             feats.shape, feats.norm(dim=-1).mean().item())

    # Test checkpoint save/load
    model.save_checkpoint(
        "outputs/checkpoints/test_ckpt.pth",
        epoch=0,
        metrics={"val_f1": 0.0},
    )
    model2, meta = VisualForensicModel.load_checkpoint(
        "outputs/checkpoints/test_ckpt.pth", device=device
    )
    log.info("Re-loaded model class: %s", type(model2).__name__)
    log.info("visual_model.py self-test PASSED.")
