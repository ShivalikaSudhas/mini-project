"""
train_visual.py — EfficientNet-B0 Visual Model Training Pipeline
Phase 3: Visual Model Development
Two-phase training:
  Phase 1: Classifier head training with frozen EfficientNet-B0 backbone
  Phase 2: Fine-tuning top MBConv blocks with differential learning rate
Evaluation & Outputs:
  - Checkpoints: models/visual/best_visual_model.pth
  - Metrics: results/metrics/visual_test_metrics.json
  - Graphs: results/graphs/visual_training_curves.png, results/graphs/visual_confusion_matrix.png
  - Error Analysis: results/reports/visual_error_analysis.csv
"""

import os
import sys
import json
import random
import tempfile
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# Set writable matplotlib config dir
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from preprocessing import load_split_dataloaders
from visual_model import VisualForensicModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def compute_metrics(y_true: List[int], y_pred: List[int], y_prob: List[float]) -> Dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = 0.5
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auroc": float(auroc),
        "confusion_matrix": cm,
    }


def evaluate(model: nn.Module, loader, criterion: nn.Module, device: str) -> Tuple[float, Dict, List[Dict]]:
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []
    all_prob = []
    sample_records = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            out = model(images)
            loss = criterion(out["logits"], labels)
            total_loss += loss.item() * len(labels)

            probs = out["proba"][:, 1].cpu().numpy()
            preds = torch.argmax(out["logits"], dim=1).cpu().numpy()
            targets = labels.cpu().numpy()

            all_true.extend(targets.tolist())
            all_pred.extend(preds.tolist())
            all_prob.extend(probs.tolist())

            for i in range(len(targets)):
                sample_records.append({
                    "sample_id": batch["sample_id"][i],
                    "doc_type": batch["doc_type"][i],
                    "source_id": batch["source_id"][i],
                    "true_label": int(targets[i]),
                    "pred_label": int(preds[i]),
                    "prob_forged": float(probs[i]),
                    "is_correct": bool(targets[i] == preds[i]),
                    "image_path": batch["image_path"][i]
                })

    avg_loss = total_loss / max(1, len(all_true))
    metrics = compute_metrics(all_true, all_pred, all_prob)
    return avg_loss, metrics, sample_records


def train_visual_model(
    split_csv_path: str = "D:/visual/data/metadata/split_in_domain.csv",
    output_model_dir: str = "D:/visual/models/visual",
    results_dir: str = "D:/visual/results",
    batch_size: int = 16,
    phase1_epochs: int = 5,
    phase2_epochs: int = 8,
    seed: int = 42,
):
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Starting Visual Model training on device: %s (Seed=%d)", device, seed)

    out_model_path = Path(output_model_dir)
    out_model_path.mkdir(parents=True, exist_ok=True)
    res_path = Path(results_dir)
    (res_path / "graphs").mkdir(parents=True, exist_ok=True)
    (res_path / "metrics").mkdir(parents=True, exist_ok=True)
    (res_path / "reports").mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    loaders = load_split_dataloaders(split_csv_path, batch_size=batch_size, num_workers=0)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # 2. Initialize Model
    model = VisualForensicModel(num_classes=2, pretrained=True, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [], "val_loss": [],
        "val_acc": [], "val_f1": [], "val_auroc": []
    }

    best_val_f1 = -1.0
    best_ckpt_path = out_model_path / "best_visual_model.pth"

    # ── Phase 1: Train classification head (Frozen backbone) ───────────────────
    log.info("=== Phase 1: Classification Head Warmup (Backbone Frozen, %d Epochs) ===", phase1_epochs)
    optimizer_head = AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-2)

    for epoch in range(1, phase1_epochs + 1):
        model.train()
        running_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer_head.zero_grad()
            out = model(images)
            loss = criterion(out["logits"], labels)
            loss.backward()
            optimizer_head.step()

            running_loss += loss.item() * len(labels)
            n_samples += len(labels)

        train_loss = running_loss / max(1, n_samples)
        val_loss, val_metrics, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auroc"].append(val_metrics["auroc"])

        log.info(
            "Phase 1 Epoch %d/%d — Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f | Val F1: %.4f | Val AUROC: %.4f",
            epoch, phase1_epochs, train_loss, val_loss, val_metrics["accuracy"], val_metrics["f1"], val_metrics["auroc"]
        )

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            model.save_checkpoint(str(best_ckpt_path), epoch=epoch, metrics=val_metrics)

    # ── Phase 2: Fine-Tuning Top MBConv Blocks ─────────────────────────────────
    log.info("=== Phase 2: Fine-Tuning Top 3 MBConv Blocks (%d Epochs) ===", phase2_epochs)
    model.unfreeze_top_n_blocks(n=3)

    # Differential learning rate: lower for backbone, higher for head
    backbone_params = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters())
    optimizer_ft = AdamW([
        {"params": backbone_params, "lr": 1e-4, "weight_decay": 1e-3},
        {"params": head_params, "lr": 5e-4, "weight_decay": 1e-2}
    ])
    scheduler = CosineAnnealingLR(optimizer_ft, T_max=phase2_epochs, eta_min=1e-6)

    for epoch in range(1, phase2_epochs + 1):
        curr_epoch = phase1_epochs + epoch
        model.train()
        running_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer_ft.zero_grad()
            out = model(images)
            loss = criterion(out["logits"], labels)
            loss.backward()
            optimizer_ft.step()

            running_loss += loss.item() * len(labels)
            n_samples += len(labels)

        scheduler.step()
        train_loss = running_loss / max(1, n_samples)
        val_loss, val_metrics, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_f1"].append(val_metrics["f1"])
        history["val_auroc"].append(val_metrics["auroc"])

        log.info(
            "Phase 2 Epoch %d/%d (Total %d) — Train Loss: %.4f | Val Loss: %.4f | Val Acc: %.4f | Val F1: %.4f | Val AUROC: %.4f",
            epoch, phase2_epochs, curr_epoch, train_loss, val_loss, val_metrics["accuracy"], val_metrics["f1"], val_metrics["auroc"]
        )

        if val_metrics["f1"] >= best_val_f1:
            best_val_f1 = val_metrics["f1"]
            model.save_checkpoint(str(best_ckpt_path), epoch=curr_epoch, metrics=val_metrics)

    log.info("Visual model training completed. Best Val F1: %.4f", best_val_f1)

    # ── Final Test Evaluation ──────────────────────────────────────────────────
    log.info("Evaluating Best Checkpoint on Test Set (150 in-domain samples)...")
    best_model, meta = VisualForensicModel.load_checkpoint(str(best_ckpt_path), device=device)
    test_loss, test_metrics, test_sample_records = evaluate(best_model, test_loader, criterion, device)

    log.info("=== TEST SET RESULTS ===")
    log.info("Accuracy  : %.4f", test_metrics["accuracy"])
    log.info("Precision : %.4f", test_metrics["precision"])
    log.info("Recall    : %.4f", test_metrics["recall"])
    log.info("F1-Score  : %.4f", test_metrics["f1"])
    log.info("AUROC     : %.4f", test_metrics["auroc"])
    log.info("Confusion Matrix:\n%s", np.array(test_metrics["confusion_matrix"]))

    # Save test metrics
    metrics_path = res_path / "metrics" / "visual_test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "test_loss": float(test_loss),
            "test_metrics": test_metrics,
            "best_checkpoint_epoch": meta.get("epoch", -1),
            "training_history": history,
            "config": {
                "architecture": "EfficientNet-B0",
                "phase1_epochs": phase1_epochs,
                "phase2_epochs": phase2_epochs,
                "batch_size": batch_size,
                "seed": seed,
            }
        }, f, indent=2)
    log.info("Saved test metrics to: %s", metrics_path)

    # Save Error Analysis CSV
    df_samples = pd.DataFrame(test_sample_records)
    error_csv_path = res_path / "reports" / "visual_error_analysis.csv"
    df_samples.to_csv(error_csv_path, index=False)
    log.info("Saved test error analysis report to: %s", error_csv_path)

    # ── Plot Training Curves ───────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    # Loss curve
    ax1.plot(epochs_range, history["train_loss"], label="Train Loss", color="royalblue", lw=2)
    ax1.plot(epochs_range, history["val_loss"], label="Val Loss", color="crimson", lw=2)
    ax1.axvline(x=phase1_epochs, color="gray", linestyle="--", label="Fine-Tuning Start")
    ax1.set_title("EfficientNet-B0 Visual Branch — Training Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Metric curve
    ax2.plot(epochs_range, history["val_acc"], label="Val Accuracy", color="forestgreen", lw=2)
    ax2.plot(epochs_range, history["val_f1"], label="Val F1-Score", color="darkorange", lw=2)
    ax2.plot(epochs_range, history["val_auroc"], label="Val AUROC", color="purple", lw=2)
    ax2.axvline(x=phase1_epochs, color="gray", linestyle="--", label="Fine-Tuning Start")
    ax2.set_title("Validation Performance Metrics", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Score [0 - 1]")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = res_path / "graphs" / "visual_training_curves.png"
    plt.savefig(curves_path, dpi=200)
    plt.close()
    log.info("Saved training curves plot to: %s", curves_path)

    # ── Plot Confusion Matrix ──────────────────────────────────────────────────
    cm = np.array(test_metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    classes = ["Genuine (0)", "Forged (1)"]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=f"Visual Model Test Confusion Matrix\n(F1={test_metrics['f1']:.3f}, Acc={test_metrics['accuracy']:.3f})",
        ylabel="True Label",
        xlabel="Predicted Label"
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")

    plt.tight_layout()
    cm_path = res_path / "graphs" / "visual_confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    log.info("Saved confusion matrix plot to: %s", cm_path)

    return test_metrics


if __name__ == "__main__":
    train_visual_model(
        batch_size=16,
        phase1_epochs=4,
        phase2_epochs=6,
        seed=42
    )
