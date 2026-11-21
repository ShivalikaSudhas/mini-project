# src/run_fusion.py
import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported reliably
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score
from src.fusion import build_feature_table, train_fusion, evaluate_fusion

workspace = Path("D:/visual")
metadata_dir = workspace / "data" / "metadata"
if not metadata_dir.exists():
    print(f"[WARN] Metadata directory {metadata_dir} not found. Skipping LODO evaluation.")
    sys.exit(0)
ocr_dir = workspace / "ocr_pilot"
results_dir = workspace / "results" / "metrics"
results_dir.mkdir(parents=True, exist_ok=True)

# Build the feature cache once across all samples to avoid re-extracting 5 times
master_csv = metadata_dir / "dataset_master.csv"
print("Pre-computing full visual + OCR feature cache for 1500 samples...")
full_feature_df = build_feature_table(master_csv, ocr_dir)
print(f"Feature table ready: shape {full_feature_df.shape}")

summary_rows = []
for fold_file in sorted(metadata_dir.glob("lodo_fold_*.csv")):
    fold_id = fold_file.stem.replace("lodo_fold_", "fold_")
    split_meta = pd.read_csv(fold_file)

    train_ids = split_meta[split_meta["split"] == "train"]["sample_id"]
    val_ids   = split_meta[split_meta["split"] == "val"]["sample_id"]
    test_ids  = split_meta[split_meta["split"] == "test"]["sample_id"]

    train_feat = full_feature_df[full_feature_df["sample_id"].isin(train_ids)].copy()
    val_feat   = full_feature_df[full_feature_df["sample_id"].isin(val_ids)].copy()
    test_feat  = full_feature_df[full_feature_df["sample_id"].isin(test_ids)].copy()

    doc_type_held_out = split_meta[split_meta["split"] == "test"]["doc_type"].iloc[0]
    print(f"\n--- Running LODO Fold: {fold_id} (Held-out test: {doc_type_held_out}) ---")

    # 1. Train & evaluate Multimodal Fusion model
    model, val_metrics = train_fusion(train_feat, val_feat)
    test_metrics = evaluate_fusion(model, test_feat)

    # 2. Visual-only baseline
    X_train_vis = np.vstack(train_feat["visual_feat"].values)
    y_train = train_feat["label"].values
    vis_clf = LogisticRegression(max_iter=500, class_weight="balanced")
    vis_clf.fit(X_train_vis, y_train)

    X_test_vis = np.vstack(test_feat["visual_feat"].values)
    y_test = test_feat["label"].values
    vis_prob = vis_clf.predict_proba(X_test_vis)[:, 1]
    vis_metrics = {
        "auroc": float(roc_auc_score(y_test, vis_prob)),
        "f1":    float(f1_score(y_test, (vis_prob > 0.5).astype(int))),
    }

    # 3. OCR-only baseline
    X_train_ocr = np.vstack(train_feat["ocr_feat"].values)
    ocr_clf = LogisticRegression(max_iter=500, class_weight="balanced")
    ocr_clf.fit(X_train_ocr, y_train)

    X_test_ocr = np.vstack(test_feat["ocr_feat"].values)
    ocr_prob = ocr_clf.predict_proba(X_test_ocr)[:, 1]
    ocr_metrics = {
        "auroc": float(roc_auc_score(y_test, ocr_prob)),
        "f1":    float(f1_score(y_test, (ocr_prob > 0.5).astype(int))),
    }

    summary_rows.append({
        "fold_id":    fold_id,
        "doc_type":   doc_type_held_out,
        "visual_auc": f"{vis_metrics['auroc']:.4f}",
        "visual_f1":  f"{vis_metrics['f1']:.4f}",
        "ocr_auc":    f"{ocr_metrics['auroc']:.4f}",
        "ocr_f1":     f"{ocr_metrics['f1']:.4f}",
        "fusion_auc": f"{test_metrics['auroc']:.4f}",
        "fusion_f1":  f"{test_metrics['f1']:.4f}",
    })
    print(f"  Visual AUC: {vis_metrics['auroc']:.4f} | OCR AUC: {ocr_metrics['auroc']:.4f} | Fusion AUC: {test_metrics['auroc']:.4f}")

summary_df = pd.DataFrame(summary_rows)
summary_path = results_dir / "fusion_lodo_results.csv"
summary_df.to_csv(summary_path, index=False)
print(f"\nSummary written to {summary_path}")

md_path = results_dir / "fusion_summary_table.md"
with open(md_path, "w", encoding="utf-8") as f:
    f.write("| Fold | Doc Type | Visual AUC | Visual F1 | OCR AUC | OCR F1 | Fusion AUC | Fusion F1 |\n")
    f.write("|------|----------|------------|-----------|---------|--------|------------|-----------|\n")
    for _, r in summary_df.iterrows():
        f.write(f"| {r['fold_id']} | {r['doc_type']} | {r['visual_auc']} | {r['visual_f1']} | "
                f"{r['ocr_auc']} | {r['ocr_f1']} | {r['fusion_auc']} | {r['fusion_f1']} |\n")
print(f"Markdown table written to {md_path}")
