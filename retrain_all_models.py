"""
=============================================================================
  COMPLETE RETRAINING PIPELINE — INDIAN MULTILINGUAL ID DATASET
=============================================================================
  Scans the Indian Multilingual ID Dataset (2040 images), extracts:
    • 11-D OCR features  (ocr_feature_extractor)
    • 16-D Visual features (visual_feature_extractor)
  Saves 3 CSV datasets, then trains 3 MLP classifiers:
    1. OCR-Only MLP         → models/ocr_only_mlp_model.joblib
    2. Visual-Only MLP      → models/visual_only_mlp_model.joblib
    3. Multimodal Fusion MLP → models/multimodal_fusion_mlp_model.joblib

  CRITICAL: Stratified train/test split — test set NEVER overlaps with train.
=============================================================================
"""

import os
import sys
import time
import glob
import warnings
import traceback
import numpy as np
import pandas as pd
import joblib

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Suppress verbose logging
os.environ["GLOG_minloglevel"] = "2"
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# Project imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ocr_feature_extractor import extract_ocr_features, OCR_FEATURE_NAMES
from visual_feature_extractor import VisualStreamExtractor, VISUAL_FEATURE_NAMES

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "indian_multilingual_id_dataset_ALL",
    "indian_multilingual_id_dataset"
)
OUTPUT_MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
RANDOM_STATE = 42
TEST_SIZE = 0.30  # 70% train, 30% test

OCR_CSV = "ocr_features_dataset.csv"
VISUAL_CSV = "visual_features_dataset.csv"
FUSED_CSV = "multimodal_fused_dataset.csv"

FUSED_COLS = VISUAL_FEATURE_NAMES + OCR_FEATURE_NAMES  # 16 + 11 = 27


def discover_dataset(root_dir):
    """
    Walks the Indian Multilingual ID Dataset and returns a list of dicts:
      { filepath, language, doc_type, label (0=genuine, 1=forged), label_name }
    """
    records = []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Dataset root not found: {root_dir}")

    for lang_name in sorted(os.listdir(root_dir)):
        lang_path = os.path.join(root_dir, lang_name)
        if not os.path.isdir(lang_path):
            continue

        for doc_type in sorted(os.listdir(lang_path)):
            doc_path = os.path.join(lang_path, doc_type)
            if not os.path.isdir(doc_path):
                continue

            for cls_name in sorted(os.listdir(doc_path)):
                cls_path = os.path.join(doc_path, cls_name)
                if not os.path.isdir(cls_path):
                    continue

                label = 0 if cls_name.lower() == "genuine" else 1
                label_name = cls_name.lower()

                for img_file in sorted(os.listdir(cls_path)):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        records.append({
                            "filepath": os.path.join(cls_path, img_file),
                            "filename": img_file,
                            "language": lang_name,
                            "doc_type": doc_type,
                            "label": label,
                            "label_name": label_name,
                        })

    return records


def extract_all_features(records):
    """
    Extracts OCR (11-D) and Visual (16-D) features for every image.
    Returns a DataFrame with all 27 features + metadata.
    """
    print(f"\n[STEP 1] Extracting features for {len(records)} images...")
    print(f"         OCR Features: {len(OCR_FEATURE_NAMES)}-D")
    print(f"         Visual Features: {len(VISUAL_FEATURE_NAMES)}-D")
    print(f"         Combined: {len(FUSED_COLS)}-D\n")

    vis_extractor = VisualStreamExtractor()
    all_rows = []
    errors = []
    t0 = time.time()

    for i, rec in enumerate(records):
        filepath = rec["filepath"]
        try:
            # OCR features (11-D)
            ocr_vec, _ = extract_ocr_features(filepath)

            # Visual features (16-D)
            vis_vec = vis_extractor.extract_features(filepath)

            row = {
                "filepath": filepath,
                "filename": rec["filename"],
                "language": rec["language"],
                "doc_type": rec["doc_type"],
                "label": rec["label"],
                "label_name": rec["label_name"],
            }
            for name, val in zip(VISUAL_FEATURE_NAMES, vis_vec):
                row[name] = round(float(val), 6)
            for name, val in zip(OCR_FEATURE_NAMES, ocr_vec):
                row[name] = round(float(val), 6)

            all_rows.append(row)

        except Exception as e:
            errors.append((filepath, str(e)))

        # Progress reporting
        if (i + 1) % 100 == 0 or (i + 1) == len(records):
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta = (len(records) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1:>4}/{len(records)}] "
                  f"{elapsed:.0f}s elapsed | {rate:.1f} img/s | ETA: {eta:.0f}s | "
                  f"errors: {len(errors)}")

    if errors:
        print(f"\n[WARNING] {len(errors)} images failed feature extraction:")
        for fp, err in errors[:10]:
            print(f"  • {os.path.basename(fp)}: {err}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")

    df = pd.DataFrame(all_rows)
    print(f"\n[DONE] Successfully extracted features for {len(df)} / {len(records)} images "
          f"({time.time() - t0:.1f}s total)")
    return df


def save_csv_datasets(df):
    """Saves the 3 separate CSV files for OCR, Visual, and Fused features."""
    meta_cols = ["filepath", "filename", "language", "doc_type", "label", "label_name"]

    # OCR features CSV
    ocr_cols = meta_cols + list(OCR_FEATURE_NAMES)
    df[ocr_cols].to_csv(OCR_CSV, index=False)
    print(f"  ✓ OCR features   → {OCR_CSV}  ({len(df)} rows × {len(OCR_FEATURE_NAMES)} features)")

    # Visual features CSV
    vis_cols = meta_cols + list(VISUAL_FEATURE_NAMES)
    df[vis_cols].to_csv(VISUAL_CSV, index=False)
    print(f"  ✓ Visual features → {VISUAL_CSV}  ({len(df)} rows × {len(VISUAL_FEATURE_NAMES)} features)")

    # Fused (multimodal) CSV
    fused_cols = meta_cols + list(FUSED_COLS)
    df[fused_cols].to_csv(FUSED_CSV, index=False)
    print(f"  ✓ Fused features  → {FUSED_CSV}  ({len(df)} rows × {len(FUSED_COLS)} features)")


def train_model(X_train, X_test, y_train, y_test, model_name, model_path, scaler_path):
    """
    Trains an MLP classifier, evaluates on the held-out test set,
    and saves the model + scaler.
    """
    # Standardize
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # Train MLP
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        max_iter=1000,
        random_state=RANDOM_STATE,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )
    mlp.fit(X_train_sc, y_train)

    # Evaluate on test set
    y_pred = mlp.predict(X_test_sc)
    y_proba = mlp.predict_proba(X_test_sc)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = 0.50

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n  ┌─── {model_name} ───")
    print(f"  │ Test Accuracy  : {acc*100:.2f}%")
    print(f"  │ Precision      : {prec*100:.2f}%")
    print(f"  │ Recall         : {rec*100:.2f}%")
    print(f"  │ F1-Score       : {f1*100:.2f}%")
    print(f"  │ AUROC          : {auc:.4f}")
    print(f"  │ Confusion Matrix:")
    print(f"  │   Pred Genuine | Pred Forged")
    print(f"  │   True Gen: [{cm[0][0]:>3}]      | [{cm[0][1]:>3}]")
    print(f"  │   True For: [{cm[1][0]:>3}]      | [{cm[1][1]:>3}]")

    # Classification report
    print(f"  │")
    report = classification_report(y_test, y_pred, target_names=["Genuine", "Forged"])
    for line in report.strip().split("\n"):
        print(f"  │ {line}")

    # Save model and scaler
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(mlp, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"  │")
    print(f"  │ ✓ Model  → {model_path}")
    print(f"  │ ✓ Scaler → {scaler_path}")
    print(f"  └{'─' * 50}")

    return {
        "model_name": model_name,
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1": f1, "auroc": auc,
        "mlp": mlp, "scaler": scaler
    }


def run_cross_validation(X, y, model_name, n_splits=5):
    """Runs stratified k-fold cross-validation for robust reporting."""
    print(f"\n  [CV] {model_name} — {n_splits}-Fold Stratified Cross-Validation:")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=1000,
        random_state=RANDOM_STATE, early_stopping=True,
        validation_fraction=0.15, n_iter_no_change=20
    )
    scores = cross_validate(
        mlp, X_sc, y, cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        key = f'test_{metric}'
        mean_val = np.mean(scores[key]) * 100
        std_val = np.std(scores[key]) * 100
        print(f"       {metric:<10}: {mean_val:.2f}% (±{std_val:.2f}%)")


def main():
    print("=" * 75)
    print("  COMPLETE RETRAINING PIPELINE — INDIAN MULTILINGUAL ID DATASET")
    print("=" * 75)
    print(f"  Dataset root : {DATASET_ROOT}")
    print(f"  Test size    : {TEST_SIZE*100:.0f}%")
    print(f"  Random state : {RANDOM_STATE}")
    print(f"  Output dir   : {OUTPUT_MODEL_DIR}")
    print("=" * 75)

    # -------------------------------------------------------------------------
    # PHASE 1: Discover dataset
    # -------------------------------------------------------------------------
    print("\n[PHASE 1] Discovering dataset...")
    records = discover_dataset(DATASET_ROOT)
    print(f"  Found {len(records)} images")

    # Print breakdown
    df_meta = pd.DataFrame(records)
    print(f"\n  By doc_type:")
    for dt, count in df_meta.groupby("doc_type").size().items():
        genuine = len(df_meta[(df_meta["doc_type"] == dt) & (df_meta["label"] == 0)])
        forged = len(df_meta[(df_meta["doc_type"] == dt) & (df_meta["label"] == 1)])
        print(f"    {dt:<20}: {count:>4} images ({genuine} genuine, {forged} forged)")

    print(f"\n  By language:")
    for lang, count in df_meta.groupby("language").size().items():
        print(f"    {lang:<20}: {count:>4} images")

    # -------------------------------------------------------------------------
    # PHASE 2: Extract features
    # -------------------------------------------------------------------------
    df = extract_all_features(records)

    if len(df) < 100:
        print(f"\n[ERROR] Only {len(df)} images extracted successfully. Aborting.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # PHASE 3: Save CSV datasets
    # -------------------------------------------------------------------------
    print("\n[PHASE 3] Saving CSV datasets...")
    save_csv_datasets(df)

    # -------------------------------------------------------------------------
    # PHASE 4: Stratified train/test split
    # -------------------------------------------------------------------------
    print("\n[PHASE 4] Creating stratified train/test split...")

    y = df["label"].values
    doc_types = df["doc_type"].values

    # Stratify by both label AND doc_type to ensure balanced representation
    stratify_key = df["label"].astype(str) + "_" + df["doc_type"]

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_key
    )

    y_train, y_test = y[train_idx], y[test_idx]
    print(f"  Train set: {len(train_idx)} samples "
          f"({np.sum(y_train == 0)} genuine, {np.sum(y_train == 1)} forged)")
    print(f"  Test set : {len(test_idx)} samples "
          f"({np.sum(y_test == 0)} genuine, {np.sum(y_test == 1)} forged)")
    print(f"  Overlap check: {len(set(train_idx) & set(test_idx))} shared indices (MUST be 0)")

    # -------------------------------------------------------------------------
    # PHASE 5: Train all 3 models
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  PHASE 5: TRAINING ALL 3 STREAM CLASSIFIERS")
    print("=" * 75)

    # Feature matrices
    X_ocr = df[list(OCR_FEATURE_NAMES)].values
    X_vis = df[list(VISUAL_FEATURE_NAMES)].values
    X_fused = df[list(FUSED_COLS)].values

    # 5a. OCR-Only MLP
    print("\n── [MODEL 1/3] OCR-Only Stream (11-D) ──")
    ocr_result = train_model(
        X_ocr[train_idx], X_ocr[test_idx],
        y_train, y_test,
        "OCR-Only MLP (11-D)",
        os.path.join(OUTPUT_MODEL_DIR, "ocr_only_mlp_model.joblib"),
        os.path.join(OUTPUT_MODEL_DIR, "ocr_scaler.joblib"),
    )

    # 5b. Visual-Only MLP
    print("\n── [MODEL 2/3] Visual-Only Stream (16-D) ──")
    vis_result = train_model(
        X_vis[train_idx], X_vis[test_idx],
        y_train, y_test,
        "Visual-Only MLP (16-D)",
        os.path.join(OUTPUT_MODEL_DIR, "visual_only_mlp_model.joblib"),
        os.path.join(OUTPUT_MODEL_DIR, "visual_scaler.joblib"),
    )

    # 5c. Multimodal Fusion MLP
    print("\n── [MODEL 3/3] Multimodal Fusion Stream (27-D) ──")
    fused_result = train_model(
        X_fused[train_idx], X_fused[test_idx],
        y_train, y_test,
        "Multimodal Fusion MLP (27-D)",
        os.path.join(OUTPUT_MODEL_DIR, "multimodal_fusion_mlp_model.joblib"),
        os.path.join(OUTPUT_MODEL_DIR, "multimodal_scaler.joblib"),
    )

    # -------------------------------------------------------------------------
    # PHASE 6: Cross-validation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  PHASE 6: 5-FOLD CROSS-VALIDATION (PUBLICATION-GRADE)")
    print("=" * 75)
    run_cross_validation(X_ocr, y, "OCR-Only (11-D)")
    run_cross_validation(X_vis, y, "Visual-Only (16-D)")
    run_cross_validation(X_fused, y, "Multimodal Fusion (27-D)")

    # -------------------------------------------------------------------------
    # PHASE 7: Final Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 75)
    print("  FINAL 3-WAY COMPARATIVE BENCHMARK (TEST SET)")
    print("=" * 75)
    print(f"{'Stream':<28} | {'Acc':>7} | {'Prec':>7} | {'Recall':>7} | {'F1':>7} | {'AUROC':>7}")
    print("-" * 75)
    for r in [ocr_result, vis_result, fused_result]:
        print(f"{r['model_name']:<28} | {r['accuracy']*100:>6.2f}% | {r['precision']*100:>6.2f}% | "
              f"{r['recall']*100:>6.2f}% | {r['f1']*100:>6.2f}% | {r['auroc']:>7.4f}")
    print("=" * 75)

    # Verify all model files exist
    print("\n[VERIFICATION] Checking saved model files:")
    expected_files = [
        "models/ocr_only_mlp_model.joblib",
        "models/ocr_scaler.joblib",
        "models/visual_only_mlp_model.joblib",
        "models/visual_scaler.joblib",
        "models/multimodal_fusion_mlp_model.joblib",
        "models/multimodal_scaler.joblib",
    ]
    all_ok = True
    for f in expected_files:
        full = os.path.join(os.path.dirname(os.path.abspath(__file__)), f)
        exists = os.path.exists(full)
        size = os.path.getsize(full) if exists else 0
        status = f"✓ {size/1024:.1f} KB" if exists else "✗ MISSING"
        print(f"  {f:<50} {status}")
        if not exists:
            all_ok = False

    if all_ok:
        print("\n╔═══════════════════════════════════════════════════╗")
        print("║  ALL 6 MODEL FILES TRAINED & SAVED SUCCESSFULLY  ║")
        print("╚═══════════════════════════════════════════════════╝")
    else:
        print("\n[ERROR] Some model files are missing!")

    print(f"\nTotal pipeline time: {time.time() - float(os.environ.get('PIPELINE_START', time.time())):.1f}s")


if __name__ == "__main__":
    os.environ["PIPELINE_START"] = str(time.time())
    main()
