# src/fusion.py
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------------
# 1. OCR feature extraction (scalar features)
# ------------------------------------------------------------------
def ocr_scalar_features(ocr_json_path: Path):
    """
    Returns a 3‑element vector:
        [mean_confidence, total_text_length, num_boxes]
    """
    data = json.loads(ocr_json_path.read_text(encoding="utf-8"))
    if "error" in data:
        # fallback to zeros if OCR failed
        return np.array([0.0, 0.0, 0.0])
    confidences = np.array(data.get("confidences", []), dtype=float)
    texts = data.get("texts", [])
    mean_conf = confidences.mean() if confidences.size else 0.0
    total_len = sum(len(t) for t in texts)
    num_boxes = len(texts)
    return np.array([mean_conf, total_len, num_boxes], dtype=float)

# ------------------------------------------------------------------
# 2. Visual feature loading (uses the visual model wrapper)
# ------------------------------------------------------------------
def visual_features(image_path: Path):
    """
    Loads the 1280‑dim L2‑normalized visual embedding
    using the public interface defined in visual_model.py.
    """
    from src.visual_model import extract_visual_features
    return extract_visual_features(str(image_path))

# ------------------------------------------------------------------
# 3. Build a DataFrame with combined features for a given split
# ------------------------------------------------------------------
def build_feature_table(split_csv: Path, ocr_dir: Path):
    """
    Returns a DataFrame with columns:
    sample_id, label, visual_feat (list), ocr_feat (list)
    """
    rows = []
    df = pd.read_csv(split_csv)
    for _, row in df.iterrows():
        img_path = Path(row["absolute_path"])
        # 1) visual embedding
        vis = visual_features(img_path)
        # 2) OCR scalars (pre‑computed JSON files from Phase 4)
        ocr_json = ocr_dir / f"{img_path.stem}.json"
        ocr = ocr_scalar_features(ocr_json)
        rows.append({
            "sample_id": row["sample_id"],
            "label": int(row["label"]),
            "visual_feat": vis,
            "ocr_feat": ocr,
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------------
# 4. Fusion model (concatenate + LogisticRegression)
# ------------------------------------------------------------------
def train_fusion(train_df: pd.DataFrame, val_df: pd.DataFrame):
    """
    Returns a fitted LogisticRegression model and validation metrics.
    """
    X_train = np.concatenate([
        np.vstack(train_df["visual_feat"].values),
        np.vstack(train_df["ocr_feat"].values)
    ], axis=1)
    y_train = train_df["label"].values

    X_val = np.concatenate([
        np.vstack(val_df["visual_feat"].values),
        np.vstack(val_df["ocr_feat"].values)
    ], axis=1)
    y_val = val_df["label"].values

    model = LogisticRegression(max_iter=500, class_weight="balanced", n_jobs=1)
    model.fit(X_train, y_train)

    val_prob = model.predict_proba(X_val)[:, 1]
    val_pred = model.predict(X_val)
    metrics = {
        "auroc": float(roc_auc_score(y_val, val_prob)),
        "f1": float(f1_score(y_val, val_pred)),
        "precision": float(precision_score(y_val, val_pred, zero_division=0)),
        "recall": float(recall_score(y_val, val_pred, zero_division=0)),
    }
    return model, metrics

# ------------------------------------------------------------------
# 5. Test evaluation (returns same metric dict)
# ------------------------------------------------------------------
def evaluate_fusion(model, test_df: pd.DataFrame):
    X_test = np.concatenate([
        np.vstack(test_df["visual_feat"].values),
        np.vstack(test_df["ocr_feat"].values)
    ], axis=1)
    y_test = test_df["label"].values
    prob = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)
    return {
        "auroc": float(roc_auc_score(y_test, prob)),
        "f1": float(f1_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
    }