"""
Reproducible classifier refresh using the repository's precomputed feature tables.
Run from the repository root: python retrain_from_feature_tables.py

This retrains the three MLP classifiers on a group-isolated 80/20 split.
It does NOT re-extract image features or retrain the EfficientNet backbone.
"""
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, roc_auc_score, confusion_matrix)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "multimodal_fused_dataset.csv"
OUT = ROOT / "results" / "retrained_current_dataset"
OCR = ["mean_confidence","min_confidence","max_confidence","std_confidence",
       "low_conf_count","low_conf_ratio","num_text_boxes","text_area_ratio",
       "mean_aspect_ratio","box_height_std","mean_char_density"]
VISUAL = ["cnn_feat_1","cnn_feat_2","cnn_feat_3","cnn_feat_4","cnn_feat_5",
          "cnn_feat_6","cnn_feat_7","cnn_feat_8","laplacian_var","laplacian_mean",
          "gradient_mean","gradient_std","high_freq_ratio","ela_mean","ela_std",
          "color_std_ratio"]
STREAMS = {"Visual-only": VISUAL, "OCR-only": OCR, "Multimodal fusion": VISUAL + OCR}

def main():
    df = pd.read_csv(DATA)
    required = set(VISUAL + OCR + ["label","doc_type","language","filename"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {DATA}: {sorted(missing)}")
    if df[VISUAL + OCR].isna().any().any():
        raise ValueError("Feature table contains missing values; audit before training.")
    groups = (df.doc_type.astype(str) + "_" + df.language.astype(str) + "_" +
              df.filename.astype(str))
    train_idx, test_idx = next(GroupShuffleSplit(
        n_splits=1, test_size=0.20, random_state=42).split(df, df.label, groups))
    assert not (set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))
    y_train, y_test = df.iloc[train_idx].label.to_numpy(), df.iloc[test_idx].label.to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "models").mkdir(exist_ok=True)
    results = {}
    for name, cols in STREAMS.items():
        scaler = StandardScaler()
        X_train = scaler.fit_transform(df.iloc[train_idx][cols].to_numpy())
        X_test = scaler.transform(df.iloc[test_idx][cols].to_numpy())
        model = MLPClassifier(hidden_layer_sizes=(32,16), max_iter=700,
            random_state=42, early_stopping=True, n_iter_no_change=25)
        model.fit(X_train, y_train)
        pred, prob = model.predict(X_test), model.predict_proba(X_test)[:,1]
        key = name.lower().replace("-", "_").replace(" ", "_")
        joblib.dump(model, OUT/"models"/f"{key}_mlp.joblib")
        joblib.dump(scaler, OUT/"models"/f"{key}_scaler.joblib")
        results[name] = {
            "accuracy": float(accuracy_score(y_test,pred)),
            "precision": float(precision_score(y_test,pred,zero_division=0)),
            "recall": float(recall_score(y_test,pred,zero_division=0)),
            "f1": float(f1_score(y_test,pred,zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test,pred)),
            "roc_auc": float(roc_auc_score(y_test,prob)),
            "confusion_matrix": confusion_matrix(y_test,pred).tolist(),
            "iterations": int(model.n_iter_)
        }
    meta = {
        "dataset":"multimodal_fused_dataset.csv",
        "total_samples":int(len(df)),
        "feature_dimensions":{"visual":16,"ocr":11,"fusion":27},
        "split":{"method":"GroupShuffleSplit","test_size":0.20,"random_state":42,
            "train_samples":int(len(train_idx)),"test_samples":int(len(test_idx)),
            "train_genuine":int((y_train==0).sum()),"train_forged":int((y_train==1).sum()),
            "test_genuine":int((y_test==0).sum()),"test_forged":int((y_test==1).sum()),
            "group_key":"doc_type + language + filename","group_overlap":0},
        "models":results,
        "caveat":"Classifier-only retraining from precomputed features. EfficientNet backbone and raw-image feature extraction were not retrained."
    }
    (OUT/"evaluation_metrics.json").write_text(json.dumps(meta,indent=2))
    pd.DataFrame([{"Model":k,**{m:v[m] for m in
        ["accuracy","precision","recall","f1","balanced_accuracy","roc_auc","iterations"]}}
        for k,v in results.items()]).to_csv(OUT/"model_comparison.csv",index=False)
    print(json.dumps(meta,indent=2))

if __name__ == "__main__":
    main()
