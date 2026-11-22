import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, accuracy_score,
    confusion_matrix, classification_report
)

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# -------------------------------------------------------------------------
# FAST-TRACK DAY 1: OCR-ONLY CLASSIFIER BASELINE
# -------------------------------------------------------------------------
# Goal: Train an MLP and Random Forest model strictly using the 11-D OCR
#       numerical features. Establish the "OCR-Only" baseline performance
#       for precision, recall, F1, and AUROC before multimodal fusion.
# -------------------------------------------------------------------------

FEATURE_COLS = [
    'mean_confidence',
    'min_confidence',
    'max_confidence',
    'std_confidence',
    'low_conf_count',
    'low_conf_ratio',
    'num_text_boxes',
    'text_area_ratio',
    'mean_aspect_ratio',
    'box_height_std',
    'mean_char_density'
]

def load_data(csv_path="ocr_features_dataset.csv"):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset '{csv_path}' not found! Run day6_forgery_generator.py first.")
        
    df = pd.read_csv(csv_path)
    X = df[FEATURE_COLS].values
    y = df['label'].values
    doc_types = df['doc_type'].values
    return df, X, y, doc_types

def train_and_evaluate_ocr_baseline(csv_path="ocr_features_dataset.csv", output_model_dir="models"):
    print("=" * 70)
    print("  FAST-TRACK DAY 1: OCR-ONLY BASELINE MODEL TRAINING & EVALUATION")
    print("=" * 70)
    
    df, X, y, doc_types = load_data(csv_path)
    print(f"[INFO] Loaded {len(df)} samples across {len(np.unique(doc_types))} document categories.")
    print(f"       Class distribution: {np.sum(y == 0)} Genuine (0) | {np.sum(y == 1)} Forged (1)")

    # 1. Stratified Train / Test Split (70% Train, 30% Test)
    X_train, X_test, y_train, y_test, docs_train, docs_test = train_test_split(
        X, y, doc_types, test_size=0.30, random_state=42, stratify=y
    )

    # 2. Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Model Architecture 1: Multi-Layer Perceptron (Neural Network Stream)
    mlp_model = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation='relu',
        max_iter=500,
        random_state=42,
        early_stopping=False
    )
    mlp_model.fit(X_train_scaled, y_train)

    # 4. Model Architecture 2: Random Forest (Tree Ensemble Reference)
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    rf_model.fit(X_train, y_train)

    # 5. Evaluate on Hold-out Test Set
    models = {
        "OCR-Only MLP Neural Network": (mlp_model, X_test_scaled),
        "OCR-Only Random Forest": (rf_model, X_test)
    }

    results = {}
    print("\n" + "-" * 70)
    print(f"{'MODEL ARCHITECTURE':<30} | {'ACCURACY':<8} | {'PRECISION':<9} | {'RECALL':<8} | {'F1':<8} | {'AUROC'}")
    print("-" * 70)

    for name, (clf, test_feats) in models.items():
        y_pred = clf.predict(test_feats)
        y_proba = clf.predict_proba(test_feats)[:, 1] if hasattr(clf, "predict_proba") else y_pred
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_proba)
        except Exception:
            auc = 0.50

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auroc": auc}
        print(f"{name:<30} | {acc*100:>7.2f}% | {prec*100:>8.2f}% | {rec*100:>7.2f}% | {f1*100:>7.2f}% | {auc:>6.4f}")

    # 6. Stratified 5-Fold Cross-Validation for Robust Academic Reporting
    print("-" * 70)
    print("\n[INFO] Running 5-Fold Cross Validation for Publication-Grade Statistics:")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        mlp_model, scaler.fit_transform(X), y, cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
    print(f"  • 5-Fold Mean Accuracy : {np.mean(cv_scores['test_accuracy'])*100:.2f}% (+/- {np.std(cv_scores['test_accuracy'])*100:.2f}%)")
    print(f"  • 5-Fold Mean F1-Score : {np.mean(cv_scores['test_f1'])*100:.2f}% (+/- {np.std(cv_scores['test_f1'])*100:.2f}%)")
    print(f"  • 5-Fold Mean AUROC    : {np.mean(cv_scores['test_roc_auc']):.4f} (+/- {np.std(cv_scores['test_roc_auc']):.4f})")

    # 7. Confusion Matrix
    y_pred_mlp = mlp_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_mlp)
    print("\n[CONFUSION MATRIX - MLP TEST EVALUATION]")
    print(f"  Predicted Genuine | Predicted Forged")
    print(f"  True Genuine: [{cm[0][0]:>2}]          | [{cm[0][1]:>2}]")
    print(f"  True Forged : [{cm[1][0]:>2}]          | [{cm[1][1]:>2}]")

    # 8. Feature Importances (from Random Forest)
    importances = rf_model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    print("\n[OCR FEATURE IMPORTANCE RANKING (Top 5)]")
    for i in range(min(5, len(sorted_idx))):
        idx = sorted_idx[i]
        print(f"  {i+1}. {FEATURE_COLS[idx]:<20} : {importances[idx]*100:.2f}% impact")

    # 9. Save Trained Model & Scaler for Multimodal Fusion (Day 3)
    os.makedirs(output_model_dir, exist_ok=True)
    model_save_path = os.path.join(output_model_dir, "ocr_only_mlp_model.joblib")
    scaler_save_path = os.path.join(output_model_dir, "ocr_scaler.joblib")
    joblib.dump(mlp_model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"\n[SUCCESS] Trained OCR Baseline Model saved to: '{model_save_path}'")
    print(f"[SUCCESS] Feature Scaler saved to: '{scaler_save_path}'")
    print("=" * 70)

    return results

if __name__ == "__main__":
    train_and_evaluate_ocr_baseline()
