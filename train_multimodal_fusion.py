import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, accuracy_score,
    confusion_matrix
)

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# -------------------------------------------------------------------------
# FAST-TRACK DAY 3: MULTIMODAL FEATURE FUSION ENGINE (VISUAL + OCR)
# -------------------------------------------------------------------------
# Goal: Fuse 16-D Visual features and 11-D OCR features into a unified 27-D
#       feature representation. Train the Multimodal Fusion MLP and run
#       the formal 3-way comparative benchmark:
#       1. OCR-Only
#       2. Visual-Only
#       3. Multimodal (Visual + OCR)
# -------------------------------------------------------------------------

OCR_COLS = [
    'mean_confidence', 'min_confidence', 'max_confidence', 'std_confidence',
    'low_conf_count', 'low_conf_ratio', 'num_text_boxes', 'text_area_ratio',
    'mean_aspect_ratio', 'box_height_std', 'mean_char_density'
]

VISUAL_COLS = [
    'cnn_feat_1', 'cnn_feat_2', 'cnn_feat_3', 'cnn_feat_4',
    'cnn_feat_5', 'cnn_feat_6', 'cnn_feat_7', 'cnn_feat_8',
    'laplacian_var', 'laplacian_mean', 'gradient_mean', 'gradient_std',
    'high_freq_ratio', 'ela_mean', 'ela_std', 'color_std_ratio'
]

FUSED_COLS = VISUAL_COLS + OCR_COLS

def merge_and_build_multimodal_dataset(
    ocr_csv="ocr_features_dataset.csv",
    visual_csv="visual_features_dataset.csv",
    output_fused_csv="multimodal_fused_dataset.csv"
):
    """Combines OCR and Visual features on matching filepaths."""
    df_ocr = pd.read_csv(ocr_csv)
    df_visual = pd.read_csv(visual_csv)
    
    # Merge datasets on metadata keys
    meta_keys = ['filepath', 'filename', 'doc_type', 'label', 'label_name']
    df_fused = pd.merge(df_visual[meta_keys + VISUAL_COLS], df_ocr[meta_keys + OCR_COLS], on=meta_keys)
    
    df_fused.to_csv(output_fused_csv, index=False)
    print(f"[INFO] Merged 16 Visual + 11 OCR features -> 27-D Multimodal Dataset: '{output_fused_csv}'")
    print(f"       Total records: {len(df_fused)}")
    return df_fused

def run_multimodal_comparative_experiment(csv_path="multimodal_fused_dataset.csv", output_model_dir="models"):
    print("\n" + "=" * 75)
    print("  FAST-TRACK DAY 3: MULTIMODAL FEATURE FUSION & 3-WAY COMPARATIVE BENCHMARK")
    print("=" * 75)
    
    df = pd.read_csv(csv_path)
    y = df['label'].values
    doc_types = df['doc_type'].values
    
    # Extract feature subsets
    X_ocr = df[OCR_COLS].values
    X_visual = df[VISUAL_COLS].values
    X_fused = df[FUSED_COLS].values
    
    # Identical stratified split for fair comparison (70% train, 30% test)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.30, random_state=42, stratify=y)
    
    streams = {
        "1. OCR-Only Stream (11-D)": (X_ocr, OCR_COLS),
        "2. Visual-Only Stream (16-D)": (X_visual, VISUAL_COLS),
        "3. Multimodal Fusion (27-D)": (X_fused, FUSED_COLS)
    }
    
    results = []
    trained_models = {}
    
    print("\n" + "-" * 75)
    print(f"{'PIPELINE CONFIGURATION':<30} | {'ACCURACY':<8} | {'PRECISION':<9} | {'RECALL':<8} | {'F1':<8} | {'AUROC'}")
    print("-" * 75)
    
    for name, (X_mat, cols) in streams.items():
        X_train, X_test = X_mat[train_idx], X_mat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        # Standard MLP classifier (matching project architecture: MLP -> Genuine/Forged)
        mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
        mlp.fit(X_train_sc, y_train)
        
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
            
        trained_models[name] = (mlp, scaler)
        results.append({
            "Pipeline": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "AUROC": auc
        })
        
        print(f"{name:<30} | {acc*100:>7.2f}% | {prec*100:>8.2f}% | {rec*100:>7.2f}% | {f1*100:>7.2f}% | {auc:>6.4f}")
        
    print("-" * 75)
    
    # Run 5-fold cross-validation on the multimodal fusion stream
    print("\n[INFO] Multimodal Fusion (27-D) 5-Fold Cross Validation:")
    scaler_full = StandardScaler()
    X_fused_sc = scaler_full.fit_transform(X_fused)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mlp_fused = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    cv_scores = cross_validate(
        mlp_fused, X_fused_sc, y, cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
    print(f"  • 5-Fold Mean Accuracy : {np.mean(cv_scores['test_accuracy'])*100:.2f}% (+/- {np.std(cv_scores['test_accuracy'])*100:.2f}%)")
    print(f"  • 5-Fold Mean F1-Score : {np.mean(cv_scores['test_f1'])*100:.2f}% (+/- {np.std(cv_scores['test_f1'])*100:.2f}%)")
    print(f"  • 5-Fold Mean AUROC    : {np.mean(cv_scores['test_roc_auc']):.4f} (+/- {np.std(cv_scores['test_roc_auc']):.4f})")

    # Save Multimodal Fusion Model Artifacts
    os.makedirs(output_model_dir, exist_ok=True)
    fused_mlp, fused_scaler = trained_models["3. Multimodal Fusion (27-D)"]
    joblib.dump(fused_mlp, os.path.join(output_model_dir, "multimodal_fusion_mlp_model.joblib"))
    joblib.dump(fused_scaler, os.path.join(output_model_dir, "multimodal_scaler.joblib"))
    print(f"\n[SUCCESS] Trained Multimodal Fusion MLP saved to: '{output_model_dir}/multimodal_fusion_mlp_model.joblib'")
    print(f"[SUCCESS] Multimodal Scaler saved to: '{output_model_dir}/multimodal_scaler.joblib'")
    print("=" * 75)
    
    return results

if __name__ == "__main__":
    merge_and_build_multimodal_dataset()
    run_multimodal_comparative_experiment()
