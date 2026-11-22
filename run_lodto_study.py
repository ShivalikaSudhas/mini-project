import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
)

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# -------------------------------------------------------------------------
# FAST-TRACK DAY 4: LEAVE-ONE-DOCUMENT-TYPE-OUT (LODTO) GENERALIZATION STUDY
# -------------------------------------------------------------------------
# Research Question:
# "Can a multimodal visual + OCR forgery detector trained on known document
#  types generalize to a completely unseen document type?"
#
# Protocol:
#   For each of the 5 document types:
#     - Train on 4 known document types (Seen)
#     - Test on 1 held-out document type (Unseen)
#   Compare:
#     1. OCR-Only
#     2. Visual-Only
#     3. Multimodal (Visual + OCR)
#   Metrics:
#     Precision, Recall, F1, AUROC, Seen vs Unseen Acc, Generalization Gap
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

def run_lodto_experiments(dataset_csv="multimodal_fused_dataset.csv", output_csv="lodto_experiment_results.csv"):
    print("=" * 80)
    print("  FAST-TRACK DAY 4: LEAVE-ONE-DOCUMENT-TYPE-OUT (LODTO) GENERALIZATION STUDY")
    print("=" * 80)
    
    if not os.path.exists(dataset_csv):
        raise FileNotFoundError(f"Dataset '{dataset_csv}' not found. Run Day 3 first.")
        
    df = pd.read_csv(dataset_csv)
    doc_types = sorted(df['doc_type'].unique())
    y = df['label'].values
    
    print(f"[INFO] Discovered {len(doc_types)} document types for LODTO rotation: {doc_types}")
    print(f"       Total benchmark samples: {len(df)} ({np.sum(y==0)} Genuine, {np.sum(y==1)} Forged)")
    
    streams = {
        "OCR-Only": OCR_COLS,
        "Visual-Only": VISUAL_COLS,
        "Multimodal (Visual+OCR)": FUSED_COLS
    }
    
    lodto_records = []
    
    for held_out_doc in doc_types:
        print(f"\n>>> ROTATION FOLD: Holding out Unseen Type: '{held_out_doc.upper()}'")
        
        # Split: Train on 4 doc types, Test strictly on the 1 held-out doc type
        train_mask = (df['doc_type'] != held_out_doc).values
        test_mask = (df['doc_type'] == held_out_doc).values
        
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        for stream_name, feature_cols in streams.items():
            X_raw = df[feature_cols].values
            X_train = X_raw[train_mask]
            X_test = X_raw[test_mask]
            
            # Standardize strictly using training distribution
            scaler = StandardScaler()
            X_train_sc = scaler.fit_transform(X_train)
            X_test_sc = scaler.transform(X_test)
            
            # Train MLP Classifier
            clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
            clf.fit(X_train_sc, y_train)
            
            # Evaluate on SEEN (Train) data
            y_train_pred = clf.predict(X_train_sc)
            seen_acc = accuracy_score(y_train, y_train_pred)
            
            # Evaluate on UNSEEN (Held-Out) data
            y_test_pred = clf.predict(X_test_sc)
            y_test_proba = clf.predict_proba(X_test_sc)[:, 1] if hasattr(clf, "predict_proba") else y_test_pred
            
            unseen_acc = accuracy_score(y_test, y_test_pred)
            unseen_prec = precision_score(y_test, y_test_pred, zero_division=0)
            unseen_rec = recall_score(y_test, y_test_pred, zero_division=0)
            unseen_f1 = f1_score(y_test, y_test_pred, zero_division=0)
            
            try:
                unseen_auc = roc_auc_score(y_test, y_test_proba)
            except Exception:
                unseen_auc = 0.50
                
            gen_gap = seen_acc - unseen_acc # Drop in performance when encountering unseen card
            
            lodto_records.append({
                "Held_Out_Doc": held_out_doc,
                "Stream": stream_name,
                "Seen_Acc": seen_acc,
                "Unseen_Acc": unseen_acc,
                "Gen_Gap_Acc": gen_gap,
                "Unseen_Precision": unseen_prec,
                "Unseen_Recall": unseen_rec,
                "Unseen_F1": unseen_f1,
                "Unseen_AUROC": unseen_auc
            })
            
            print(f"    [{stream_name:<23}] Seen: {seen_acc*100:>5.1f}% | Unseen: {unseen_acc*100:>5.1f}% | "
                  f"Gap: {gen_gap*100:>+5.1f}% | F1: {unseen_f1*100:>5.1f}% | AUROC: {unseen_auc:.4f}")

    df_results = pd.DataFrame(lodto_records)
    df_results.to_csv(output_csv, index=False)
    
    # Generate Overall LODTO Benchmark Summary Table
    print("\n" + "=" * 80)
    print("  FINAL CROSS-DOCUMENT GENERALIZATION (LODTO) SUMMARY TABLE")
    print("=" * 80)
    
    summary = df_results.groupby("Stream").agg({
        "Seen_Acc": lambda x: f"{np.mean(x)*100:.2f}%",
        "Unseen_Acc": lambda x: f"{np.mean(x)*100:.2f}%",
        "Gen_Gap_Acc": lambda x: f"{np.mean(x)*100:.2f}%",
        "Unseen_Precision": lambda x: f"{np.mean(x)*100:.2f}%",
        "Unseen_Recall": lambda x: f"{np.mean(x)*100:.2f}%",
        "Unseen_F1": lambda x: f"{np.mean(x)*100:.2f}%",
        "Unseen_AUROC": lambda x: f"{np.mean(x):.4f}"
    }).reset_index()
    
    print(f"{'STREAM':<24} | {'SEEN ACC':<9} | {'UNSEEN ACC':<10} | {'GEN GAP':<8} | {'PRECISION':<9} | {'RECALL':<8} | {'F1-SCORE':<8} | {'AUROC'}")
    print("-" * 105)
    for _, row in summary.iterrows():
        print(f"{row['Stream']:<24} | {row['Seen_Acc']:<9} | {row['Unseen_Acc']:<10} | {row['Gen_Gap_Acc']:<8} | "
              f"{row['Unseen_Precision']:<9} | {row['Unseen_Recall']:<8} | {row['Unseen_F1']:<8} | {row['Unseen_AUROC']}")
    print("=" * 105)
    
    print(f"\n[SUCCESS] Detailed fold-by-fold results exported to: '{output_csv}'")
    return df_results, summary

if __name__ == "__main__":
    run_lodto_experiments()
