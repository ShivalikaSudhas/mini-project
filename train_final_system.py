"""
train_final_system.py
Complete training & evaluation pipeline for the new combined Aadhaar & ID dataset.
Trains Visual-Only, OCR-Only, and 27-D Multimodal Fusion MLP classifiers,
updates scalers, updates XAI baseline, and generates tested demo samples.
"""

import os
import sys
import json
import shutil
import random
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

# Project imports
from visual_feature_extractor import VisualStreamExtractor, VISUAL_FEATURE_NAMES
from ocr_feature_extractor import extract_ocr_features, OCR_FEATURE_NAMES

FEATURE_NAMES_27D = VISUAL_FEATURE_NAMES + OCR_FEATURE_NAMES

def main():
    print("=" * 70)
    print("🚀 STARTING FINAL SYSTEM TRAINING ON NEW AADHAAR & ID DATASET")
    print("=" * 70)

    # 1. Discover all images
    p1 = Path("final_processed_dataset")
    p2 = Path("dataset/dataset")

    genuine_paths = []
    forged_paths = []

    if p1.exists():
        genuine_paths.extend(list(p1.glob("genuine/*.*")))
        forged_paths.extend(list(p1.glob("forged/*.*")))

    if p2.exists():
        genuine_paths.extend(list(p2.glob("original/*.*")))
        forged_paths.extend(list(p2.glob("forged/*.*")))

    # Filter valid image extensions
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    genuine_paths = [p for p in genuine_paths if p.suffix.lower() in valid_exts]
    forged_paths = [p for p in forged_paths if p.suffix.lower() in valid_exts]

    print(f"📊 Dataset Summary:")
    print(f"   • Genuine/Original images : {len(genuine_paths)}")
    print(f"   • Forged/Tampered images  : {len(forged_paths)}")
    print(f"   • Total dataset size      : {len(genuine_paths) + len(forged_paths)}")

    # 2. Extract Features
    print("\n🔍 Initializing Feature Extractors...")
    vis_extractor = VisualStreamExtractor("models/visual/best_visual_model.pth")

    all_records = []
    
    print("⏳ Extracting 16-D Visual + 11-D OCR features for all images...")
    for label, paths, cname in [(0, genuine_paths, "genuine"), (1, forged_paths, "forged")]:
        for i, img_path in enumerate(paths):
            try:
                vis_feats = vis_extractor.extract_features(img_path)
                ocr_feats_vec, _ = extract_ocr_features(img_path)
                fused_feats = np.concatenate([vis_feats, ocr_feats_vec])

                record = {
                    "path": str(img_path),
                    "filename": img_path.name,
                    "label": label,
                    "class_name": cname,
                }
                for idx, fname in enumerate(VISUAL_FEATURE_NAMES):
                    record[fname] = float(vis_feats[idx])
                for idx, fname in enumerate(OCR_FEATURE_NAMES):
                    record[fname] = float(ocr_feats_vec[idx])

                all_records.append(record)
            except Exception as e:
                print(f"   ⚠️ Warning: failed to extract {img_path.name}: {e}")

            if (i + 1) % 50 == 0:
                print(f"   Processed {i + 1}/{len(paths)} {cname} cards...")

    df = pd.DataFrame(all_records)
    print(f"\n✅ Successfully extracted features for {len(df)} samples.")

    # 3. Stratified 80/20 Train-Test Split
    print("\n✂️ Creating Stratified 80/20 Train/Test Split...")
    train_df, test_df = train_test_split(
        df, test_size=0.20, stratify=df["label"], random_state=42
    )

    print(f"   • Training samples : {len(train_df)} ({sum(train_df['label']==0)} genuine, {sum(train_df['label']==1)} forged)")
    print(f"   • Testing samples  : {len(test_df)} ({sum(test_df['label']==0)} genuine, {sum(test_df['label']==1)} forged)")

    X_train_vis = train_df[VISUAL_FEATURE_NAMES].values
    X_test_vis  = test_df[VISUAL_FEATURE_NAMES].values

    X_train_ocr = train_df[OCR_FEATURE_NAMES].values
    X_test_ocr  = test_df[OCR_FEATURE_NAMES].values

    X_train_fused = train_df[FEATURE_NAMES_27D].values
    X_test_fused  = test_df[FEATURE_NAMES_27D].values

    y_train = train_df["label"].values
    y_test  = test_df["label"].values

    # 4. Standard Scalers
    print("\n⚖️ Fitting and saving Feature Scalers...")
    scaler_vis = StandardScaler().fit(X_train_vis)
    scaler_ocr = StandardScaler().fit(X_train_ocr)
    scaler_fused = StandardScaler().fit(X_train_fused)

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    joblib.dump(scaler_vis, models_dir / "visual_scaler.joblib")
    joblib.dump(scaler_ocr, models_dir / "ocr_scaler.joblib")
    joblib.dump(scaler_fused, models_dir / "multimodal_scaler.joblib")

    X_train_vis_scaled = scaler_vis.transform(X_train_vis)
    X_test_vis_scaled  = scaler_vis.transform(X_test_vis)

    X_train_ocr_scaled = scaler_ocr.transform(X_train_ocr)
    X_test_ocr_scaled  = scaler_ocr.transform(X_test_ocr)

    X_train_fused_scaled = scaler_fused.transform(X_train_fused)
    X_test_fused_scaled  = scaler_fused.transform(X_test_fused)

    # 5. Train Classifiers
    print("\n🧠 Training MLP Classifiers...")

    # A. Visual-Only MLP
    print("   Training Visual-Only MLP...")
    mlp_vis = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
    mlp_vis.fit(X_train_vis_scaled, y_train)
    joblib.dump(mlp_vis, models_dir / "visual_only_mlp_model.joblib")

    # B. OCR-Only MLP
    print("   Training OCR-Only MLP...")
    mlp_ocr = MLPClassifier(
        hidden_layer_sizes=(32, 16),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True
    )
    mlp_ocr.fit(X_train_ocr_scaled, y_train)
    joblib.dump(mlp_ocr, models_dir / "ocr_only_mlp_model.joblib")

    # C. Multimodal Fusion MLP (27-D)
    print("   Training 27-D Multimodal Fusion MLP...")
    mlp_fused = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=600,
        random_state=42,
        early_stopping=True
    )
    mlp_fused.fit(X_train_fused_scaled, y_train)
    joblib.dump(mlp_fused, models_dir / "multimodal_fusion_mlp_model.joblib")

    # 6. Evaluation on Unseen Test Set
    print("\n📈 EVALUATION ON UNSEEN TEST SET:")
    results = {}
    for name, model, X_test in [
        ("Visual-Only MLP", mlp_vis, X_test_vis_scaled),
        ("OCR-Only MLP", mlp_ocr, X_test_ocr_scaled),
        ("27-D Multimodal Fusion", mlp_fused, X_test_fused_scaled)
    ]:
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        
        acc = float(accuracy_score(y_test, preds))
        f1  = float(f1_score(y_test, preds, zero_division=0))
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        try:
            auroc = float(roc_auc_score(y_test, probs))
        except:
            auroc = 0.5
        cm = confusion_matrix(y_test, preds).tolist()

        results[name] = {
            "accuracy": acc, "f1": f1, "precision": prec,
            "recall": rec, "auroc": auroc, "confusion_matrix": cm
        }
        print(f"   • {name:24s} | Acc: {acc*100:5.2f}% | F1: {f1:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | AUROC: {auroc:.4f}")

    with open(models_dir / "retrained_evaluation_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # 7. Update Genuine XAI Baseline
    print("\n📐 Calibrating Genuine XAI Baseline from Authentic Cards...")
    genuine_df = df[df["label"] == 0]
    baseline_stats = {}
    for col in FEATURE_NAMES_27D:
        baseline_stats[col] = {
            "mean": float(genuine_df[col].mean()),
            "std": float(genuine_df[col].std()) if float(genuine_df[col].std()) > 1e-6 else 1.0,
            "min": float(genuine_df[col].min()),
            "max": float(genuine_df[col].max())
        }
    with open(models_dir / "genuine_feature_baseline.json", "w") as f:
        json.dump(baseline_stats, f, indent=2)
    print("   Baseline saved to models/genuine_feature_baseline.json")

    # 8. Create / Update Demo Samples from Test Set
    print("\n🎯 Generating Fresh Tested Demo Samples...")
    demo_dir = Path("demo_samples")
    demo_dir.mkdir(exist_ok=True)

    test_genuine = test_df[test_df["label"] == 0]
    test_forged = test_df[test_df["label"] == 1]

    # Aadhaar demo samples
    aadhaar_gen = test_genuine[test_genuine["filename"].str.contains("gen_")]
    if len(aadhaar_gen) == 0:
        aadhaar_gen = test_genuine
    chosen_gen_aadh = aadhaar_gen.iloc[0]
    shutil.copy(chosen_gen_aadh["path"], demo_dir / "aadhaar_genuine.jpg")
    print(f"   • Saved aadhaar_genuine.jpg from {chosen_gen_aadh['filename']}")

    aadhaar_forg = test_forged[test_forged["filename"].str.contains("forged_blur|forged_text|forged_phone")]
    if len(aadhaar_forg) == 0:
        aadhaar_forg = test_forged
    chosen_forg_aadh = aadhaar_forg.iloc[0]
    shutil.copy(chosen_forg_aadh["path"], demo_dir / "aadhaar_forged.jpg")
    print(f"   • Saved aadhaar_forged.jpg from {chosen_forg_aadh['filename']}")

    # PAN card demo samples
    pan_gen = test_genuine[test_genuine["filename"].str.lower().str.contains("pan")]
    if len(pan_gen) == 0:
        pan_gen = df[(df["label"] == 0) & (df["filename"].str.lower().str.contains("pan"))]
    if len(pan_gen) > 0:
        chosen_gen_pan = pan_gen.iloc[0]
        shutil.copy(chosen_gen_pan["path"], demo_dir / "pan_genuine.jpg")
        print(f"   • Saved pan_genuine.jpg from {chosen_gen_pan['filename']}")

    pan_forg = test_forged[test_forged["filename"].str.lower().str.contains("pan")]
    if len(pan_forg) == 0:
        pan_forg = df[(df["label"] == 1) & (df["filename"].str.lower().str.contains("pan"))]
    if len(pan_forg) > 0:
        chosen_forg_pan = pan_forg.iloc[0]
        shutil.copy(chosen_forg_pan["path"], demo_dir / "pan_forged.jpg")
        print(f"   • Saved pan_forged.jpg from {chosen_forg_pan['filename']}")

    # Voter ID demo samples
    voter_gen = test_genuine[test_genuine["filename"].str.lower().str.contains("voter|election")]
    if len(voter_gen) == 0:
        voter_gen = df[(df["label"] == 0) & (df["filename"].str.lower().str.contains("voter|election"))]
    if len(voter_gen) > 0:
        chosen_gen_voter = voter_gen.iloc[0]
        shutil.copy(chosen_gen_voter["path"], demo_dir / "voter_id_genuine.jpg")
        print(f"   • Saved voter_id_genuine.jpg from {chosen_gen_voter['filename']}")

    voter_forg = test_forged[test_forged["filename"].str.lower().str.contains("voter|election")]
    if len(voter_forg) == 0:
        voter_forg = df[(df["label"] == 1) & (df["filename"].str.lower().str.contains("voter|election"))]
    if len(voter_forg) > 0:
        chosen_forg_voter = voter_forg.iloc[0]
        shutil.copy(chosen_forg_voter["path"], demo_dir / "voter_id_forged.jpg")
        print(f"   • Saved voter_id_forged.jpg from {chosen_forg_voter['filename']}")

    print("\n🎉 ALL TRAINING, CALIBRATION & SAMPLE GENERATION COMPLETED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
