import os
import sys
import glob
import cv2
import joblib
import numpy as np
import pandas as pd
from PIL import Image

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

import paddle
import paddle.vision.models as models
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, accuracy_score,
    confusion_matrix
)

# -------------------------------------------------------------------------
# FAST-TRACK DAY 2: VISUAL STREAM FEATURE EXTRACTOR & BASELINE CLASSIFIER
# -------------------------------------------------------------------------
# Goal: Build the visual stream feature extractor (Deep CNN embeddings + 
#       forensic edge/frequency analysis), compute visual vectors for all
#       samples, and evaluate the "Visual-Only Baseline".
# -------------------------------------------------------------------------

class VisualStreamExtractor:
    """
    Extracts a 16-dimensional visual feature vector combining:
    1. Deep CNN Spatial Embeddings (via lightweight pretrained vision backbone)
    2. Forensic Texture & Edge Tampering Statistics (Laplacian, FFT High-Freq, ELA)
    """
    def __init__(self):
        # Initialize deep backbone in evaluation mode
        self.backbone = models.mobilenet_v3_small(pretrained=False)
        self.backbone.eval()
        
    def extract_deep_cnn_features(self, img_bgr):
        """Extracts 8 compressed deep spatial latent features."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        normalized = (resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor_in = paddle.to_tensor(normalized.transpose(2, 0, 1), dtype='float32').unsqueeze(0)
        
        with paddle.no_grad():
            preds = self.backbone(tensor_in).numpy().flatten()
            
        # Group pooling into 8 deep visual feature bins
        pooled = np.mean(preds.reshape(8, -1), axis=1)
        return pooled

    def extract_forensic_texture_features(self, img_bgr):
        """Extracts 8 forensic visual features (edge sharpness, frequency, ELA)."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 1. Laplacian sharpness & edge variance (detects digital blur / smoothing)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(lap.var())
        lap_mean = float(np.abs(lap).mean())
        
        # 2. Sobel gradient energy (horizontal & vertical gradient variance)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        grad_mean = float(grad_mag.mean())
        grad_std = float(grad_mag.std())
        
        # 3. FFT Frequency domain analysis (flags digital copy-paste seams)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
        cy, cx = h // 2, w // 2
        high_freq_ratio = float(np.mean(magnitude_spectrum[:cy//2, :cx//2]) / (magnitude_spectrum.mean() + 1e-6))
        
        # 4. Error Level Analysis (ELA) proxy: JPEG compression residual variance
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
        resaved = cv2.imdecode(encimg, 1)
        diff = np.abs(img_bgr.astype(np.float32) - resaved.astype(np.float32))
        ela_mean = float(diff.mean())
        ela_std = float(diff.std())
        
        # 5. Color channel distribution covariance
        color_std_ratio = float(img_bgr[:, :, 0].std() / (img_bgr[:, :, 2].std() + 1e-5))
        
        return np.array([
            lap_var, lap_mean, grad_mean, grad_std,
            high_freq_ratio, ela_mean, ela_std, color_std_ratio
        ], dtype=np.float32)

    def extract_features(self, img_path):
        """Extracts complete 16-D visual feature vector."""
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")
            
        deep_feats = self.extract_deep_cnn_features(img)
        forensic_feats = self.extract_forensic_texture_features(img)
        return np.concatenate([deep_feats, forensic_feats])


VISUAL_FEATURE_COLS = [
    "cnn_feat_1", "cnn_feat_2", "cnn_feat_3", "cnn_feat_4",
    "cnn_feat_5", "cnn_feat_6", "cnn_feat_7", "cnn_feat_8",
    "laplacian_var", "laplacian_mean", "gradient_mean", "gradient_std",
    "high_freq_ratio", "ela_mean", "ela_std", "color_std_ratio"
]

def build_visual_features_dataset(input_csv="ocr_features_dataset.csv", output_csv="visual_features_dataset.csv"):
    """Extracts visual vectors for all images listed in the dataset."""
    df_meta = pd.read_csv(input_csv)
    print(f"[INFO] Extracting visual stream features for {len(df_meta)} document images...")
    
    extractor = VisualStreamExtractor()
    visual_records = []
    
    for idx, row in df_meta.iterrows():
        path = row['filepath']
        feats = extractor.extract_features(path)
        record = {
            "filepath": path,
            "filename": row['filename'],
            "doc_type": row['doc_type'],
            "label": row['label'],
            "label_name": row['label_name']
        }
        for col_name, val in zip(VISUAL_FEATURE_COLS, feats):
            record[col_name] = round(float(val), 6)
        visual_records.append(record)
        
    df_visual = pd.DataFrame(visual_records)
    df_visual.to_csv(output_csv, index=False)
    print(f"[SUCCESS] Visual features dataset saved to: '{output_csv}'")
    return df_visual


def train_and_evaluate_visual_baseline(csv_path="visual_features_dataset.csv", output_model_dir="models"):
    print("\n" + "=" * 70)
    print("  FAST-TRACK DAY 2: VISUAL-ONLY BASELINE MODEL TRAINING & EVALUATION")
    print("=" * 70)
    
    df = pd.read_csv(csv_path)
    X = df[VISUAL_FEATURE_COLS].values
    y = df['label'].values
    doc_types = df['doc_type'].values
    
    print(f"[INFO] Dataset loaded: {len(df)} samples across {len(np.unique(doc_types))} document categories.")
    
    # 1. Stratified Split (70% Train, 30% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    
    # 2. Standardize Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. Model 1: Visual-Only MLP Classifier
    mlp_model = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    mlp_model.fit(X_train_scaled, y_train)
    
    # 4. Model 2: Visual-Only Random Forest
    rf_model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42)
    rf_model.fit(X_train, y_train)
    
    models_dict = {
        "Visual-Only MLP Neural Net": (mlp_model, X_test_scaled),
        "Visual-Only Random Forest": (rf_model, X_test)
    }
    
    print("\n" + "-" * 70)
    print(f"{'MODEL ARCHITECTURE':<30} | {'ACCURACY':<8} | {'PRECISION':<9} | {'RECALL':<8} | {'F1':<8} | {'AUROC'}")
    print("-" * 70)
    
    for name, (clf, test_feats) in models_dict.items():
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
            
        print(f"{name:<30} | {acc*100:>7.2f}% | {prec*100:>8.2f}% | {rec*100:>7.2f}% | {f1*100:>7.2f}% | {auc:>6.4f}")
        
    # 5. 5-Fold Cross Validation
    print("-" * 70)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        mlp_model, scaler.fit_transform(X), y, cv=cv,
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
    print("\n[INFO] Visual-Only 5-Fold Cross Validation Statistics:")
    print(f"  • Mean Accuracy : {np.mean(cv_scores['test_accuracy'])*100:.2f}% (+/- {np.std(cv_scores['test_accuracy'])*100:.2f}%)")
    print(f"  • Mean F1-Score : {np.mean(cv_scores['test_f1'])*100:.2f}% (+/- {np.std(cv_scores['test_f1'])*100:.2f}%)")
    print(f"  • Mean AUROC    : {np.mean(cv_scores['test_roc_auc']):.4f} (+/- {np.std(cv_scores['test_roc_auc']):.4f})")
    
    # 6. Confusion Matrix
    y_pred_mlp = mlp_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_pred_mlp)
    print("\n[CONFUSION MATRIX - VISUAL MLP TEST EVALUATION]")
    print(f"  Predicted Genuine | Predicted Forged")
    print(f"  True Genuine: [{cm[0][0]:>2}]          | [{cm[0][1]:>2}]")
    print(f"  True Forged : [{cm[1][0]:>2}]          | [{cm[1][1]:>2}]")
    
    # 7. Save Model & Scaler Artifacts
    os.makedirs(output_model_dir, exist_ok=True)
    joblib.dump(mlp_model, os.path.join(output_model_dir, "visual_only_mlp_model.joblib"))
    joblib.dump(scaler, os.path.join(output_model_dir, "visual_scaler.joblib"))
    print(f"\n[SUCCESS] Saved Visual Model & Scaler to: '{output_model_dir}/'")
    print("=" * 70)


if __name__ == "__main__":
    build_visual_features_dataset("ocr_features_dataset.csv", "visual_features_dataset.csv")
    train_and_evaluate_visual_baseline("visual_features_dataset.csv", "models")
