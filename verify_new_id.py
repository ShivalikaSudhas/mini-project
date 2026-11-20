import os
import sys
import glob
import argparse
import joblib
import cv2
import numpy as np
import pandas as pd

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Suppress verbose paddle logging
os.environ["GLOG_minloglevel"] = "2"
from paddleocr import PaddleOCR
from ocr_feature_extractor import extract_ocr_features
from visual_feature_extractor import VisualStreamExtractor

# -------------------------------------------------------------------------
# NEW DATASET & SINGLE IMAGE VERIFICATION TOOL
# -------------------------------------------------------------------------
# Takes ANY unseen ID card image or entire folder of images, extracts the
# 27-D multimodal feature vector, and outputs:
#   1. VERDICT: GENUINE vs. FORGED
#   2. Tampering Probability Score (0.0% to 100.0%)
#   3. Forensic Explanations (Why it was flagged)
# -------------------------------------------------------------------------

MODEL_PATH = "models/multimodal_fusion_mlp_model.joblib"
SCALER_PATH = "models/multimodal_scaler.joblib"

class IDVerificationEngine:
    def __init__(self, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Trained model not found at '{model_path}'. Run Fast-Track Day 3 first.")
            
        print("[INFO] Loading trained Multimodal AI Model and Scaler...")
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.ocr_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        self.visual_extractor = VisualStreamExtractor()
        print("[INFO] Verification Engine initialized and ready.")

    def verify_single_image(self, image_path):
        """Processes a single unseen image and returns prediction details."""
        if not os.path.exists(image_path):
            return {"error": f"File '{image_path}' not found."}
            
        # 1. Extract 16-D Visual features
        vis_feats = self.visual_extractor.extract_features(image_path)
        
        # 2. Extract 11-D OCR features
        ocr_vec, ocr_dict = extract_ocr_features(image_path, self.ocr_engine)
        
        # 3. Concatenate into 27-D Multimodal Feature Vector
        fused_vec = np.concatenate([vis_feats, ocr_vec]).reshape(1, -1)
        
        # 4. Standardize and Predict
        fused_scaled = self.scaler.transform(fused_vec)
        prediction = int(self.model.predict(fused_scaled)[0])
        probabilities = self.model.predict_proba(fused_scaled)[0]
        
        forged_prob = float(probabilities[1])
        genuine_prob = float(probabilities[0])
        verdict = "FORGED" if prediction == 1 else "GENUINE"
        confidence = forged_prob if verdict == "FORGED" else genuine_prob
        
        # Diagnostic forensics
        diagnostics = []
        if ocr_dict.get("std_confidence", 0) > 0.025:
            diagnostics.append("High OCR confidence variance (suggests spliced/inconsistent text)")
        if ocr_dict.get("box_height_std", 0) > 12.0:
            diagnostics.append("Unusual font size variance across detected text lines")
        if vis_feats[8] > 2000: # Laplacian variance
            diagnostics.append("High localized high-frequency edge energy detected")
            
        if not diagnostics:
            diagnostics.append("Consistent visual texture and uniform print clarity")

        return {
            "image_path": image_path,
            "verdict": verdict,
            "prediction": prediction,
            "confidence_pct": round(confidence * 100, 2),
            "forged_probability_pct": round(forged_prob * 100, 2),
            "genuine_probability_pct": round(genuine_prob * 100, 2),
            "mean_ocr_conf": ocr_dict.get("mean_confidence", 0),
            "text_boxes_detected": ocr_dict.get("num_text_boxes", 0),
            "diagnostics": diagnostics
        }

    def verify_folder(self, folder_path, output_csv="verification_results.csv"):
        """Scans an entire directory of unseen images and exports a batch report."""
        supported_exts = ["*.png", "*.jpg", "*.jpeg"]
        image_files = []
        for ext in supported_exts:
            image_files.extend(glob.glob(os.path.join(folder_path, "**", ext), recursive=True))
            
        if not image_files:
            print(f"[ERROR] No image files found in '{folder_path}'.")
            return None
            
        print(f"\n[INFO] Found {len(image_files)} images to verify in '{folder_path}'. Running verification...")
        
        results = []
        for i, img_path in enumerate(image_files):
            res = self.verify_single_image(img_path)
            res["filename"] = os.path.basename(img_path)
            results.append(res)
            
            tag = "[FORGED]" if res["verdict"] == "FORGED" else "[GENUINE]"
            print(f"  [{i+1}/{len(image_files)}] {tag:<9} ({res['confidence_pct']:>5.1f}% cert) : {res['filename']}")
            
        df_res = pd.DataFrame(results)
        df_res.to_csv(output_csv, index=False)
        print(f"\n[SUCCESS] Verification report saved to: '{output_csv}'")
        return df_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake ID Verification Engine")
    parser.add_argument("--image", type=str, help="Path to single ID image")
    parser.add_argument("--folder", type=str, help="Path to a new dataset folder")
    args = parser.parse_args()

    engine = IDVerificationEngine()

    if args.image:
        print("\n" + "=" * 70)
        print("  SINGLE ID VERIFICATION RESULT")
        print("=" * 70)
        result = engine.verify_single_image(args.image)
        print(f"Target Image      : {result['image_path']}")
        print(f"VERDICT           : {result['verdict']}")
        print(f"Model Certainty   : {result['confidence_pct']}%")
        print(f"Forged Probability: {result['forged_probability_pct']}%")
        print(f"Genuine Probability: {result['genuine_probability_pct']}%")
        print(f"Fields Detected   : {result['text_boxes_detected']}")
        print(f"Average OCR Conf  : {result['mean_ocr_conf']}")
        print("\nForensic Diagnostics:")
        for diag in result['diagnostics']:
            print(f"  • {diag}")
        print("=" * 70)
        
    elif args.folder:
        engine.verify_folder(args.folder)
    else:
        # Default demo: test one genuine and one forged sample from the dataset
        demo_genuine = "dataset/national_id/genuine/national_id_genuine_1.png"
        demo_forged = "dataset/national_id/forged/national_id_forged_1.png"
        
        print("\n" + "=" * 70)
        print("  DEMO: RUNNING VERIFICATION ON UNSEEN SAMPLES")
        print("=" * 70)
        for path in [demo_genuine, demo_forged]:
            if os.path.exists(path):
                res = engine.verify_single_image(path)
                status_color = "FORGED" if res['verdict'] == "FORGED" else "GENUINE"
                print(f"\nFile: {os.path.basename(path)}")
                print(f"  -> Prediction : {res['verdict']} (Certainty: {res['confidence_pct']}%)")
                print(f"  -> Forged Prob: {res['forged_probability_pct']}%")
                print(f"  -> Clues: {', '.join(res['diagnostics'])}")
        print("=" * 70)
