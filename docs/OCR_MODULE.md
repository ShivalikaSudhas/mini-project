# OCR Stream Module: PaddleOCR & Forensic Layout Engine
**Team Lead / Contributor:** Vidyaaa (OCR Lead)  
**Project:** AI-Based Multimodal Fake Identity & Document Screening System

---

## 📌 Overview

The OCR stream forms one half of the dual-branch screening pipeline. While the Visual Stream (EfficientNet-B0) detects microscopic digital resampling, spliced textures, and edge artifacts, the OCR stream focuses on:
1. **Multilingual Text Extraction & Alignment** via **PaddleOCR** (PP-OCRv3/v4).
2. **Forensic Geometry & Density Analysis** across isolated bounding boxes.
3. **Statistical Confidence Profiling** to detect optical recognition anomalies typical of tampered text fields.
4. **Standalone OCR Verification Classifier** (MLP Neural Network) trained on 11-dimensional feature vectors.

---

## 📐 11-Dimensional OCR Feature Vector

The module (`ocr_feature_extractor.py`) computes an 11-dimensional numerical vector representing statistical confidence and spatial layout metrics:

| # | Feature Name | Category | Description & Forensic Intuition |
|---|--------------|----------|----------------------------------|
| 1 | `mean_confidence` | Confidence | Average OCR recognition confidence across all fields. Forged documents frequently exhibit localized blurry text with lower mean confidence. |
| 2 | `min_confidence` | Confidence | Lowest confidence score observed in any single text region. Identifies spliced or pasted individual fields. |
| 3 | `max_confidence` | Confidence | Highest confidence score observed. Provides dynamic range calibration. |
| 4 | `std_confidence` | Confidence | Standard deviation of confidence scores. High variance indicates sharp genuine fields mixed with altered/low-quality fraudulent fields. |
| 5 | `low_conf_count` | Confidence | Total count of bounding boxes having confidence score `< 0.85`. |
| 6 | `low_conf_ratio` | Confidence | Fraction of text fields with low confidence (`low_conf_count / total_boxes`). |
| 7 | `num_text_boxes` | Geometric | Total number of isolated text regions. Forged templates often omit mandatory micro-text or official issuing labels. |
| 8 | `text_area_ratio` | Geometric | Ratio of total bounding box surface area to full document image area. |
| 9 | `mean_aspect_ratio` | Geometric | Average width-to-height ratio ($W / H$) of bounding boxes. Detects unnatural stretching or squashing of text lines. |
| 10 | `box_height_std` | Geometric | Variance in text box heights. In genuine IDs, standard font sizes are consistent; in forged IDs, text heights fluctuate due to manual font edits. |
| 11 | `mean_char_density` | Geometric | Total extracted character count divided by total bounding box area ($\text{chars} / \text{px}^2$). |

---

## 🧠 OCR-Only Classifier Architecture

Trained using `train_ocr_baseline.py` on the benchmark dataset:
- **Model Type:** Multi-Layer Perceptron (`MLPClassifier`)
- **Hidden Layers:** `(32, 16)` with ReLU activation
- **Preprocessing:** `StandardScaler` fitted on 11-D feature vectors
- **Target Output:** 
  - `0`: Genuine Document
  - `1`: Forged Document
- **Model Checkpoints:** 
  - `models/ocr_only_mlp_model.joblib`
  - `models/ocr_scaler.joblib`

---

## 🚀 Usage & API Contract

### 1. Extract 11-D Numerical Features
```python
from ocr_feature_extractor import extract_ocr_features

feature_vector, feature_dict = extract_ocr_features("path/to/id_card.jpg")
print(feature_vector.shape)  # (11,)
print(feature_dict)
```

### 2. Run OCR Inference & Forensic Text Bounding Boxes
```python
from ocr_feature_extractor import run_ocr

ocr_result = run_ocr("path/to/id_card.jpg")
# Returns:
# {
#     "boxes": [[x1, y1, x2, y2], ...],
#     "texts": ["Government of India", ...],
#     "confidences": [0.98, ...],
#     "engine": "PaddleOCR"  # or "Morphological Fallback"
# }
```

### 3. Run Standalone OCR Classification
```python
from ocr_feature_extractor import predict_ocr

prediction = predict_ocr("path/to/id_card.jpg")
print(prediction["prediction"])          # 'Genuine' or 'Forged'
print(prediction["confidence"])          # e.g., 0.9412
print(prediction["probability_forged"])  # e.g., 0.958
```

---

## 🛡️ Robust Offline / Restricted Environment Fallback

In environments where PaddleOCR network access or directory write permissions are restricted, `ocr_feature_extractor.py` implements a graceful fallback mechanism:
- Uses adaptive thresholding and morphological gradient analysis to isolate text regions.
- Computes localized Laplacian high-frequency edge sharpness to formulate confidence scores.
- Maintains the identical 11-D feature vector contract, ensuring zero disruption to Streamlit or evaluation pipelines.
