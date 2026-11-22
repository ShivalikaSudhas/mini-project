# Multimodal Fusion Stream: 27-D Feature Synthesis & Cross-Document LODTO Study
**Team Lead / Contributor:** Liza (Fusion Lead)  
**Project:** AI-Based Multimodal Fake Identity & Document Screening System

---

## 📌 Overview

Single-modality forgery screening is vulnerable to sophisticated attacks:
- **Visual-only CNNs** can be deceived by clean digital font replacements that lack resampling artifacts.
- **OCR-only text analyzers** can miss high-quality forged photos or altered holographic seals where text fonts are copied faithfully.

The **Multimodal Fusion Engine** synthesizes:
1. **16-Dimensional Visual Forensic Features** (MobileNet-V3 spatial latents + Laplacian edge sharpness + Sobel gradient variance + FFT frequency domain cutoff energy + Error Level Analysis compression residuals + Color covariance).
2. **11-Dimensional OCR Layout Features** (PaddleOCR text confidences + low confidence ratios + box geometries + font aspect ratio + box height standard deviation + character density).
3. **27-Dimensional Multimodal Feature Vector** fed into a trained Multi-Layer Perceptron (MLP) Classifier to produce the ultimate screening verdict.

---

## 📐 27-Dimensional Multimodal Feature Vector Composition

$$\mathbf{x}_{\text{fused}} = [\mathbf{v}_1, \dots, \mathbf{v}_{16}, \mathbf{o}_1, \dots, \mathbf{o}_{11}] \in \mathbb{R}^{27}$$

### Part A: 16-D Visual Forensic Features (`visual_feature_extractor.py`)
| # | Feature Key | Category | Forensic Intuition |
|---|-------------|----------|-------------------|
| 1–8 | `cnn_feat_1` to `cnn_feat_8` | Deep Spatial Latents | Compressed MobileNet-V3 spatial feature maps capturing macro structural patterns. |
| 9 | `laplacian_var` | Edge Sharpness | Variance of Laplacian filter (flags artificial blurring/smoothing around faces/text). |
| 10 | `laplacian_mean` | High-Pass Energy | Mean absolute Laplacian response. |
| 11 | `gradient_mean` | Gradient Field | Mean Sobel gradient magnitude across both axes. |
| 12 | `gradient_std` | Gradient Contrast | Standard deviation of Sobel gradient magnitudes. |
| 13 | `high_freq_ratio` | FFT Frequency Analysis | 2D Fast Fourier Transform high-frequency cutoff ratio (flags digital copy-paste seams). |
| 14 | `ela_mean` | Error Level Analysis | Mean JPEG resaving compression residual variance. |
| 15 | `ela_std` | Error Level Analysis | Standard deviation of JPEG compression residuals. |
| 16 | `color_std_ratio` | Color Covariance | Ratio of blue to red channel standard deviations ($B_{\sigma} / R_{\sigma}$). |

### Part B: 11-D OCR Layout Features (`ocr_feature_extractor.py`)
| # | Feature Key | Category | Forensic Intuition |
|---|-------------|----------|-------------------|
| 17 | `mean_confidence` | Confidence | Average PaddleOCR word recognition score. |
| 18 | `min_confidence` | Confidence | Minimum confidence score observed in a field. |
| 19 | `max_confidence` | Confidence | Peak confidence score observed. |
| 20 | `std_confidence` | Confidence | Standard deviation of field confidences. |
| 21 | `low_conf_count` | Confidence | Count of text boxes with confidence $< 0.85$. |
| 22 | `low_conf_ratio` | Confidence | Proportion of low-confidence text boxes. |
| 23 | `num_text_boxes` | Geometric | Total number of isolated text blocks. |
| 24 | `text_area_ratio` | Geometric | Ratio of text bounding box area to total card surface area. |
| 25 | `mean_aspect_ratio` | Geometric | Average width-to-height ratio ($W / H$). |
| 26 | `box_height_std` | Geometric | Variance in text box heights (font size inconsistency). |
| 27 | `mean_char_density` | Density | Character count per pixel of text box surface. |

---

## 🧠 Multimodal Fusion MLP Architecture

- **Model Type:** Multi-Layer Perceptron (`MLPClassifier`)
- **Input Dimension:** 27 numerical features (standardized via `StandardScaler`)
- **Hidden Layers:** `(64, 32)` with ReLU activations
- **Optimizer & Regularization:** Adam optimizer, L2 penalty ($\alpha = 0.0001$), maximum 500 iterations
- **Output:** Softmax probabilities for:
  - Class `0`: **GENUINE**
  - Class `1`: **FORGED**
- **Artifacts:**
  - `models/multimodal_fusion_mlp_model.joblib`
  - `models/multimodal_scaler.joblib`

---

## 🔬 Leave-One-Document-Type-Out (LODTO) Cross-Document Generalization Study

To prove rigorous anti-overfitting to academic evaluation committees, Liza conducted a **5-fold Leave-One-Document-Type-Out study** (`run_lodto_study.py`).
Instead of random splits, the model is trained on 4 document types and evaluated on a completely unseen 5th document type.

### Empirical Generalization Results (`lodto_experiment_results.csv`):
| Hold-Out Unseen Document Type | Multimodal AUROC | Multimodal Accuracy | Generalization Gap vs Random Split |
|-------------------------------|------------------|---------------------|-----------------------------------|
| Driver License | **98.2%** | **95.0%** | Minimal (< 3%) |
| Employee Badge | **97.8%** | **94.0%** | Minimal (< 3%) |
| Health Card | **99.1%** | **96.5%** | Minimal (< 2%) |
| National ID | **96.9%** | **93.5%** | Minimal (< 4%) |
| Student Card | **98.5%** | **95.5%** | Minimal (< 3%) |

**Key Finding:** Multimodal feature fusion achieves **superior cross-template generalization** compared to single-modality baselines, bridging the domain gap when evaluating entirely novel identity document templates.

---

## 🚀 Inference & API Usage

```python
from verify_new_id import verify_id_multimodal

result = verify_id_multimodal("path/to/any_id.jpg")
print("Verdict:", result["verdict"])                       # 'GENUINE' or 'FORGED'
print("Certainty:", result["confidence_pct"], "%")
print("Forged Probability:", result["forged_probability_pct"], "%")
print("Diagnostics:", result["diagnostics"])
print("27-D Features:", result["fused_features_27d"].shape)  # (27,)
```
