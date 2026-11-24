# TECHNICAL SPECIFICATION SHEET
## AI-Based Fake Identity & Document Screening System

**Document Version:** 2.0.0 (Final Production & Multi-Dataset Release)  
**Date of Audit & Compilation:** November 25, 2026  
**Workspace Path:** `D:\visual`  
**Git Branch of Record:** `latest`  
**Target Audience:** Project Report, Technical Viva Committee, Faculty Evaluators, and Academic Defense  

---

### VERBATIM PROJECT OBJECTIVES (Unaltered)
1. *To collect and preprocess a dataset of genuine and forged/tampered identity document images, including synthetic and augmented samples representing diverse tampering styles.*
2. *To design and train a dual-stream deep learning model that combines visual forensic features (texture, noise, edge artifacts) with OCR-extracted textual field consistency to classify documents as authentic or forged and localize tampered regions.*
3. *To evaluate detection performance both on tampering styles seen during training and on unseen, novel manipulation styles, and to develop a simple interface for uploading a document and viewing the authenticity verdict.*

---

## 1. PROJECT IN 60 SECONDS

* **The Problem:** Identity fraud using digitally manipulated Aadhaar cards, PAN cards, Driving Licences, Passports, and Voter IDs undermines digital Know-Your-Customer (e-KYC), financial authentication, and border screening. Modern digital tampering (text splicing, regional Gaussian blur, mobile phone recapture distortion, and photo substitution) leaves subtle multi-scale traces that human screeners and single-modality classifiers frequently fail to detect.
* **The Solution:** A unified tri-stream forensic screening platform that couples deep spatial embeddings with mathematical texture/frequency descriptors and OCR typographical layout consistency metrics into an integrated 27-dimensional multimodal decision engine.
* **Input:** A single RGB image of an Indian identity document (JPEG/PNG format, native resolution).
* **Output:**
  1. Authenticity verdict: `GENUINE` or `FORGED`.
  2. Multi-stream confidence breakdown: Visual CNN ($40\%$), OCR Layout ($30\%$), and 27-D MLP Fusion ($30\%$).
  3. Spatial Localization: Contrastive Grad-CAM attention heatmap highlighting tampered pixel coordinates.
  4. Dynamic Statistical XAI: Human-readable narrative detailing exact Z-score deviations from authentic baseline cards.
* **Tested Documents:** Aadhaar Cards, PAN Cards, Voter ID Cards, Passports, and Driving Licences.

---

## 2. REPOSITORY ARCHITECTURE & FILE INVENTORY

```
D:\visual
├── .env/                                # Python 3.10.10 Virtual Environment (NumPy 1.26.4, PyTorch, Streamlit)
├── .streamlit/
│   └── config.toml                      # Theme configuration enforcing light editorial palette (#F1F2FA / #9498B2)
├── dataset/
│   └── dataset/                         # Real-world format PAN and Voter ID cards (46 images)
│       ├── original/                    # 23 authentic reference documents
│       └── forged/                      # 23 digitally tampered documents
├── final_processed_dataset/             # Dedicated Aadhaar Card Dataset (164 images)
│   ├── genuine/                         # 41 clean high-resolution Aadhaar cards (gen_0 to gen_40)
│   └── forged/                          # 123 tampered Aadhaar cards across 3 tampering styles:
│       ├── forged_blur_*.png            # 41 regional blur & boundary smoothing samples
│       ├── forged_text_*.png            # 41 digital text splicing & font mismatch samples
│       └── forged_distort_*.png         # 41 phone recapture & geometric warping samples
├── demo_samples/                        # Verified benchmark specimens loaded by the web UI:
│   ├── aadhaar_genuine.jpg              # Tested genuine Aadhaar (evaluates to 31.9% -> GENUINE)
│   ├── aadhaar_forged.jpg               # Tested forged Aadhaar (evaluates to 84.9% -> FORGED)
│   ├── pan_genuine.jpg                  # Tested genuine PAN card (evaluates to 40.6% -> GENUINE)
│   ├── pan_forged.jpg                   # Tested forged PAN card (evaluates to 80.7% -> FORGED)
│   ├── voter_id_genuine.jpg             # Tested genuine Voter ID
│   ├── voter_id_forged.jpg              # Tested forged Voter ID
│   ├── passport_genuine.jpg             # Tested genuine Passport
│   ├── passport_forged.jpg              # Tested forged Passport
│   ├── driving_licence_genuine.jpg      # Tested genuine Driving Licence
│   └── driving_licence_forged.jpg       # Tested forged Driving Licence
├── docs/
│   ├── PROJECT_SPEC_SHEET.md            # Master technical specification sheet (this file)
│   └── VIVA_CHEATSHEET.md               # Quick oral defense summary and trap question guide
├── frontend/
│   └── app.py                           # 4-tab Streamlit 1.65.0 web application (Magazine Editorial UI)
├── models/
│   ├── visual/
│   │   ├── best_visual_model.pth        # Fine-tuned EfficientNet-B0 checkpoint (17.6 MB)
│   │   └── best_visual_model_phase3_backup.pth # Archive of Phase 3 baseline
│   ├── visual_only_mlp_model.joblib     # Visual-Only MLP classifier
│   ├── ocr_only_mlp_model.joblib        # OCR Layout MLP classifier
│   ├── multimodal_fusion_mlp_model.joblib # 27-D Multimodal Fusion MLP classifier
│   ├── visual_scaler.joblib             # 16-D StandardScaler
│   ├── ocr_scaler.joblib                # 11-D StandardScaler
│   ├── multimodal_scaler.joblib         # 27-D StandardScaler
│   ├── genuine_feature_baseline.json    # Calibrated empirical baseline (mean/std) for XAI Z-scores
│   └── retrained_evaluation_metrics.json# Test split evaluation metrics across all streams
├── src/
│   ├── visual_model.py                  # VisualForensicModel (EfficientNet-B0 + dynamic MLP head)
│   └── explainability.py               # Contrastive Grad-CAM hook and Z-score XAI generator
├── tests/
│   ├── test_pipeline.py                 # 15 end-to-end integration test suites (100% passing)
│   └── test_tampered_split.py           # Split validation test suite
├── ocr_feature_extractor.py             # 11-D typographical layout extractor with morphological fallback
├── visual_feature_extractor.py          # 16-D visual feature extractor (8 deep latents + 8 texture/ELA/FFT)
├── verify_new_id.py                     # Single-image command-line verification engine
└── train_final_system.py                # Turnkey end-to-end training, calibration, and sample export pipeline
```

---

## 3. ACTIVE DATASETS SPECIFICATION

The system operates on **two complementary datasets** representing realistic and programmatically generated tampering:

### 3.1 `final_processed_dataset` (Aadhaar Card Focus)
* **Total Images:** 164 images.
* **Class Distribution:**
  * **Genuine:** 41 authentic Aadhaar cards (`gen_0.png` through `gen_40.png`).
  * **Forged:** 123 tampered cards categorized into 3 realistic fraud mechanisms:
    1. `forged_blur` (41 images): Gaussian blur filtering applied over citizen names and numbers to erase forensic micro-text.
    2. `forged_text` (41 images): Digitally inserted typography with mismatched stroke width, altered kerning, and pixel contrast seams.
    3. `forged_distort` (41 images): Mobile phone recapture artifacts, perspective skew, and non-linear geometric barrel distortion.

### 3.2 `dataset/dataset` (PAN Card & Voter ID Focus)
* **Total Images:** 46 images.
* **Class Distribution:**
  * **Original / Genuine:** 23 real-world template identity cards (PAN cards, Voter IDs).
  * **Forged / Tampered:** 23 corresponding tampered cards exhibiting text splicing, photo replacement, and badge cloning.

### 3.3 Unified Combined Training Corpus
* **Total Size:** 210 images (64 Genuine, 146 Forged).
* **Partitioning:** Stratified 80/20 train/test split:
  * **Training Split:** 168 images (51 Genuine, 117 Forged).
  * **Testing Split:** 42 unseen images (13 Genuine, 29 Forged).

---

## 4. MULTIMODAL FEATURE ARCHITECTURE (27 DIMENSIONS)

The multimodal fusion vector $\mathbf{x} \in \mathbb{R}^{27}$ concatenates the **16-D Visual Stream** and the **11-D OCR Stream**:

$$\mathbf{x}_{\text{multimodal}} = [\mathbf{v}_{\text{visual}} \;\Vert\; \mathbf{v}_{\text{ocr}}]^T$$

### 4.1 Visual Forensic Features (16-D) — Shivalika Sudhas
* **Features 0–7 (`cnn_feat_1` to `cnn_feat_8`):** Deep spatial latents obtained by partitioning the 1280-dimensional `conv_head` activation tensor from EfficientNet-B0 into 8 uniform bins and spatially pooling each bin.
* **Feature 8 (`laplacian_var`):** Variance of the discrete Laplacian operator $\operatorname{Var}(\nabla^2 I)$. Quantifies regional edge blur and Gaussian smoothing.
* **Feature 9 (`laplacian_mean`):** Mean absolute Laplacian energy $\mathbb{E}[|\nabla^2 I|]$, capturing global edge density.
* **Feature 10 (`gradient_mean`):** First-order Sobel gradient magnitude mean $\mathbb{E}[\sqrt{S_x^2 + S_y^2}]$, tracking overall card sharpness.
* **Feature 11 (`gradient_std`):** First-order Sobel standard deviation $\operatorname{Std}(G)$, detecting sharp cut-and-paste splicing boundaries.
* **Feature 12 (`high_freq_ratio`):** 2D Fast Fourier Transform (FFT) high-frequency corner energy ratio. Identifies periodic interpolation artifacts and digital resampling grids.
* **Features 13–14 (`ela_mean`, `ela_std`):** Error Level Analysis (ELA). Resaves image at JPEG Quality 90 and measures the residual $|I - \text{JPEG}_{90}(I)|$. Foreign pasted elements produce higher compression variance.
* **Feature 15 (`color_std_ratio`):** Ratio of blue-to-red chrominance standard deviations $\frac{\operatorname{Std}(I_B)}{\operatorname{Std}(I_R) + \epsilon}$, detecting color temperature mismatches between card substrate and pasted patches.

### 4.2 OCR Typographical Features (11-D) — Vidyashree
* **Features 16–21 (Confidence Metrics):** `mean_confidence`, `min_confidence`, `max_confidence`, `std_confidence`, `low_conf_count`, `low_conf_ratio`. Tracks character recognition stability (tampered characters produce erratic confidence drops).
* **Features 22–26 (Geometric Layout Metrics):** `num_text_boxes`, `text_area_ratio`, `mean_aspect_ratio`, `box_height_std` (flags font-size inconsistency), and `mean_char_density` (characters per unit box area).

---

## 5. MODELS & TRAINING PIPELINE

### 5.1 Deep Visual Model: EfficientNet-B0
* **Backbone:** EfficientNet-B0 (`timm`), 5.3M parameters.
* **Classification Head:** 2-layer MLP (`Linear(1280, 256) -> BatchNorm1d -> ReLU -> Dropout(0.2) -> Linear(256, 2)`).
* **Checkpoint:** `models/visual/best_visual_model.pth` (17.6 MB).
* **Training Schedule (Google Colab GPU):**
  * *Phase 1 (5 epochs):* Backbone frozen; AdamW optimizer ($\text{lr} = 10^{-3}$) trains the classification head.
  * *Phase 2 (10 epochs):* Entire network fine-tuned with differential learning rates ($\text{lr}_{\text{backbone}} = 10^{-4}$, $\text{lr}_{\text{head}} = 5 \times 10^{-4}$) under a Cosine Annealing scheduler.

### 5.2 Multimodal Fusion Classifier (PyTorch & Scikit-Learn)
* **Architecture:** Multi-Layer Perceptron (MLP) with hidden layers $(64, 32)$, ReLU activations, and early stopping.
* **Input:** Scaled 27-dimensional multimodal vector via `StandardScaler`.
* **Output:** Calibrated posterior probabilities $[P(\text{Genuine}), P(\text{Forged})]$.

---

## 6. EMPIRICAL TEST EVALUATION & BENCHMARKS

Evaluated on the held-out unseen test split ($N=42$ samples, 13 genuine, 29 forged):

| Stream Classifier | Test Accuracy | Test F1-Score | Precision | Recall | AUROC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Visual-Only Stream** | 61.90% | 0.7647 | 0.6667 | 0.8966 | 0.4589 |
| **OCR Layout Stream** | 66.67% | 0.8000 | 0.6829 | **0.9655** | **0.5623** |
| **27-D Multimodal Fusion** | **66.67%** | **0.7941** | **0.6923** | 0.9310 | 0.5570 |

### 6.1 Verified Live Demo Card Inferences
Evaluated through the end-to-end consensus pipeline ($0.40 \cdot \text{Vis} + 0.30 \cdot \text{OCR} + 0.30 \cdot \text{MLP}$ at threshold $\theta = 0.60$):

1. **`aadhaar_genuine.jpg` (gen_2.png):**
   * Visual CNN: $0.2\%$ | OCR: $58.9\%$ | MLP Fusion: $47.1\%$
   * **Consensus Score:** **31.9% Forged $\longrightarrow$ VERDICT: GENUINE** (68.1% Confidence)
2. **`aadhaar_forged.jpg` (forged_distort_0.png):**
   * Visual CNN: $96.5\%$ | OCR: $80.1\%$ | MLP Fusion: $74.2\%$
   * **Consensus Score:** **84.9% Forged $\longrightarrow$ VERDICT: FORGED** (84.9% Confidence)
3. **`pan_genuine.jpg` (Yuvraj-Mom-Pan-Card):**
   * Visual CNN: $2.3\%$ | OCR: $60.4\%$ | MLP Fusion: $71.9\%$
   * **Consensus Score:** **40.6% Forged $\longrightarrow$ VERDICT: GENUINE** (59.4% Confidence)
4. **`pan_forged.jpg` (forged_PAN-Card-Sample):**
   * Visual CNN: $97.2\%$ | OCR: $56.2\%$ | MLP Fusion: $83.4\%$
   * **Consensus Score:** **80.7% Forged $\longrightarrow$ VERDICT: FORGED** (80.7% Confidence)

---

## 7. EXPLAINABLE AI (XAI) & TAMPER LOCALIZATION

### 7.1 Contrastive Grad-CAM Heatmap
* **Layer Hooked:** `model.backbone.conv_head` (producing $A \in \mathbb{R}^{1280 \times 7 \times 7}$).
* **Contrastive Score:** $y^c = \text{logits}[\text{Forged}] - \text{logits}[\text{Genuine}]$. Isolates features that actively push the prediction toward *forged* relative to *genuine*.
* **Global Pooling & Rectification:**
  $$\alpha_k^c = \frac{1}{Z}\sum_{i}\sum_{j}\frac{\partial y^c}{\partial A_{i,j}^k}, \quad L_{\text{Grad-CAM}} = \operatorname{ReLU}\left(\sum_k \alpha_k^c A^k\right)$$
* **Visualization:** Min-max normalized, upscaled bilinearly to input resolution, and alpha-blended with OpenCV Jet colormap ($60\%$ original image $+ 40\%$ heatmap).

### 7.2 Dynamic Statistical Z-Score Attribution
* **Baseline Repository:** `models/genuine_feature_baseline.json`, containing mean ($\mu_i$) and standard deviation ($\sigma_i$) of all 27 features profiled from authentic identity cards.
* **Attribution Rule:** At runtime, the system calculates $z_i = \frac{x_i - \mu_i}{\sigma_i}$. Any feature with $|z_i| \ge 1.75$ is automatically output in plain English, for example:
  > *"Laplacian variance (28.4) deviated by -2.4σ from authentic cards (μ=142.1), indicating regional blur and edge-smoothing manipulation."*

---

## 8. FRONTEND WEB APPLICATION

* **Framework:** Streamlit 1.65.0 (`frontend/app.py`).
* **Design Philosophy:** Soft Editorial Lavender Publication:
  * `#FEFEFE` (Paper): Floating content cards.
  * `#F1F2FA` (Canvas): Page background canvas.
  * `#D2D6E4` (Elevated Surface): Navigation sidebar and primary cards.
  * `#9498B2` (Accent): Editorial highlights and badges.
  * `#141415` / `#525355`: High-contrast editorial typography.
* **Layout Structure:** 68/32 Asymmetrical Grid:
  * **Left Pane (68%):** Document preview, interactive OCR bounding box toggles, Grad-CAM attention heatmaps, and 27-D feature inspector.
  * **Right Pane (32%):** Forensic assessment card, consensus probability gauge, and stream-by-stream breakdown.
* **Interactive Controls:** Dynamic decision threshold slider ($\theta \in [0.40, 0.80]$), file uploader, and preset selector for all 5 document types.

---

## 9. TEAM RESPONSIBILITY MATRIX

| Team Member | Official Role | Core Engineering Contributions |
| :--- | :--- | :--- |
| **Shivalika Sudhas** | **Team Leader & Visual Lead** | EfficientNet-B0 visual stream, 16-D visual feature extractor (texture/FFT/ELA), Grad-CAM explainability engine, tri-stream consensus arbitration, and system integration. |
| **Vidyashree (Vidyaaa)** | **OCR Lead** | 11-D typographical layout analyzer, bounding-box geometry profiling, font height variance tracking, and morphological fallback engine. |
| **Liza** | **Fusion Lead** | 27-D multimodal feature synthesis, PyTorch/Scikit-Learn MLP fusion classifiers, LODTO cross-document study, and single-image verification tool. |

---

## 10. SYSTEM REQUIREMENTS & REPRODUCTION

```bash
# 1. Activate Environment
.\.env\Scripts\activate

# 2. Run Comprehensive Test Suite (All 15 Tests Pass)
pytest tests/ -v

# 3. Launch the Interactive Screening Web Application
streamlit run frontend/app.py
```
