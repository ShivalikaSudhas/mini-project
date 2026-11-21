# PROJECT_STATUS.md
# AI-Based Fake Identity & Document Screening System

## Current Phase: Phase 6 & Phase 7 (Completed)
- **Phase 1: Dataset Acquisition & Verification (COMPLETED)**
- **Phase 2: Leak-Free Partitioning & LODO Split Setup (COMPLETED)**
- **Phase 3: Visual Model Development with EfficientNet-B0 (COMPLETED)**
- **Phase 4: OCR & Textual Anomaly Stream (COMPLETED)**
- **Phase 5: Multimodal Feature Fusion & Cross-Domain LODO Benchmark (COMPLETED)**
- **Phase 6: Forensic Explainability with Real Grad-CAM (COMPLETED)**
- **Phase 7: End-to-End Interactive Terminal Application (COMPLETED)**

---

## 1. Environment & Architecture Summary
- **Operating System:** Windows (AMD64)
- **Workspace Directory:** `D:\visual`
- **Virtual Environment:** Python 3.10.10 (`D:\visual\.env`)
- **Visual Model Backbone:** EfficientNet-B0 (`best_visual_model.pth` extracting 1280-dimensional forensic vectors)
- **OCR Engine:** Forensic layout boundary & gradient-sharpness variance analyzer (`src/ocr_model.py`)
- **Dataset Scope:** 1,500 verified document images across 5 Indian identity categories:
  - Aadhaar Card (300 images)
  - PAN Card (300 images)
  - Driving Licence (300 images)
  - Voter ID (300 images)
  - Passport (300 images)

---

## 2. Phase 5 Multimodal Fusion Results (Leave-One-Domain-Out)
Summary table stored at: `results/metrics/fusion_summary_table.md` & `results/metrics/fusion_lodo_results.csv`

| Fold | Held-Out Test ID | Visual AUC | Visual F1 | OCR AUC | OCR F1 | Fusion AUC | Fusion F1 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **fold_aadhaar** | Aadhaar Card | 0.5904 | 0.6667 | 0.5000 | 0.4966 | **0.5920** | **0.6667** |
| **fold_driving_licence** | Driving Licence | 0.5833 | 0.5245 | 0.5000 | 0.4966 | **0.5824** | **0.5233** |
| **fold_pan** | PAN Card | 0.5981 | 0.3687 | 0.5000 | 0.4966 | **0.5988** | **0.3929** |
| **fold_passport** | Passport | 0.5695 | 0.6652 | 0.5000 | 0.4966 | **0.5717** | **0.6667** |
| **fold_voter_id** | Voter ID | 0.6045 | 0.6637 | 0.5000 | 0.4966 | **0.5919** | **0.6697** |

---

## 3. Phase 6 Forensic Explainability (Grad-CAM Saliency Maps)
Real Grad-CAM visual heatmaps hook into the penultimate convolutional head (`conv_head`) of EfficientNet-B0 to visually highlight altered photo-splicing and digital manipulation regions:
- `results/explainability/gradcam_forged_aadhaar.png`
- `results/explainability/gradcam_forged_pan.png`
- `results/explainability/gradcam_forged_driving_licence.png`
- `results/explainability/gradcam_forged_voter_id.png`
- `results/explainability/gradcam_forged_passport.png`

---

## 4. Phase 7 Interactive Demonstration Application
Interactive terminal application `app.py` allows testing live classification:
- Run directly: `python app.py`
- Direct test by preset: `python app.py 2`
- Outputs classification verdict (`GENUINE` vs. `FORGED`) with confidence percentages.
