# PROJECT_STATUS.md
# AI-Based Fake Identity & Document Screening System

## Current Phase
- **Phase 2: Dataset Preparation (COMPLETED)**
- **Next Phase:** Phase 3 — Visual Model Development (EfficientNet-B0)

---

## 1. Environment & Hardware Summary
- **Operating System:** Windows (AMD64)
- **Workspace Directory:** `D:\visual`
- **Python Runtime:** Python 3.10.10 (virtual environment: `D:\visual\.env`)
- **PyTorch Installation:** PyTorch 2.14.1+cpu (CUDA: False / CPU only)
- **Installed Packages:** `torch`, `torchvision`, `numpy`, `pandas`, `cv2`, `PIL`, `sklearn`
- **Background Installs in Progress:** `timm`, `albumentations`, `tabulate`

---

## 2. Dataset Preparation Summary (Phase 2 Findings & Artifacts)
- **Master Metadata Generated:** `data/metadata/dataset_master.csv` (1,000 samples)
  - Unified Schema: `sample_id`, `source_id`, `doc_type`, `label`, `label_name`, `manip_type`, `filename`, `relative_path`, `absolute_path`, `width`, `height`, `channels`
- **In-Domain Stratified Split:** `data/metadata/split_in_domain.csv`
  - Grouped by `source_id` to strictly prevent train-test contamination.
  - Train: 700 samples (350 genuine, 350 forged — 70 source pairs per document type)
  - Validation: 150 samples (75 genuine, 75 forged — 15 source pairs per document type)
  - Test: 150 samples (75 genuine, 75 forged — 15 source pairs per document type)
  - Zero leakage guarantee verified across all splits.
- **Leave-One-Document-Type-Out (LODO) 5-Fold Protocol:**
  - Fold 1 (`aadhaar` held out): 680 Train, 120 Val, 200 Test (`lodo_fold_aadhaar.csv`)
  - Fold 2 (`driving_licence` held out): 680 Train, 120 Val, 200 Test (`lodo_fold_driving_licence.csv`)
  - Fold 3 (`pan` held out): 680 Train, 120 Val, 200 Test (`lodo_fold_pan.csv`)
  - Fold 4 (`passport` held out): 680 Train, 120 Val, 200 Test (`lodo_fold_passport.csv`)
  - Fold 5 (`voter_id` held out): 680 Train, 120 Val, 200 Test (`lodo_fold_voter_id.csv`)
  - Summary: `data/metadata/lodo_folds_summary.csv`
- **Documentation:** `data/metadata/dataset_preparation_report.md`

---

## 3. Completed Tasks
- [x] Initialized Git repository and created baseline commit for Phase 1.
- [x] Developed automated data preparation pipeline (`src/prepare_dataset.py`).
- [x] Standardized `sample_id` and `source_id` naming conventions.
- [x] Generated master metadata CSV (`data/metadata/dataset_master.csv`).
- [x] Created source-ID grouped In-Domain split (`data/metadata/split_in_domain.csv`).
- [x] Created all 5 LODO cross-validation fold splits.
- [x] Formally verified zero data leakage and 50:50 class balance across every split.
- [x] Generated Phase 2 technical documentation.

---

## 4. Next Recommended Task
- **Phase 3 — Visual Model Development:**
  - Build the visual branch using `EfficientNet-B0`.
  - Implement preprocessing and augmentation for EfficientNet-B0.
  - Implement `predict_visual(image)` and `extract_visual_features(image)`.
  - Train on in-domain training set with frozen backbone, then fine-tune.
  - Evaluate on in-domain test set (Accuracy, Precision, Recall, F1, AUROC, Confusion Matrix).
