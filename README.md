# Fake ID Screening: Cross-Document Generalization with Multimodal Visual + OCR Feature Fusion

**Computer Science Engineering Mini-Project — Codebase & Verification Suite**

---

## 📁 Clean Repository Structure

```text
mini_project/
│
├── dataset/                     # Balanced 5-document synthetic benchmark
│   ├── driver_license/          # Genuine & Forged samples
│   ├── employee_badge/          # Genuine & Forged samples
│   ├── health_card/             # Genuine & Forged samples
│   ├── national_id/             # Genuine & Forged samples
│   └── student_card/            # Genuine & Forged samples
│
├── models/                      # Saved trained models & feature scalers
│   ├── multimodal_fusion_mlp_model.joblib
│   ├── multimodal_scaler.joblib
│   ├── ocr_only_mlp_model.joblib
│   ├── ocr_scaler.joblib
│   ├── visual_only_mlp_model.joblib
│   └── visual_scaler.joblib
│
├── reports/                     # Publication-grade evaluation figures
│   ├── lodto_generalization_gap.png
│   ├── lodto_fold_breakdown.png
│   └── roc_curves_comparison.png
│
├── ocr_feature_extractor.py     # 11-D OCR numerical feature extractor module
├── visual_feature_extractor.py  # 16-D Visual forensic + CNN feature extractor module
├── dataset_generator.py         # Synthetic benchmark generator & OCR batch processor
│
├── train_ocr_baseline.py        # Trains & evaluates Condition 1 (OCR-Only)
├── train_visual_baseline.py     # Trains & evaluates Condition 2 (Visual-Only)
├── train_multimodal_fusion.py   # Trains & evaluates Condition 3 (Multimodal Fusion)
├── run_lodto_study.py           # Executes the 5-fold cross-document generalization study
├── generate_plots.py            # Generates publication charts in reports/
│
├── verify_new_id.py             # Inference tool: verify ANY new image or dataset folder
├── requirements.txt             # Verified environment package dependencies
│
├── ocr_features_dataset.csv     # Extracted 11-D OCR feature matrix
├── visual_features_dataset.csv  # Extracted 16-D Visual feature matrix
├── multimodal_fused_dataset.csv # Combined 27-D Multimodal feature matrix
└── lodto_experiment_results.csv # Empirical fold-by-fold cross-document results
```

---

## 🚀 Quickstart Guide for Teammates

### 1. Setup Virtual Environment & Dependencies
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

### 2. Verify ANY New Image or Dataset (Inference Mode)
You do **not** need to retrain the model; pre-trained weights are saved in `models/`.

* **To verify a single ID image:**
  ```powershell
  python verify_new_id.py --image "path/to/any_id.png"
  ```

* **To verify an entire folder / new dataset:**
  ```powershell
  python verify_new_id.py --folder "path/to/new_dataset_folder"
  ```
  *(Exports full predictions and certainty percentages to `verification_results.csv`)*

---

### 3. Re-Running the Experimental Pipeline (Optional)

1. **Re-generate Dataset & Features:**
   ```powershell
   python dataset_generator.py
   ```
2. **Train & Evaluate OCR Baseline (Condition 1):**
   ```powershell
   python train_ocr_baseline.py
   ```
3. **Train & Evaluate Visual Baseline (Condition 2):**
   ```powershell
   python train_visual_baseline.py
   ```
4. **Train & Evaluate Multimodal Fusion (Condition 3):**
   ```powershell
   python train_multimodal_fusion.py
   ```
5. **Run the LODTO Generalization Study:**
   ```powershell
   python run_lodto_study.py
   ```
6. **Generate Evaluation Plots:**
   ```powershell
   python generate_plots.py
   ```
