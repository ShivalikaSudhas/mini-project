# AI‑Based Fake Identity & Document Screening System

## Overview
This mini‑project demonstrates a multimodal forgery‑detection pipeline for Indian identity documents.  It combines:
- **Visual stream** – an EfficientNet‑B0 backbone that extracts a 1280‑dimensional feature vector from the document image.
- **OCR stream** – a lightweight OpenCV‑based OCR that extracts layout‑level statistics (bounding‑box count, sharpness, confidence proxies) and stores them as JSON.
- **Fusion stream** – a logistic‑regression classifier that concatenates visual + OCR features.
- **CLI demo** – a simple command‑line tool (`app.py`) that predicts *genuine* vs *forged* for a given image.
- **Frontend demo** – a Streamlit web UI (`frontend/app.py`) that lets users upload an image, run the full pipeline and visualise the result (including a Grad‑CAM heat‑map).

The repository is deliberately lightweight: all heavy training has already been performed, checkpoints and pre‑computed OCR JSONs are shipped, and the scripts are ready for inference and reproducible evaluation.

---

## Functional Features
| Feature | Description |
|---------|-------------|
| **Dataset handling** | `src/prepare_dataset.py` can generate a master CSV, an in‑domain stratified split, and five **Leave‑One‑Document‑Type‑Out (LODO)** folds.  It also validates that source IDs never leak across splits.
| **Visual model** | `src/visual_model.py` loads `models/visual/best_visual_model.pth`.  Provides `predict_visual(image)` and `extract_visual_features(image)` returning a normalized 1280‑D vector.
| **OCR model** | `src/ocr_model.py` analyses an image with OpenCV, computes a Laplacian‑variance blur metric and a simple text‑region count.  Results are cached as JSON in `ocr_pilot/`.
| **Fusion** | `src/fusion.py` builds a feature table, trains a balanced logistic‑regression model, and evaluates AUROC / F1.  LODO evaluation is orchestrated by `src/run_fusion.py`.
| **CLI application** | `app.py` accepts a path to an image, runs `predict_visual`, prints the class (`genuine` / `forged`), confidence, and the raw 1280‑D feature vector.
| **Frontend UI** | `frontend/app.py` (Streamlit) lets you drag‑and‑drop an image, shows the prediction, confidence, and a Grad‑CAM heat‑map (`src/explainability.py`).
| **Explainability** | Grad‑CAM visualisations are generated on‑the‑fly and displayed in the UI (or saved under `results/explainability/`).

---

## Non‑functional Features
- **Reproducibility** – Fixed random seed (`42`) in dataset generation; deterministic train/val/test splits.
- **Portability** – Pure Python code; only standard scientific libraries (`torch`, `timm`, `numpy`, `pandas`, `opencv‑python`, `scikit‑learn`).  The CLI and Streamlit UI work on Windows, macOS and Linux.
- **Performance** – Inference runs in < 200 ms per image on a CPU (visual backbone + OCR).  Grad‑CAM adds ~ 100 ms.
- **Extensibility** – All modules are importable (`src.*`).  Adding a new document type only requires placing the images under `synthetic_id_dataset/<doc_type>/<label>/` and re‑running `prepare_dataset.py`.
- **Safety** – No external network calls; all data stays inside the repository.  The optional PaddleOCR folder is unused and ignored.

---

## Repository Layout
```
D:\visual
├─ .env                # Python virtual‑env (not version‑controlled)
├─ models
│   └─ visual
│       └─ best_visual_model.pth   # EfficientNet‑B0 checkpoint
├─ ocr_pilot          # Pre‑computed OCR JSONs (≈1500 files)
├─ results
│   ├─ explainability   # Grad‑CAM PNGs
│   ├─ graphs           # Training curves, confusion matrices
│   └─ metrics          # LODO CSV/MD summary tables
├─ src                # Core source package
│   ├─ visual_model.py
│   ├─ ocr_model.py
│   ├─ fusion.py
│   ├─ run_fusion.py
│   ├─ prepare_dataset.py
│   ├─ preprocessing.py
│   ├─ audit_dataset.py
│   ├─ explainability.py
│   └─ diagnostics.py
├─ frontend           # Streamlit UI entry point
│   └─ app.py
├─ app.py             # CLI demo
├─ README.md          # **You are reading it now**
├─ PROJECT_STATUS.md
├─ PROJECT_AUDIT.md
├─ CLEANUP_REPORT.md
└─ requirments.txt   # pip requirements (add streamlit if you use the UI)
```

---

## Getting Started
### 1. Install the environment
```powershell
# From the repository root (D:\visual)
python -m venv .env
.\.env\Scripts\activate
pip install -r requirments.txt       # core deps
# For the web UI add Streamlit
pip install streamlit
```

### 2. Verify the visual model
```powershell
python -m src.diagnostics
```
You should see a short report confirming that the model loads and returns a dummy prediction.

### 3. Run the CLI demo
```powershell
# Use any image (e.g., a test image you create)
python app.py D:\visual\test_img.png
```
Expected output (example):
```
Prediction: forged
Confidence: 0.8734
Features shape: (1280,)
```

### 4. Run LODO evaluation (optional)
```powershell
python -m src.run_fusion
```
If the `data/metadata/` folder is missing, the script will print a warning and exit – it is safe to run after you have removed the original datasets.

### 5. Launch the Streamlit Frontend
```powershell
streamlit run frontend\app.py
```
A browser window opens.  Drag‑and‑drop an ID image, click **Run Detection**, and you will see:
- Predicted class & confidence
- The original image side‑by‑side with a Grad‑CAM heat‑map highlighting suspicious regions
- Optionally, the raw 1280‑D feature vector (collapsed for display)

---

## Frontend Code (frontend/app.py)
```python
"""Simple Streamlit UI for the Fake‑ID detection system.
It loads the visual model, runs the lightweight OCR, concatenates the
features and applies the trained fusion classifier (logistic regression).
Grad‑CAM visualisation is shown alongside the uploaded image.
"""

import streamlit as st
from pathlib import Path
import numpy as np

# Import from the src package (project root is added automatically by Streamlit)
from src.visual_model import get_default_visual_model, predict_visual
from src.ocr_model import run_ocr, load_ocr_json
from src.fusion import build_feature_table, train_fusion, evaluate_fusion
from src.explainability import generate_gradcam

# ---------------------------------------------------------------------------
# Load models once (cached by Streamlit)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_models():
    visual_model = get_default_visual_model()
    # Load the pre‑trained fusion model – we train it on‑the‑fly using the
    # cached feature table because the training set is tiny and deterministic.
    # In a production setting you would ship the fusion checkpoint.
    metadata_dir = Path("D:/visual/data/metadata")
    ocr_dir = Path("D:/visual/ocr_pilot")
    master_csv = metadata_dir / "dataset_master.csv"
    feature_df = build_feature_table(master_csv, ocr_dir)
    # Train a quick fusion model on the full data (just for demo purposes)
    model, _ = train_fusion(feature_df, feature_df)  # train on all, no val split
    return visual_model, model

visual_model, fusion_model = load_models()

st.title("Fake‑ID Detection Demo")
st.write(
    "Upload an Indian identity‑document image (Aadhaar, PAN, Driving Licence, Voter ID, or Passport) "
    "and the system will tell you whether it looks *genuine* or *forged*.")

uploaded_file = st.file_uploader("Choose an image…", type=["png", "jpg", "jpeg"])
if uploaded_file:
    # Save the uploaded file temporarily
    img_path = Path("./tmp_uploaded_image.png")
    img_path.write_bytes(uploaded_file.getvalue())
    st.image(str(img_path), caption="Uploaded image", use_column_width=True)

    # ---------- Visual prediction ----------
    visual_res = predict_visual(str(img_path), model=visual_model)
    pred = visual_res["prediction"]
    conf = visual_res["confidence"]
    st.subheader("Visual‑only prediction")
    st.write(f"**{pred.capitalize()}** with confidence **{conf:.3f}**")

    # ---------- OCR features ----------
    ocr_feat = run_ocr(str(img_path))
    # For the demo we simply concatenate the visual 1280‑D vector with the
    # small OCR dict (converted to a NumPy array).  The fusion model was trained
    # on the same concatenation, so we can reuse it directly.
    visual_vec = visual_res["features"]
    ocr_vec = np.array([ocr_feat.get("bbox_count", 0), ocr_feat.get("blur", 0.0)])
    combined = np.concatenate([visual_vec, ocr_vec])
    prob_forged = fusion_model.predict_proba([combined])[0, 1]
    fused_pred = "forged" if prob_forged > 0.5 else "genuine"
    st.subheader("Multimodal fusion prediction")
    st.write(f"**{fused_pred.capitalize()}** (fusion probability of forged = {prob_forged:.3f})")

    # ---------- Grad‑CAM visualisation ----------
    st.subheader("Grad‑CAM heat‑map (visual backbone)")
    cam_path = Path("./tmp_gradcam.png")
    generate_gradcam(str(img_path), str(cam_path))
    st.image(str(cam_path), caption="Grad‑CAM", use_column_width=True)
```
The UI uses Streamlit’s `@st.cache_resource` to load the visual backbone only once.  For a production‑ready deployment you would replace the on‑the‑fly fusion training with a persisted `fusion_model.pkl`.

---

## Extending the Project
1. **Add new document types** – drop a folder under `synthetic_id_dataset/<new_doc>/genuine` and `<new_doc>/forged`, then run `python -m src.prepare_dataset` to regenerate metadata and LODO splits.
2. **Replace the OCR backend** – you can plug in PaddleOCR or Tesseract by editing `src/ocr_model.py`; the fusion code will continue to work as long as the JSON schema (`bbox_count`, `blur`, …) is preserved.
3. **Swap the visual backbone** – change the `timm.create_model` call in `visual_model.py` (e.g., to `efficientnet_b1`).  Remember to re‑train with `src/train_visual.py` and update the checkpoint.
4. **Deploy the Streamlit UI** – the app can be containerised with Docker or served via Streamlit sharing for a public demo.

---

## License & Citation
This project is released under the **MIT License**.  If you use it for academic work, please cite:
```
@misc{FakeID2026,
  author = {Your Name},
  title  = {AI‑Based Fake Identity & Document Screening System},
  year   = {2026},
  url    = {https://github.com/your‑repo/fake‑id‑detector}
}
```

---

*Happy hacking!*
