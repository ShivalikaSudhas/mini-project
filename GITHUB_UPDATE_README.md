# Fusion / Generalization Refresh — GitHub Update

This update adds a reproducible classifier refresh and a Training Results Dashboard to the existing Streamlit website.

## Files to copy into the repository
- `frontend/app.py` — updated existing frontend; adds the Training Results Dashboard tab and points the LODTO tab to the current LODTO graphs.
- `retrain_from_feature_tables.py` — repeatable group-isolated MLP training script.
- `lodto_experiment_results.csv` — freshly regenerated five-fold LODTO results.
- `results/retrained_current_dataset/` — metrics, trained classifiers/scalers, graphs, and PDF report.

## Reproduce
Run from the repository root:
```bash
python retrain_from_feature_tables.py
python run_lodto_study.py
streamlit run frontend/app.py
```

## What was retrained
The classifier heads (visual-only, OCR-only, and 27-D multimodal MLP) were retrained from the repository's existing `multimodal_fused_dataset.csv` feature table. The split is 80/20 GroupShuffleSplit, with grouping by document type + language + filename; there is no group overlap between train and test.

This is classifier retraining from precomputed features. It does NOT re-extract features from raw images and does NOT retrain the EfficientNet-B0 backbone. The report clearly records this limitation.

## Main holdout results
- Visual-only accuracy: 61.03%
- OCR-only accuracy: 64.95%
- Multimodal fusion accuracy: 67.40%

## LODTO summary
- OCR-only mean unseen accuracy: 53.22%
- Visual-only mean unseen accuracy: 50.03%
- Multimodal mean unseen accuracy: 47.93%

LODTO results show that cross-document generalization remains challenging. Do not describe these results as near-98% accuracy or claim fusion wins on every unseen document type.

## GitHub upload
In GitHub, replace `frontend/app.py`, add `retrain_from_feature_tables.py`, replace `lodto_experiment_results.csv`, and upload the complete `results/retrained_current_dataset/` folder. Keep the repository's existing files and folders.
