# Dataset Preparation Report — Phase 2

## 1. Schema Standardization
- **`sample_id`**: Format `<doc_type>_<label>_<id:03d>` (e.g. `aadhaar_genuine_000`, `aadhaar_forged_000`)
- **`source_id`**: Format `<doc_type>_<id:03d>` (e.g. `aadhaar_000`) shared across original & altered pairs
- **`doc_type`**: `aadhaar`, `driving_licence`, `pan`, `passport`, `voter_id`
- **`label`**: `0` for Genuine, `1` for Forged
- **`manip_type`**: `none` (genuine) | `photo_splicing_and_text_alteration` (forged)

## 2. In-Domain Stratified Split (Source-ID Grouped)
- **Train**: 700 samples (350 genuine, 350 forged) — 70 source pairs per doc type
- **Val**: 150 samples (75 genuine, 75 forged) — 15 source pairs per doc type
- **Test**: 150 samples (75 genuine, 75 forged) — 15 source pairs per doc type
- **Leakage Guarantee**: Zero source-id overlap between train, val, and test splits.

## 3. Leave-One-Document-Type-Out (LODO) 5-Fold Design
held_out_doc_type  train_samples  val_samples  test_samples (held-out)                              train_doc_types
          aadhaar            680          120                      200     driving_licence, pan, passport, voter_id
  driving_licence            680          120                      200             aadhaar, pan, passport, voter_id
              pan            680          120                      200 aadhaar, driving_licence, passport, voter_id
         passport            680          120                      200      aadhaar, driving_licence, pan, voter_id
         voter_id            680          120                      200      aadhaar, driving_licence, pan, passport

## 4. Verification Check Results
- Total master samples: 1,000
- Missing files: 0
- Corrupted images: 0
- Split leakages detected: 0
- Class balance across splits: Exactly 50% Genuine / 50% Forged in all folds
