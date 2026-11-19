# Dataset Audit Report — Phase 1
## AI-Based Fake Identity & Document Screening System

- **Dataset Root Path:** `D:\visual\synthetic_id_dataset`
- **Total Image Files:** 1000
- **Corrupted Files:** 0

### 1. Class Distribution Across Document Types

label            forged  genuine  total
doc_type                               
aadhaar             100      100    200
driving_licence     100      100    200
pan                 100      100    200
passport            100      100    200
voter_id            100      100    200

### 2. Image Properties & Quality Metrics

- **Formats:** `{"JPEG": 1000}`
- **Color Modes:** `{"RGB": 1000}`
- **Resolutions:** `{"320x200": 1000}`
- **Aspect Ratio (Mean):** `1.6`
- **File Size Range:** 6384 to 12318 bytes (Mean: 8001.9 B)
- **Blur Metric (Laplacian Var):** Min=2622.23, Max=7822.14, Mean=3965.79, Median=3286.22

### 3. Duplicate and Data Leakage Analysis

- **Exact Duplicates (SHA-256):** 0 duplicate clusters found.
- **Perceptual Near-Duplicates (dHash):** 37 clusters found.

### 4. Source-Document Relationship & Pairing Analysis

| Document Type | Genuine Count | Forged Count | Shared Filename Stems | Paired by Filename? |
| :--- | :--- | :--- | :--- | :--- |
| aadhaar | 100 | 100 | 100 | True |
| driving_licence | 100 | 100 | 100 | True |
| pan | 100 | 100 | 100 | True |
| passport | 100 | 100 | 100 | True |
| voter_id | 100 | 100 | 100 | True |

### 5. Document Authenticity and Synthesis Nature

- **Nature of Dataset:** The documents are programmatically generated synthetic specimen identity cards (`synthetic_id_dataset`).
- **Privacy Impact:** Fictional synthetic data — no live real-world Indian PII detected.
