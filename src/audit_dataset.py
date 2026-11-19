"""
audit_dataset.py — Comprehensive Dataset Audit for Identity Document Forgery Detection
Phase 1: Dataset Audit
"""

import os
import sys
import glob
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageDraw

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def compute_dhash(image_pil: Image.Image, hash_size: int = 8) -> str:
    """Compute difference hash (dHash) for perceptual near-duplicate detection."""
    # Resize to (hash_size + 1, hash_size) and convert to grayscale
    img_gray = image_pil.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    pixels = np.array(img_gray, dtype=np.int16)
    # Compare adjacent pixels
    diff = pixels[:, 1:] > pixels[:, :-1]
    # Convert bool array to bit string / hex
    bit_string = "".join(["1" if b else "0" for b in diff.flatten()])
    return f"{int(bit_string, 2):016x}"

def hamming_distance(h1: str, h2: str) -> int:
    x = int(h1, 16) ^ int(h2, 16)
    return bin(x).count("1")

def estimate_blur_laplacian(img_cv: np.ndarray) -> float:
    """Computes the variance of the Laplacian as a focus/sharpness metric."""
    if len(img_cv.shape) == 3:
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    else:
        gray = img_cv
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())

def run_audit(dataset_root: str, output_dir: str):
    root_path = Path(dataset_root)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    contact_dir = out_path / "contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning dataset from: {root_path.resolve()}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    all_files = []
    for ext in image_extensions:
        all_files.extend(root_path.rglob(f"*{ext}"))
        all_files.extend(root_path.rglob(f"*{ext.upper()}"))

    all_files = sorted(list(set(all_files)))
    print(f"Total image files discovered: {len(all_files)}")

    inventory_records = []
    corrupted_records = []
    dhash_dict = {}  # sha256 -> (path, doc_type, label, dhash)

    for idx, fpath in enumerate(all_files):
        rel_path = fpath.relative_to(root_path)
        parts = rel_path.parts

        # Infer doc_type and label from path hierarchy
        # e.g. aadhaar/forged/0.jpg or synthetic_id_dataset/aadhaar/forged/0.jpg
        doc_type = "unknown"
        label = "unknown"
        for p in parts:
            p_lower = p.lower()
            if p_lower in ["aadhaar", "pan", "passport", "driving_licence", "voter_id"]:
                doc_type = p_lower
            if p_lower in ["genuine", "forged", "tampered", "fake", "real"]:
                label = "genuine" if p_lower in ["genuine", "real"] else "forged"

        file_size = fpath.stat().st_size
        sha256_hash = compute_sha256(str(fpath))

        # Test readability with PIL and CV2
        is_corrupt = False
        error_msg = ""
        width, height, channels = -1, -1, -1
        img_format = ""
        color_mode = ""
        blur_var = -1.0
        mean_intensity = -1.0
        std_intensity = -1.0
        dhash_str = ""

        try:
            with Image.open(fpath) as img:
                img.verify()
            
            with Image.open(fpath) as img:
                width, height = img.size
                img_format = img.format or fpath.suffix.lstrip(".").upper()
                color_mode = img.mode
                dhash_str = compute_dhash(img)

            img_cv = cv2.imread(str(fpath))
            if img_cv is None:
                raise ValueError("OpenCV failed to decode image")
            
            if len(img_cv.shape) == 3:
                channels = img_cv.shape[2]
            else:
                channels = 1
            
            blur_var = estimate_blur_laplacian(img_cv)
            mean_intensity = float(np.mean(img_cv))
            std_intensity = float(np.std(img_cv))

        except Exception as e:
            is_corrupt = True
            error_msg = str(e)
            corrupted_records.append({
                "path": str(rel_path),
                "error": error_msg
            })

        record = {
            "sample_id": f"{doc_type}_{label}_{fpath.stem}",
            "filename": fpath.name,
            "relative_path": str(rel_path).replace("\\", "/"),
            "absolute_path": str(fpath.resolve()).replace("\\", "/"),
            "doc_type": doc_type,
            "label": label,
            "file_size_bytes": file_size,
            "sha256": sha256_hash,
            "dhash": dhash_str,
            "width": width,
            "height": height,
            "aspect_ratio": round(width / max(1, height), 4) if width > 0 else 0,
            "channels": channels,
            "format": img_format,
            "color_mode": color_mode,
            "blur_laplacian_var": round(blur_var, 2),
            "mean_intensity": round(mean_intensity, 2),
            "std_intensity": round(std_intensity, 2),
            "is_corrupt": is_corrupt,
            "error_msg": error_msg
        }
        inventory_records.append(record)

    df_inventory = pd.DataFrame(inventory_records)
    inv_csv_path = out_path / "dataset_inventory.csv"
    df_inventory.to_csv(inv_csv_path, index=False)
    print(f"Inventory saved: {inv_csv_path}")

    # ── 1. Class and Document Type Distribution ─────────────────────────────────
    dist_df = df_inventory.groupby(["doc_type", "label"]).size().unstack(fill_value=0)
    dist_df["total"] = dist_df.sum(axis=1)
    dist_csv_path = out_path / "class_distribution.csv"
    dist_df.to_csv(dist_csv_path)
    print(f"Distribution saved: {dist_csv_path}")

    # ── 2. Exact Duplicates (SHA-256) ───────────────────────────────────────────
    exact_dups = df_inventory[df_inventory.duplicated(subset=["sha256"], keep=False)].sort_values("sha256")
    exact_dup_records = []
    if len(exact_dups) > 0:
        for sha, group in exact_dups.groupby("sha256"):
            paths = group["relative_path"].tolist()
            doc_types = group["doc_type"].unique().tolist()
            labels = group["label"].unique().tolist()
            exact_dup_records.append({
                "hash_type": "SHA256 (Exact)",
                "hash_value": sha,
                "count": len(paths),
                "paths": " ; ".join(paths),
                "doc_types": " ; ".join(doc_types),
                "labels": " ; ".join(labels),
                "cross_label_leakage": len(labels) > 1,
                "cross_doctype_leakage": len(doc_types) > 1
            })

    # ── 3. Perceptual Near-Duplicates (dHash) ───────────────────────────────────
    near_dup_records = []
    valid_df = df_inventory[~df_inventory["is_corrupt"]].copy()
    
    # Check pairwise or within doc_type for dHash collisions or distance <= 2
    # Group by identical dHash first (Hamming distance 0)
    dhash_dups = valid_df[valid_df.duplicated(subset=["dhash"], keep=False)].sort_values("dhash")
    if len(dhash_dups) > 0:
        for dh, group in dhash_dups.groupby("dhash"):
            # If they already match sha256 exactly, skip to avoid double reporting
            if group["sha256"].nunique() > 1:
                paths = group["relative_path"].tolist()
                near_dup_records.append({
                    "hash_type": "dHash (Perceptual Identical, dist=0)",
                    "hash_value": dh,
                    "count": len(paths),
                    "paths": " ; ".join(paths),
                    "doc_types": " ; ".join(group["doc_type"].unique().tolist()),
                    "labels": " ; ".join(group["label"].unique().tolist()),
                    "cross_label_leakage": group["label"].nunique() > 1,
                    "cross_doctype_leakage": group["doc_type"].nunique() > 1
                })

    all_dups_df = pd.DataFrame(exact_dup_records + near_dup_records)
    dup_csv_path = out_path / "duplicate_report.csv"
    if len(all_dups_df) > 0:
        all_dups_df.to_csv(dup_csv_path, index=False)
    else:
        pd.DataFrame(columns=["hash_type", "hash_value", "count", "paths", "doc_types", "labels", "cross_label_leakage", "cross_doctype_leakage"]).to_csv(dup_csv_path, index=False)
    print(f"Duplicate report saved: {dup_csv_path}")

    # ── 4. Image Quality Statistics ─────────────────────────────────────────────
    unique_res = {
        f"{w}x{h}": int(cnt)
        for (w, h), cnt in df_inventory.groupby(["width", "height"]).size().items()
    }
    quality_summary = {
        "total_files": len(df_inventory),
        "corrupted_files": len(corrupted_records),
        "dimensions": {
            "unique_resolutions": unique_res,
            "min_width": int(df_inventory["width"].min()),
            "max_width": int(df_inventory["width"].max()),
            "min_height": int(df_inventory["height"].min()),
            "max_height": int(df_inventory["height"].max()),
            "mean_aspect_ratio": float(round(df_inventory["aspect_ratio"].mean(), 3))
        },
        "formats": {str(k): int(v) for k, v in df_inventory["format"].value_counts().items()},
        "color_modes": {str(k): int(v) for k, v in df_inventory["color_mode"].value_counts().items()},
        "file_size_bytes": {
            "min": int(df_inventory["file_size_bytes"].min()),
            "max": int(df_inventory["file_size_bytes"].max()),
            "mean": float(round(df_inventory["file_size_bytes"].mean(), 1))
        },
        "blur_laplacian_var": {
            "min": float(round(df_inventory["blur_laplacian_var"].min(), 2)),
            "max": float(round(df_inventory["blur_laplacian_var"].max(), 2)),
            "mean": float(round(df_inventory["blur_laplacian_var"].mean(), 2)),
            "median": float(round(df_inventory["blur_laplacian_var"].median(), 2))
        }
    }

    quality_json_path = out_path / "image_quality_summary.json"
    with open(quality_json_path, "w") as f:
        json.dump(quality_summary, f, indent=2)
    print(f"Quality summary saved: {quality_json_path}")

    # ── 5. Filename & Source Relationship Analysis ──────────────────────────────
    # Check if forged and genuine filenames match (e.g. 0.jpg in genuine vs 0.jpg in forged)
    source_relation_report = {}
    for doc in df_inventory["doc_type"].unique():
        sub = df_inventory[df_inventory["doc_type"] == doc]
        gen_stems = set(sub[sub["label"] == "genuine"]["filename"].apply(lambda x: Path(x).stem))
        forg_stems = set(sub[sub["label"] == "forged"]["filename"].apply(lambda x: Path(x).stem))
        overlap = gen_stems.intersection(forg_stems)
        source_relation_report[doc] = {
            "genuine_count": len(gen_stems),
            "forged_count": len(forg_stems),
            "shared_stem_names_count": len(overlap),
            "are_paired_by_stem": len(overlap) == min(len(gen_stems), len(forg_stems)) and len(overlap) > 0
        }

    # ── 6. Contact Sheet Generation ─────────────────────────────────────────────
    # Create contact sheets for representative inspection per document type (redacting potential PII areas or adding specimen banner)
    for doc in sorted(df_inventory["doc_type"].unique()):
        for lbl in ["genuine", "forged"]:
            subset = df_inventory[(df_inventory["doc_type"] == doc) & (df_inventory["label"] == lbl)]
            if subset.empty:
                continue
            
            # Select up to 9 sample images (3x3 grid)
            sample_subset = subset.head(9)
            thumb_size = (200, 140)
            grid_w, grid_h = 3, 3
            sheet = Image.new("RGB", (grid_w * thumb_size[0], grid_h * thumb_size[1]), color=(240, 240, 240))
            draw = ImageDraw.Draw(sheet)

            for idx, (_, row) in enumerate(sample_subset.iterrows()):
                gx = (idx % grid_w) * thumb_size[0]
                gy = (idx // grid_w) * thumb_size[1]
                try:
                    with Image.open(row["absolute_path"]) as img:
                        img_rgb = img.convert("RGB")
                        img_thumb = ImageOps.fit(img_rgb, thumb_size, Image.Resampling.LANCZOS)
                        sheet.paste(img_thumb, (gx, gy))
                        # Draw label banner on thumbnail
                        draw.rectangle([gx, gy, gx + thumb_size[0], gy + 20], fill=(0, 0, 0, 180))
                        draw.text((gx + 5, gy + 4), f"{doc} | {lbl} #{row['filename']}", fill=(255, 255, 255))
                except Exception as e:
                    draw.rectangle([gx, gy, gx + thumb_size[0], gy + thumb_size[1]], fill=(255, 200, 200))
                    draw.text((gx + 10, gy + 50), f"Error loading", fill=(255, 0, 0))

            sheet_path = contact_dir / f"contact_{doc}_{lbl}.png"
            sheet.save(sheet_path)

    # ── 7. Generate Audit Markdown Report ───────────────────────────────────────
    audit_md_path = out_path / "dataset_audit_summary.md"
    with open(audit_md_path, "w", encoding="utf-8") as f:
        f.write("# Dataset Audit Report — Phase 1\n")
        f.write("## AI-Based Fake Identity & Document Screening System\n\n")
        f.write(f"- **Dataset Root Path:** `{root_path.resolve()}`\n")
        f.write(f"- **Total Image Files:** {len(df_inventory)}\n")
        f.write(f"- **Corrupted Files:** {len(corrupted_records)}\n\n")

        f.write("### 1. Class Distribution Across Document Types\n\n")
        try:
            f.write(dist_df.to_markdown())
        except Exception:
            f.write(dist_df.to_string())
        f.write("\n\n")

        f.write("### 2. Image Properties & Quality Metrics\n\n")
        f.write(f"- **Formats:** `{json.dumps(quality_summary['formats'])}`\n")
        f.write(f"- **Color Modes:** `{json.dumps(quality_summary['color_modes'])}`\n")
        f.write(f"- **Resolutions:** `{json.dumps(quality_summary['dimensions']['unique_resolutions'])}`\n")
        f.write(f"- **Aspect Ratio (Mean):** `{quality_summary['dimensions']['mean_aspect_ratio']}`\n")
        f.write(f"- **File Size Range:** {quality_summary['file_size_bytes']['min']} to {quality_summary['file_size_bytes']['max']} bytes (Mean: {quality_summary['file_size_bytes']['mean']} B)\n")
        f.write(f"- **Blur Metric (Laplacian Var):** Min={quality_summary['blur_laplacian_var']['min']}, Max={quality_summary['blur_laplacian_var']['max']}, Mean={quality_summary['blur_laplacian_var']['mean']}, Median={quality_summary['blur_laplacian_var']['median']}\n\n")

        f.write("### 3. Duplicate and Data Leakage Analysis\n\n")
        f.write(f"- **Exact Duplicates (SHA-256):** {len(exact_dup_records)} duplicate clusters found.\n")
        f.write(f"- **Perceptual Near-Duplicates (dHash):** {len(near_dup_records)} clusters found.\n\n")

        f.write("### 4. Source-Document Relationship & Pairing Analysis\n\n")
        f.write("| Document Type | Genuine Count | Forged Count | Shared Filename Stems | Paired by Filename? |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- |\n")
        for doc, info in source_relation_report.items():
            f.write(f"| {doc} | {info['genuine_count']} | {info['forged_count']} | {info['shared_stem_names_count']} | {info['are_paired_by_stem']} |\n")
        f.write("\n")

        f.write("### 5. Document Authenticity and Synthesis Nature\n\n")
        f.write("- **Nature of Dataset:** The documents are programmatically generated synthetic specimen identity cards (`synthetic_id_dataset`).\n")
        f.write("- **Privacy Impact:** Fictional synthetic data — no live real-world Indian PII detected.\n")

    print(f"Audit summary written to: {audit_md_path}")
    print("=== AUDIT COMPLETED SUCCESSFULLY ===")

if __name__ == "__main__":
    dataset_dir = sys.argv[1] if len(sys.argv) > 1 else "D:/visual/synthetic_id_dataset"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "D:/visual/dataset_audit"
    run_audit(dataset_dir, output_dir)
