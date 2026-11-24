"""
create_tampered_split.py
Creates a balanced, leak-free Train and Test split from the new Indian_ID_Tamper_Dataset.
- Grouped by `base_id` so genuine and tampered variants of the same card never leak across splits.
- Clean directory layout:
    dataset_tampered_split/
      train/
        genuine/
        forged/
      test/
        genuine/
        forged/
- Exports a detailed `split_manifest.csv` with metadata (tamper_style, capture_type, difficulty).
"""

import os
import shutil
import pathlib
import pandas as pd
import numpy as np

def create_tampered_split(
    dataset_root: str = "Indian_ID_Tamper_Dataset",
    output_root: str = "dataset_tampered_split",
    num_train_bases_per_doc: int = 12,   # 12 cards * 5 docs = 60 base cards
    num_test_bases_per_doc: int = 4,     # 4 cards * 5 docs = 20 base cards
    seed: int = 42
):
    np.random.seed(seed)
    dataset_path = pathlib.Path(dataset_root).resolve()
    output_path = pathlib.Path(output_root).resolve()
    labels_csv = dataset_path / "labels.csv"

    if not labels_csv.is_file():
        raise FileNotFoundError(f"labels.csv not found at {labels_csv}")

    df = pd.read_csv(labels_csv)
    print(f"[INFO] Loaded labels.csv: {len(df)} total images across {df['doc_type'].nunique()} document types.")

    # Unique base_ids per doc_type
    doc_types = sorted(df['doc_type'].unique())
    selected_train_bases = []
    selected_test_bases = []

    for dt in doc_types:
        dt_df = df[df['doc_type'] == dt]
        unique_bases = sorted(dt_df['base_id'].unique())
        np.random.shuffle(unique_bases)
        
        train_bases = unique_bases[:num_train_bases_per_doc]
        test_bases = unique_bases[num_train_bases_per_doc:num_train_bases_per_doc + num_test_bases_per_doc]
        
        selected_train_bases.extend(train_bases)
        selected_test_bases.extend(test_bases)
        print(f"  • {dt.title():<18}: {len(train_bases)} train base cards, {len(test_bases)} test base cards")

    # Filter rows
    df_train = df[df['base_id'].isin(selected_train_bases)].copy()
    df_test = df[df['base_id'].isin(selected_test_bases)].copy()

    df_train['split'] = 'train'
    df_test['split'] = 'test'

    combined = pd.concat([df_train, df_test], ignore_index=True)

    # Prepare output directories
    for split in ['train', 'test']:
        for label_dir in ['genuine', 'forged']:
            (output_path / split / label_dir).mkdir(parents=True, exist_ok=True)

    copied_records = []
    print(f"\n[INFO] Copying separated genuine and forged images to {output_root}...")

    for idx, row in combined.iterrows():
        src_img = dataset_path / row['image_path']
        if not src_img.is_file():
            print(f"[WARN] Image missing: {src_img}")
            continue

        label_folder = "forged" if row['label_forged'] == 1 else "genuine"
        dest_filename = f"{row['doc_type']}_{row['image_id']}.jpg"
        dest_path = output_path / row['split'] / label_folder / dest_filename

        shutil.copy2(src_img, dest_path)

        rec = row.to_dict()
        rec['split_file_path'] = str(dest_path.relative_to(output_path))
        copied_records.append(rec)

    manifest_df = pd.DataFrame(copied_records)
    manifest_csv = output_path / "split_manifest.csv"
    manifest_df.to_csv(manifest_csv, index=False)

    print("\n" + "=" * 65)
    print("  TAMPERED DATASET TRAIN/TEST SPLIT COMPLETED")
    print("=" * 65)
    print(f"Destination Folder: {output_path}")
    print(f"Train Set: {len(df_train)} images "
          f"({(df_train['label_forged']==0).sum()} genuine, {(df_train['label_forged']==1).sum()} forged)")
    print(f"Test Set : {len(df_test)} images "
          f"({(df_test['label_forged']==0).sum()} genuine, {(df_test['label_forged']==1).sum()} forged)")
    print(f"Manifest : {manifest_csv}")
    print("=" * 65)

    return manifest_df

if __name__ == "__main__":
    create_tampered_split()
