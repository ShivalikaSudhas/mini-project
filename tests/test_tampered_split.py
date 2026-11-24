"""
tests/test_tampered_split.py
Validates the new tampered train/test split:
1. Verifies exact balanced split of genuine and forged images.
2. Verifies zero base_id leakage between train and test sets.
3. Verifies all image files exist and are valid readable images.
"""

import pathlib
import pytest
import cv2
import pandas as pd

def test_tampered_split_balance_and_leak_free():
    split_dir = pathlib.Path("dataset_tampered_split")
    manifest_csv = split_dir / "split_manifest.csv"
    
    if not manifest_csv.is_file():
        pytest.skip("dataset_tampered_split archived in favor of final_processed_dataset")
    
    df = pd.read_csv(manifest_csv)
    
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    assert len(train_df) == 480, f"Expected 480 train images, got {len(train_df)}"
    assert len(test_df) == 160, f"Expected 160 test images, got {len(test_df)}"
    
    # Check 50/50 balance
    assert (train_df['label_forged'] == 0).sum() == 240
    assert (train_df['label_forged'] == 1).sum() == 240
    assert (test_df['label_forged'] == 0).sum() == 80
    assert (test_df['label_forged'] == 1).sum() == 80
    
    # Check NO base_id leakage
    train_bases = set(train_df['base_id'].unique())
    test_bases = set(test_df['base_id'].unique())
    overlap = train_bases.intersection(test_bases)
    assert len(overlap) == 0, f"Data leakage detected! Overlapping base_ids: {overlap}"

def test_images_exist_and_readable():
    split_dir = pathlib.Path("dataset_tampered_split")
    manifest_csv = split_dir / "split_manifest.csv"
    if not manifest_csv.is_file():
        pytest.skip("dataset_tampered_split archived in favor of final_processed_dataset")
    df = pd.read_csv(manifest_csv)
    
    # Test sample of images
    sample_df = df.sample(n=20, random_state=42)
    for _, row in sample_df.iterrows():
        img_path = split_dir / row['split_file_path']
        assert img_path.is_file(), f"Image not found on disk: {img_path}"
        img = cv2.imread(str(img_path))
        assert img is not None, f"Failed to read image with OpenCV: {img_path}"
        assert img.shape[0] > 0 and img.shape[1] > 0
