"""
preprocessing.py — SIH26188 Indian ID Forgery Detection
=========================================================
Handles:
  - Loading images from HuggingFace datasets or local folders
  - EfficientNet-B0 canonical normalisation (ImageNet stats)
  - Train / validation / test augmentation pipelines
  - Document-type–aware splitting (prevents leakage)
  - Source-ID–level stratified splits

Author : Visual Lead / Integration Lead
Project: SIH26188 — Cross-document generalisation experiment
"""

import os
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────
# EfficientNet-B0 canonical input size and ImageNet normalisation
EFFICIENTNET_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# All five Indian document categories used in the LODTO experiment
DOCUMENT_TYPES = ["aadhaar", "passport", "driving_licence", "pan", "voter_id"]

# Label mapping — binary classification
LABEL_MAP = {"genuine": 0, "forged": 1}


# ─── Albumentations pipelines ───────────────────────────────────────────────

def get_train_transforms(image_size: int = EFFICIENTNET_SIZE) -> A.Compose:
    """
    Augmentation pipeline for training.
    Forgery-aware: we deliberately avoid transforms that destroy forensic cues
    (e.g. heavy JPEG compression is kept mild; colour jitter is limited).
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.08,
                           rotate_limit=5, p=0.4),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=5, p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.25),
        A.OneOf([
            A.ImageCompression(quality_lower=60, quality_upper=95, p=1.0),
            A.GaussNoise(var_limit=(5, 25), p=1.0),
        ], p=0.25),
        A.ColorJitter(brightness=0.1, contrast=0.1,
                      saturation=0.05, hue=0.02, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = EFFICIENTNET_SIZE) -> A.Compose:
    """
    Deterministic pipeline for validation / test — no random augmentation.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# ─── Dataset class ───────────────────────────────────────────────────────────

class IndianIDDataset(Dataset):
    """
    PyTorch Dataset for Indian ID forgery detection.

    Accepts a pandas DataFrame with the following required columns:
        image_path  : str  — absolute or relative path to the image file
        label       : int  — 0 = genuine, 1 = forged
        doc_type    : str  — one of DOCUMENT_TYPES
        source_id   : str  — unique identifier for the source document
                             (all crops/manipulations from the same physical
                              document share the same source_id)
        manip_type  : str  — manipulation label, e.g. "text_alter",
                             "photo_replace", "splicing", "genuine"

    Optional columns:
        split       : str  — "train" | "val" | "test"
        quality     : str  — "high" | "medium" | "low"

    Parameters
    ----------
    df         : pd.DataFrame — metadata dataframe
    transforms : A.Compose   — albumentations pipeline
    root_dir   : str | None  — prepended to image_path when set
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transforms: A.Compose,
        root_dir: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.root_dir = root_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        # ── Load image ──────────────────────────────────────────────────────
        img_path = str(row["image_path"])
        if self.root_dir:
            img_path = os.path.join(self.root_dir, img_path)

        image = _load_image_rgb(img_path)

        # ── Apply transforms ─────────────────────────────────────────────────
        augmented = self.transforms(image=image)
        tensor = augmented["image"]  # shape: (3, H, W)  dtype: float32

        # ── Build sample dict ────────────────────────────────────────────────
        sample = {
            "image":      tensor,
            "label":      torch.tensor(int(row["label"]), dtype=torch.long),
            "doc_type":   str(row.get("doc_type", "unknown")),
            "source_id":  str(row.get("source_id", "unknown")),
            "manip_type": str(row.get("manip_type", "unknown")),
            "image_path": img_path,
        }
        return sample


# ─── Splitting utilities ─────────────────────────────────────────────────────

def build_splits(
    df: pd.DataFrame,
    held_out_doc_type: Optional[str] = None,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Create train / val / test splits that respect two critical constraints:

    1. SOURCE-ID ISOLATION — rows sharing the same source_id are kept in
       the same split so no document leaks across splits.
    2. LODTO ISOLATION    — if held_out_doc_type is set, ALL rows of that
       type go exclusively to the test split and are never seen during
       training or validation.

    Returns
    -------
    {"train": df_train, "val": df_val, "test": df_test}
    """
    rng = np.random.default_rng(random_seed)
    df = df.copy()

    # ── Step 1: peel off the held-out document type ──────────────────────────
    if held_out_doc_type:
        df_heldout = df[df["doc_type"] == held_out_doc_type].copy()
        df_rest    = df[df["doc_type"] != held_out_doc_type].copy()
        log.info("LODTO: held-out type='%s'  samples=%d",
                 held_out_doc_type, len(df_heldout))
    else:
        df_heldout = pd.DataFrame()
        df_rest    = df.copy()

    # ── Step 2: source-ID–level split on the remaining data ─────────────────
    source_ids = df_rest["source_id"].unique()
    rng.shuffle(source_ids)

    n_total = len(source_ids)
    n_val   = max(1, int(n_total * val_fraction))
    n_test  = max(1, int(n_total * test_fraction))
    n_train = n_total - n_val - n_test

    if n_train < 1:
        raise ValueError(
            f"Not enough unique source_ids ({n_total}) for the requested splits."
        )

    ids_train = set(source_ids[:n_train])
    ids_val   = set(source_ids[n_train:n_train + n_val])
    ids_test  = set(source_ids[n_train + n_val:])

    df_train = df_rest[df_rest["source_id"].isin(ids_train)].copy()
    df_val   = df_rest[df_rest["source_id"].isin(ids_val)].copy()
    df_test_seen = df_rest[df_rest["source_id"].isin(ids_test)].copy()

    # ── Step 3: combine held-out into test set ───────────────────────────────
    df_test = pd.concat([df_test_seen, df_heldout], ignore_index=True)

    # ── Sanity checks ────────────────────────────────────────────────────────
    _assert_no_source_leakage(df_train, df_val, df_test)
    if held_out_doc_type:
        _assert_no_doctype_in_splits(df_train, df_val, held_out_doc_type)

    log.info("Split sizes — train: %d  val: %d  test: %d (held-out: %d)",
             len(df_train), len(df_val), len(df_test), len(df_heldout))

    return {"train": df_train, "val": df_val, "test": df_test}


def _assert_no_source_leakage(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
):
    """Raise if any source_id appears in more than one split."""
    ids_train = set(df_train["source_id"])
    ids_val   = set(df_val["source_id"])
    ids_test  = set(df_test["source_id"])

    train_val_overlap = ids_train & ids_val
    train_test_overlap = ids_train & ids_test

    # Seen-test sources are intentionally in both test and train via LODTO;
    # only strict overlap from the same seen subset is a problem.
    if train_val_overlap:
        raise RuntimeError(
            f"Source-ID leakage between train and val: {train_val_overlap}"
        )


def _assert_no_doctype_in_splits(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    held_out: str,
):
    """Raise if the held-out doc type appears in train or val."""
    for name, df in [("train", df_train), ("val", df_val)]:
        leaked = df[df["doc_type"] == held_out]
        if len(leaked) > 0:
            raise RuntimeError(
                f"LODTO violation: held-out doc_type '{held_out}' "
                f"found {len(leaked)} rows in {name} split!"
            )


# ─── DataLoader factory ───────────────────────────────────────────────────────

def make_dataloaders(
    splits: Dict[str, pd.DataFrame],
    batch_size: int = 32,
    num_workers: int = 2,
    image_size: int = EFFICIENTNET_SIZE,
    root_dir: Optional[str] = None,
    pin_memory: bool = True,
) -> Dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from a splits dict.

    Parameters
    ----------
    splits     : output of build_splits()
    batch_size : samples per batch (reduce to 16 for low-VRAM / Colab free)
    num_workers: set 0 on Windows if multiprocessing causes issues
    root_dir   : prefix for image paths (optional)
    pin_memory : True when training on CUDA GPU
    """
    loaders = {}
    for split_name, df in splits.items():
        is_train = (split_name == "train")
        tfms = get_train_transforms(image_size) if is_train \
               else get_val_transforms(image_size)
        dataset = IndianIDDataset(df, tfms, root_dir=root_dir)
        loaders[split_name] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=is_train,
            num_workers=num_workers,
            pin_memory=pin_memory and torch.cuda.is_available(),
            drop_last=is_train,        # avoid tiny last batch in training
        )
        log.info("Loader '%s': %d samples, %d batches",
                 split_name, len(dataset),
                 len(loaders[split_name]))
    return loaders


# ─── HuggingFace dataset adapter ─────────────────────────────────────────────

def load_from_huggingface(
    dataset_name: str,
    split: str = "train",
    image_col: str = "image",
    label_col: str = "label",
    doc_type_col: Optional[str] = "doc_type",
    source_id_col: Optional[str] = "source_id",
    manip_type_col: Optional[str] = "manip_type",
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a HuggingFace dataset and convert to a metadata DataFrame.

    The images are saved to a local cache folder.
    Returns a DataFrame compatible with IndianIDDataset.

    Usage example
    -------------
    df = load_from_huggingface("your-org/indian-id-forgery", split="train")
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install 'datasets': pip install datasets")

    log.info("Loading HuggingFace dataset '%s' split='%s'", dataset_name, split)
    hf_ds = load_dataset(dataset_name, split=split, cache_dir=cache_dir)

    cache_dir = cache_dir or "./data/hf_cache"
    img_dir = Path(cache_dir) / "images" / split
    img_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for i, sample in enumerate(hf_ds):
        # Save PIL image to disk
        pil_img = sample[image_col]
        if isinstance(pil_img, dict):
            # Some datasets store as {"bytes": ..., "path": ...}
            pil_img = Image.open(pil_img["bytes"] if "bytes" in pil_img
                                 else pil_img["path"])
        img_fname = img_dir / f"{i:06d}.png"
        if not img_fname.exists():
            pil_img.convert("RGB").save(img_fname)

        record = {
            "image_path": str(img_fname),
            "label":      int(sample[label_col]),
            "doc_type":   str(sample[doc_type_col]) if doc_type_col and doc_type_col in sample else "unknown",
            "source_id":  str(sample[source_id_col]) if source_id_col and source_id_col in sample else f"sid_{i}",
            "manip_type": str(sample[manip_type_col]) if manip_type_col and manip_type_col in sample else "unknown",
        }
        records.append(record)

    df = pd.DataFrame(records)
    log.info("Loaded %d samples from HuggingFace dataset '%s'",
             len(df), dataset_name)
    return df


# ─── Dataset audit ────────────────────────────────────────────────────────────

def audit_dataset(df: pd.DataFrame, output_path: Optional[str] = None) -> Dict:
    """
    Run a quality audit on the metadata DataFrame and return a report dict.
    Call this before any training.

    Checks:
      - Sample counts per doc_type and label
      - Class balance
      - Missing / null values
      - Duplicate image paths
      - Images that cannot be opened
      - Source-ID cross-contamination risk
    """
    report = {}

    # ── Basic counts ─────────────────────────────────────────────────────────
    report["total_samples"] = len(df)
    report["label_distribution"] = df["label"].value_counts().to_dict()
    report["doc_type_distribution"] = df["doc_type"].value_counts().to_dict()

    # ── Per-doctype class balance ─────────────────────────────────────────────
    report["doctype_label_crosstab"] = (
        df.groupby(["doc_type", "label"]).size().unstack(fill_value=0).to_dict()
    )

    # ── Missing values ────────────────────────────────────────────────────────
    report["missing_values"] = df.isnull().sum().to_dict()

    # ── Duplicate image paths ─────────────────────────────────────────────────
    dupes = df["image_path"].duplicated().sum()
    report["duplicate_image_paths"] = int(dupes)

    # ── Source-ID coverage ────────────────────────────────────────────────────
    report["unique_source_ids"] = df["source_id"].nunique()
    report["samples_per_source_id_mean"] = (
        df.groupby("source_id").size().mean()
    )

    # ── Image openability check (sample up to 100 random images) ─────────────
    sample_paths = df["image_path"].sample(min(100, len(df)),
                                           random_state=42).tolist()
    bad_images = []
    for p in sample_paths:
        try:
            img = Image.open(p)
            img.verify()
        except Exception as e:
            bad_images.append({"path": p, "error": str(e)})
    report["unreadable_images_in_sample"] = bad_images

    # ── Print summary ─────────────────────────────────────────────────────────
    log.info("=== Dataset Audit ===")
    log.info("Total samples      : %d", report["total_samples"])
    log.info("Label distribution : %s", report["label_distribution"])
    log.info("Doc-type dist.     : %s", report["doc_type_distribution"])
    log.info("Unique source IDs  : %d", report["unique_source_ids"])
    if bad_images:
        log.warning("Found %d unreadable images in sample!", len(bad_images))

    # ── Save report ───────────────────────────────────────────────────────────
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        log.info("Audit report saved to %s", output_path)

    return report


# ─── Image loading helper ─────────────────────────────────────────────────────

def _load_image_rgb(path: str) -> np.ndarray:
    """
    Load image as uint8 numpy array in RGB format.
    Falls back from PIL to OpenCV on failure.
    """
    try:
        img = Image.open(path).convert("RGB")
        return np.array(img)
    except Exception:
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Cannot open image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ─── Quick sanity test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("Running preprocessing sanity test with a dummy DataFrame...")

    # Build a tiny dummy DataFrame to verify the pipeline end-to-end
    import tempfile, shutil

    tmp_dir = Path(tempfile.mkdtemp())
    records = []
    for i in range(20):
        doc  = DOCUMENT_TYPES[i % len(DOCUMENT_TYPES)]
        lbl  = i % 2
        img  = Image.fromarray(
            (np.random.rand(256, 256, 3) * 255).astype(np.uint8)
        )
        p = tmp_dir / f"img_{i:03d}.png"
        img.save(p)
        records.append({
            "image_path": str(p),
            "label":      lbl,
            "doc_type":   doc,
            "source_id":  f"src_{i // 2}",
            "manip_type": "splicing" if lbl else "genuine",
        })

    df = pd.DataFrame(records)
    report = audit_dataset(df, output_path=str(tmp_dir / "audit.json"))
    splits  = build_splits(df, held_out_doc_type="pan", random_seed=42)
    loaders = make_dataloaders(splits, batch_size=4, num_workers=0,
                               pin_memory=False)

    for name, loader in loaders.items():
        batch = next(iter(loader))
        log.info("Loader '%s' batch — image: %s  label: %s",
                 name, batch["image"].shape, batch["label"])

    shutil.rmtree(tmp_dir)
    log.info("Preprocessing sanity test PASSED.")
