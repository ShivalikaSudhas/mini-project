"""
preprocessing.py — Identity Document Forgery Detection
Handles:
  - Canonical normalisation for EfficientNet-B0 (ImageNet statistics)
  - Train / validation / test augmentation pipelines (torchvision & Albumentations)
  - PyTorch Dataset and DataLoader factories with source-ID level isolation
  - Document-type-aware Leave-One-Document-Type-Out (LODO) data loading
"""

import os
import json
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

# Conditional albumentations import
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

# ─── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────────────
EFFICIENTNET_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DOCUMENT_TYPES = ["aadhaar", "passport", "driving_licence", "pan", "voter_id"]
LABEL_MAP = {"genuine": 0, "forged": 1}


# ─── Torchvision Standard Pipelines ─────────────────────────────────────────

def get_torch_train_transforms(image_size: int = EFFICIENTNET_SIZE) -> T.Compose:
    """
    Torchvision augmentation pipeline for training.
    Carefully tuned to preserve forgery cues (slight rotation, mild jitter, horizontal flip).
    """
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.3),
        T.RandomRotation(degrees=5),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_torch_val_transforms(image_size: int = EFFICIENTNET_SIZE) -> T.Compose:
    """Deterministic pipeline for validation and testing."""
    return T.Compose([
        T.Resize((image_size, image_size), interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── Optional Albumentations Pipelines ──────────────────────────────────────

def get_albumentations_train_transforms(image_size: int = EFFICIENTNET_SIZE):
    if not HAS_ALBUMENTATIONS:
        raise ImportError("albumentations not installed. Use get_torch_train_transforms() instead.")
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.04, scale_limit=0.08, rotate_limit=5, p=0.4),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_train_transforms(image_size: int = EFFICIENTNET_SIZE):
    """Default training transform getter."""
    return get_torch_train_transforms(image_size)


def get_val_transforms(image_size: int = EFFICIENTNET_SIZE):
    """Default evaluation transform getter."""
    return get_torch_val_transforms(image_size)


# ─── PyTorch Dataset ────────────────────────────────────────────────────────

class IndianIDDataset(Dataset):
    """
    PyTorch Dataset for Indian ID forgery detection.
    Accepts DataFrame with columns: image_path (or absolute_path), label, doc_type, source_id.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transforms: Optional[Union[T.Compose, object]] = None,
        root_dir: Optional[str] = None,
    ):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms or get_val_transforms()
        self.root_dir = root_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict:
        row = self.df.iloc[idx]

        # Resolve path
        if "absolute_path" in row and os.path.exists(str(row["absolute_path"])):
            img_path = str(row["absolute_path"])
        elif "image_path" in row and os.path.exists(str(row["image_path"])):
            img_path = str(row["image_path"])
        elif "relative_path" in row:
            if self.root_dir:
                img_path = os.path.join(self.root_dir, str(row["relative_path"]))
            else:
                img_path = str(row["relative_path"])
        else:
            raise FileNotFoundError(f"Could not locate image path in row: {row.to_dict()}")

        # Load image via PIL
        try:
            image_pil = Image.open(img_path).convert("RGB")
        except Exception:
            # Fallback to OpenCV
            cv_img = cv2.imread(img_path)
            if cv_img is None:
                raise FileNotFoundError(f"Unable to read image: {img_path}")
            image_pil = Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

        # Apply transforms
        if HAS_ALBUMENTATIONS and isinstance(self.transforms, A.Compose):
            np_img = np.array(image_pil)
            augmented = self.transforms(image=np_img)
            tensor = augmented["image"]
        else:
            tensor = self.transforms(image_pil)

        sample = {
            "sample_id":  str(row.get("sample_id", f"sample_{idx:04d}")),
            "image":      tensor,
            "label":      torch.tensor(int(row["label"]), dtype=torch.long),
            "doc_type":   str(row.get("doc_type", "unknown")),
            "source_id":  str(row.get("source_id", "unknown")),
            "manip_type": str(row.get("manip_type", "unknown")),
            "image_path": img_path,
        }
        return sample


# ─── DataLoader Utilities ───────────────────────────────────────────────────

def make_dataloaders(
    splits: Dict[str, pd.DataFrame],
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = EFFICIENTNET_SIZE,
    root_dir: Optional[str] = None,
    pin_memory: bool = False,
) -> Dict[str, DataLoader]:
    """
    Constructs train, val, and test DataLoaders with appropriate transforms.
    """
    train_tf = get_train_transforms(image_size)
    val_tf   = get_val_transforms(image_size)

    loaders = {}
    if "train" in splits and len(splits["train"]) > 0:
        ds_train = IndianIDDataset(splits["train"], transforms=train_tf, root_dir=root_dir)
        loaders["train"] = DataLoader(
            ds_train, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        )

    if "val" in splits and len(splits["val"]) > 0:
        ds_val = IndianIDDataset(splits["val"], transforms=val_tf, root_dir=root_dir)
        loaders["val"] = DataLoader(
            ds_val, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )

    if "test" in splits and len(splits["test"]) > 0:
        ds_test = IndianIDDataset(splits["test"], transforms=val_tf, root_dir=root_dir)
        loaders["test"] = DataLoader(
            ds_test, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )

    return loaders


def load_split_dataloaders(
    split_csv_path: str,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = EFFICIENTNET_SIZE,
    root_dir: Optional[str] = None,
) -> Dict[str, DataLoader]:
    """
    Convenience helper to load splits directly from a metadata CSV (e.g. split_in_domain.csv or lodo_fold_*.csv).
    """
    df = pd.read_csv(split_csv_path)
    splits = {
        "train": df[df["split"] == "train"].copy(),
        "val":   df[df["split"] == "val"].copy(),
        "test":  df[df["split"] == "test"].copy()
    }
    log.info("Loaded splits from %s: train=%d, val=%d, test=%d",
             split_csv_path, len(splits["train"]), len(splits["val"]), len(splits["test"]))
    return make_dataloaders(splits, batch_size=batch_size, num_workers=num_workers,
                            image_size=image_size, root_dir=root_dir)


if __name__ == "__main__":
    log.info("Testing preprocessing pipeline with split_in_domain.csv...")
    csv_path = "D:/visual/data/metadata/split_in_domain.csv"
    if os.path.exists(csv_path):
        loaders = load_split_dataloaders(csv_path, batch_size=8, num_workers=0)
        for name, ldr in loaders.items():
            batch = next(iter(ldr))
            log.info("Loader '%s': batch tensor shape = %s, labels = %s",
                     name, batch["image"].shape, batch["label"].tolist())
        log.info("Preprocessing verification PASSED.")
    else:
        log.warning("Metadata CSV not found at: %s", csv_path)
