"""data_prep.py — Stage 1: discovery, quality validation, versioned splits, transforms. 

Implement: locate the casting folders; run data-quality checks (missing/corrupt/duplicate/
dimension/class-distribution/consistency); build reproducible stratified train/val/test
splits with a versioned snapshot + metadata.json; define preprocessing + augmentation
transforms; and per-image feature extraction used by drift monitoring.
"""
from __future__ import annotations

import hashlib, json, os, random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageStat, UnidentifiedImageError

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ==========================================================
# Dataset Discovery
# ==========================================================

def find_data_root(base: Path | None = None) -> Path:
    """
    Locate the casting dataset root.

    Expected structure:
        data/
        └── casting_data/
            ├── train/
            │   ├── ok_front/
            │   └── def_front/
            └── test/
                ├── ok_front/
                └── def_front/

    Parameters
    ----------
    base : Path | None
        Optional base directory. If None, config.DATA_DIR is used.

    Returns
    -------
    Path
        Path to the dataset root (casting_data).

    Raises
    ------
    FileNotFoundError
        If the expected dataset structure is not found.
    """

    search_root = Path(base) if base else Path(config.DATA_DIR)

    dataset_root = (search_root / "casting_data").resolve()

    required_dirs = [
        dataset_root / "train" / "ok_front",
        dataset_root / "train" / "def_front",
        dataset_root / "test" / "ok_front",
        dataset_root / "test" / "def_front",
    ]

    if all(path.is_dir() for path in required_dirs):
        return dataset_root.resolve()

    raise FileNotFoundError(
        f"Casting dataset not found under: {search_root}"
    )


def list_images(split_dir: Path) -> list[tuple[Path, int]]:
    items = []
    for cls, idx in config.CLASS_TO_IDX.items():
        for p in sorted((split_dir / cls).glob("*")):
            if p.suffix.lower() in IMG_EXTS:
                items.append((p, idx))
    return items


# ==========================================================
# Utility Functions
# ==========================================================

def compute_md5(file_path: Path) -> str:
    """
    Compute the MD5 hash of an image file.

    Reading the file in chunks keeps memory usage low and
    works efficiently even for large files.

    Parameters
    ----------
    file_path : Path
        Path to the image file.

    Returns
    -------
    str
        MD5 hash represented as a hexadecimal string.
    """

    md5_hash = hashlib.md5()

    with open(file_path, "rb") as file:

        while True:

            chunk = file.read(8192)

            if not chunk:
                break

            md5_hash.update(chunk)

    return md5_hash.hexdigest()


# ==========================================================
# Dataset Validation
# ==========================================================

def validate_quality(root: Path) -> dict:
    """
    Perform dataset quality validation.

    Parameters
    ----------
    root : Path
        Dataset root containing train/ and test/ folders.

    Returns
    -------
    dict
        Dataset quality report.
    """

    report = {
    "passed": True,
    "warnings": [],

    "total_images": 0,

    "corrupt_images": [],
    "corrupt_count": 0,

    "invalid_dimensions": [],
    "invalid_dimension_count": 0,

    "duplicate_images": [],
    "duplicate_count": 0,

    "class_distribution": {}
}
    
    hash_registry = {}

    total_images = 0

    for split in ["train", "test"]:

        split_dir = root / split

        images = list_images(split_dir)

        report["class_distribution"][split] = {}

        class_counter = Counter()

        for _, label in images:
            class_counter[label] += 1
        
        for image_path, _ in images:

            try:

                # Verify image integrity
                  with Image.open(image_path) as img:
                     img.verify()

                # Re-open image to check dimensions
                  with Image.open(image_path) as img:

                       if img.size != (300, 300):

                          report["invalid_dimensions"].append(
                                 {
                                    "file": str(image_path),
                                    "size": img.size
                                 }
                          )
                # Compute image hash for duplicate detection
                  image_hash = compute_md5(image_path)

                  if image_hash in hash_registry:
                            
                            original_path = hash_registry[image_hash]

                            original_split = original_path.parts[-3]
                            duplicate_split = image_path.parts[-3]

                            report["duplicate_images"].append(
                               {
                                  "hash": image_hash,
                                  "original": str(original_path),
                                  "duplicate": str(image_path),
                                  "cross_split": original_split != duplicate_split

                               }
                            )

                  else:
                        hash_registry[image_hash] = image_path
                        

            except (UnidentifiedImageError, OSError):

                 report["corrupt_images"].append(str(image_path))

        for class_name, class_idx in config.CLASS_TO_IDX.items():
            report["class_distribution"][split][class_name] = class_counter.get(class_idx, 0)

        total_images += len(images)

    report["total_images"] = total_images

    report["corrupt_count"] = len(report["corrupt_images"])

    report["invalid_dimension_count"] = len(report["invalid_dimensions"])

    report["duplicate_count"] = len(report["duplicate_images"])

# Add warning if duplicates were detected
    if report["duplicate_count"] > 0:

        report["warnings"].append(
            f"{report['duplicate_count']} duplicate images detected. "
            "Review potential train-test data leakage."
    )

# Hard failures only
    report["passed"] = (
        report["corrupt_count"] == 0
        and report["invalid_dimension_count"] == 0
)

    return report


# ==========================================================
# Dataset Versioning
# ==========================================================

def build_splits(root: Path, version: str = "v1") -> dict:
    # TODO 1 (versioning): stratified val carve-out from train/; test from test/. Save
    #         {train,val,test}.json file lists + metadata.json (version, date, class defs,
    #         split sizes/distribution, seed) under config.SPLIT_DIR/<version>/.
    raise NotImplementedError("Build versioned stratified splits + metadata")


def load_split(version: str, name: str, root: Path) -> list[tuple[Path, int]]:
    rel = json.loads((config.SPLIT_DIR / version / f"{name}.json").read_text())
    return [(root / r, y) for r, y in rel]


# ==========================================================
# Transforms
# ==========================================================

def get_transforms(train: bool):
    from torchvision import transforms
    # TODO 2 (preprocessing + augmentation): Grayscale(3) → Resize(224) → [train: flip,
    #         affine rotation/translate, ColorJitter] → ToTensor → Normalize(ImageNet).
    raise NotImplementedError("Define the train/eval transforms")


# ==========================================================
# Feature Extraction
# ==========================================================

def image_features(img: Image.Image) -> dict:
    # TODO 4 (statistical drift): return brightness, contrast, edge_density, sharpness,
    #         mean_intensity for a PIL image (keys == config.DRIFT_FEATURES).
    raise NotImplementedError("Extract per-image drift features")
