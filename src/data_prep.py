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

from sklearn.model_selection import train_test_split
from datetime import datetime

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


def save_split(items, root: Path, output_file: Path):
    """
    Save a dataset split as a JSON file using paths
    relative to the dataset root.

    Parameters
    ----------
    items : list
        List of (Path, label) tuples.

    root : Path
        Dataset root directory.

    output_file : Path
        Destination JSON file.
    """

    records = []

    for path, label in items:

        relative_path = path.relative_to(root)

        records.append(
            [
                str(relative_path),
                label
            ]
        )

    with open(output_file, "w") as f:

        json.dump(records, f, indent=4)

    print(f"Saved: {output_file.name} ({len(records)} images)")


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
    # --------------------------------------------------
    # Create version directory
    # --------------------------------------------------

    version_dir = config.SPLIT_DIR / version

    version_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    train_items = list_images(root / "train")
    test_items = list_images(root / "test")

    # --------------------------------------------------
    # Separate paths and labels
    # --------------------------------------------------

    train_paths = [path for path, _ in train_items]
    train_labels = [label for _, label in train_items]

    # --------------------------------------------------
    # Stratified Train / Validation Split
    # --------------------------------------------------

    train_paths_split, val_paths, train_labels_split, val_labels = train_test_split(
       train_paths,
       train_labels,
       test_size=config.VAL_SPLIT,
       random_state=config.RANDOM_SEED,
       stratify=train_labels,
    )

    # --------------------------------------------------
    # Verify class distribution
    # --------------------------------------------------

    train_distribution = Counter(train_labels_split)
    val_distribution = Counter(val_labels)

    print("\nClass Distribution After Split")

    for class_name, class_idx in config.CLASS_TO_IDX.items():

        print(
            f"{class_name:<12}"
            f"Train : {train_distribution[class_idx]:2d}   "
            f"Validation : {val_distribution[class_idx]:2d}"
         )

    print("\nStratified split completed")

    print(f"Training Samples   : {len(train_paths_split)}")
    print(f"Validation Samples : {len(val_paths)}")

    # --------------------------------------------------
    # Prepare split records
    # --------------------------------------------------

    train_split = list(zip(train_paths_split, train_labels_split))

    val_split = list(zip(val_paths, val_labels))

    test_split = test_items

    print("\nPrepared Split Records")
    print(f"Train      : {len(train_split)}")
    print(f"Validation : {len(val_split)}")
    print(f"Test       : {len(test_split)}")

    save_split(
    train_split,
    root,
    version_dir / "train.json"
    )

    save_split(
    val_split,
    root,
    version_dir / "val.json"
    )

    save_split(
    test_split,
    root,
    version_dir / "test.json"
    )

    # --------------------------------------------------
    # Test split distribution
    # --------------------------------------------------

    test_distribution = Counter(label for _, label in test_items)

    # --------------------------------------------------
    # Build metadata
    # --------------------------------------------------

    metadata = {

       "dataset_version": version,

       "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

       "random_seed": config.RANDOM_SEED,

       "validation_split": config.VAL_SPLIT,

       "positive_class": config.POSITIVE_CLASS,

       "class_mapping": config.CLASS_TO_IDX,

       "split_info": {

            "train": {

                "samples": len(train_split),

                "distribution": {
                     class_name: train_distribution[class_idx]
                     for class_name, class_idx in config.CLASS_TO_IDX.items()
                }

            },

            "validation": {

                 "samples": len(val_split),

                 "distribution": {
                      class_name: val_distribution[class_idx]
                      for class_name, class_idx in config.CLASS_TO_IDX.items()
                }

            },

            "test": {

                 "samples": len(test_split),

                 "distribution": {
                      class_name: test_distribution[class_idx]
                      for class_name, class_idx in config.CLASS_TO_IDX.items()
                  }

            }

        }

    }

    metadata_path = version_dir / "metadata.json"

    with open(metadata_path, "w") as f:

        json.dump(metadata, f, indent=4)

    print(f"Saved: {metadata_path.name}")

    print(f"Dataset version directory created:\n{version_dir}")

    print(f"Training Images : {len(train_items)}")
    print(f"Testing Images  : {len(test_items)}")

    print(f"Training samples available for splitting: {len(train_paths)}")

    return metadata


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
