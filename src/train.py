"""train.py — Stage 2/3: transfer-learning training + MLflow tracking + registry. 

Implement: build splits, train the ResNet18 head (CrossEntropy, Adam on trainable params,
early stop on val F1), log params/metrics/model to MLflow, register + promote to the
@production alias, evaluate on test, and save a clean-data reference baseline (image
features + embeddings) for drift monitoring.   Run: python -m src.train
"""
from __future__ import annotations

import json, random
from pathlib import Path
import numpy as np, torch, torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep, evaluate
from src.dataset import CastingDataset, make_loaders
from src.model import build_model, trainable_parameters, save_model, EmbeddingExtractor
from torch.utils.data import DataLoader
from collections import Counter, defaultdict


def set_seed(seed: int = config.RANDOM_SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _subsample(items: list, cap: int | None) -> list:
    """
    Return a stratified random subset of dataset items.

    Parameters
    ----------
    items : list
        List of [relative_path, class_label].

    cap : int | None
        Maximum number of samples to retain.

    Returns
    -------
    list
        Stratified subset preserving class distribution.
    """

    if cap is None or cap <= 0 or cap >= len(items):
        return items

    grouped = defaultdict(list)

    for item in items:
        grouped[item[1]].append(item)

    subset = []

    for samples in grouped.values():

        n = round(len(samples) / len(items) * cap)

        subset.extend(
            random.sample(
                samples,
                min(n, len(samples))
            )
        )

    random.shuffle(subset)

    return subset


def class_weights(items) -> torch.Tensor:
    """
    Compute inverse-frequency class weights for CrossEntropyLoss.

    Parameters
    ----------
    items : list
        List of [relative_path, class_label] records.

    Returns
    -------
    torch.Tensor
        Class weights ordered by class index.
    """

    labels = [item[1] for item in items]

    counts = Counter(labels)

    num_classes = len(counts)
    total_samples = len(labels)

    weights = []

    for cls in sorted(counts.keys()):
        weight = total_samples / (num_classes * counts[cls])
        weights.append(weight)

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=config.DEVICE,
    )


def save_reference_baseline(net, ref_items) -> dict:
    # TODO 4: save reference_features.csv + reference_embeddings.npz for clean ref images.
    raise NotImplementedError


def main() -> int:
    import mlflow, mlflow.pytorch
    from mlflow import MlflowClient
    set_seed()
    root = data_prep.find_data_root()
    qc = data_prep.validate_quality(root)
    (config.ARTIFACT_DIR / "data_quality_report.json").write_text(json.dumps(qc, indent=2))
    data_prep.build_splits(root, "v1")
    
    # TODO 2/3: load splits, subsample train, build loaders, build model + optimiser + loss.
    loaders = make_loaders("v1", root)

    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # -------------------------------------------------
    # Build model
    # -------------------------------------------------
    # Model
    model = build_model().to(config.DEVICE)

    # -------------------------------------------------
    # Loss function
    # -------------------------------------------------
    # Loss
    weights = class_weights(train_loader.dataset.items)

    criterion = nn.CrossEntropyLoss(weight=weights)

    # -------------------------------------------------
    # Optimizer
    # -------------------------------------------------
    # Optimizer
    optimizer = torch.optim.Adam(
    trainable_parameters(model),
    lr=config.LEARNING_RATE,
    weight_decay=config.WEIGHT_DECAY,
    )
    
    # TODO 3 (MLflow): set_experiment; start_run; log_params; per-epoch log_metrics; early stop.
    # TODO 3 (eval + registry): test metrics; plot_eval; save_model; log_model + register +
    #         set @production alias; write model_meta.json + metrics.json.
    # TODO 4: save_reference_baseline on a clean val sample.
    raise NotImplementedError("Implement the training + MLflow + registry workflow")


if __name__ == "__main__":
    raise SystemExit(main())
