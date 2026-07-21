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
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.pytorch
from mlflow import MlflowClient
import copy

client = MlflowClient()


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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> dict:
    """
    Train the model for one epoch.

    Parameters
    ----------
    model : nn.Module
        Model to train.

    loader : DataLoader
        Training data loader.

    criterion : nn.Module
        Loss function.

    optimizer : torch.optim.Optimizer
        Optimizer.

    Returns
    -------
    dict
        Dictionary containing training loss, accuracy and F1 score.
    """

    model.train()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    for images, labels in loader:

        images = images.to(config.DEVICE)
        labels = labels.to(config.DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_predictions.extend(predictions.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    epoch_f1 = f1_score(
        all_labels,
        all_predictions,
        pos_label=config.POSITIVE_IDX,
    )

    return {
    "train_loss": epoch_loss,
    "train_accuracy": epoch_accuracy,
    "train_f1": epoch_f1,
    }


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
) -> dict:
    """
    Evaluate the model for one validation epoch.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.

    loader : DataLoader
        Validation data loader.

    criterion : nn.Module
        Loss function.

    Returns
    -------
    dict
        Dictionary containing validation loss, accuracy and F1 score.
    """

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    epoch_accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    epoch_f1 = f1_score(
        all_labels,
        all_predictions,
        pos_label=config.POSITIVE_IDX,
    )

    return {
        "val_loss": epoch_loss,
        "val_accuracy": epoch_accuracy,
        "val_f1": epoch_f1,
    }

def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
):
    """
    Train the model with validation and early stopping.

    Returns
    -------
    history : dict
        Training history.

    model_meta : dict
        Information about the best model.
    """

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "train_f1": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_f1": [],
    }

    best_val_f1 = float("-inf")
    best_epoch = 0
    patience = 0
    best_state = None

    for epoch in range(config.EPOCHS):

        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
        )

        val_metrics = validate_one_epoch(
            model,
            val_loader,
            criterion,
        )

        history["train_loss"].append(train_metrics["train_loss"])
        history["train_accuracy"].append(train_metrics["train_accuracy"])
        history["train_f1"].append(train_metrics["train_f1"])

        history["val_loss"].append(val_metrics["val_loss"])
        history["val_accuracy"].append(val_metrics["val_accuracy"])
        history["val_f1"].append(val_metrics["val_f1"])

        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | "
            f"Train F1: {train_metrics['train_f1']:.4f} | "
            f"Val F1: {val_metrics['val_f1']:.4f}"
        )

        mlflow.log_metrics(
            {
                 "train_loss": train_metrics["train_loss"],
                 "train_accuracy": train_metrics["train_accuracy"],
                 "train_f1": train_metrics["train_f1"],
                 "val_loss": val_metrics["val_loss"],
                 "val_accuracy": val_metrics["val_accuracy"],
                 "val_f1": val_metrics["val_f1"],
           },
           step=epoch + 1,
       )

        if val_metrics["val_f1"] > best_val_f1:

            best_val_f1 = val_metrics["val_f1"]
            best_epoch = epoch + 1

            best_state = copy.deepcopy(model.state_dict())

            save_model(model)

            patience = 0

        else:

            patience += 1

        if patience >= config.EARLY_STOP_PATIENCE:

            print("Early stopping triggered.")

            break

        if best_state is not None:
            model.load_state_dict(best_state)

    model_meta = {
    "backbone": config.BACKBONE,
    "freeze_backbone": config.FREEZE_BACKBONE,
    "best_epoch": best_epoch,
    "epochs_trained": len(history["train_loss"]),
    "best_val_f1": best_val_f1,
    "learning_rate": config.LEARNING_RATE,
    "batch_size": config.BATCH_SIZE,
    "random_seed": config.RANDOM_SEED,
    }

    return history, model_meta

def main() -> int:
    
    set_seed()

    # -------------------------------------------------
    # MLflow setup
    # -------------------------------------------------
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

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
    
    with mlflow.start_run(
        run_name=f"{config.BACKBONE}_freeze_{config.FREEZE_BACKBONE}"
    ):

        # Log experiment parameters

        mlflow.log_params(
           {
            "backbone": config.BACKBONE,
            "freeze_backbone": config.FREEZE_BACKBONE,
            "batch_size": config.BATCH_SIZE,
            "learning_rate": config.LEARNING_RATE,
            "epochs": config.EPOCHS,
            "weight_decay": config.WEIGHT_DECAY,
            "random_seed": config.RANDOM_SEED,
            "optimizer": "Adam",
            "loss_function": "CrossEntropyLoss",
            "device": config.DEVICE,
            "num_classes": config.NUM_CLASSES,
           }
        )

    # TODO 3 (MLflow): set_experiment; start_run; log_params; per-epoch log_metrics; early stop.
    # TODO 3 (eval + registry): test metrics; plot_eval; save_model; log_model + register +
    #         set @production alias; write model_meta.json + metrics.json.
    # TODO 4: save_reference_baseline on a clean val sample.

    # -------------------------------------------------
    # Training
    # -------------------------------------------------
        history, model_meta = fit(
             model,
             train_loader,
             val_loader,
             criterion,
             optimizer,
        )

        # -------------------------------------------------
        # Evaluate on test set
        # -------------------------------------------------
        y_true, y_pred, y_prob = evaluate.predict(
            model,
            test_loader,
        )

        evaluation_metrics = evaluate.compute_metrics(
            y_true,
            y_pred,
            y_prob,
        )

        evaluation_plot = evaluate.plot_eval(
            y_true,
            y_pred,
            y_prob,
        )

        test_items = data_prep.load_split(
            "v1",
            "test",
            root,
        )

        failures = evaluate.failure_cases(
            test_items,
            y_true,
            y_pred,
            y_prob,
        )
    # -------------------------------------------------
    # Save evaluation artifacts
    # -------------------------------------------------
        config.EVALUATION_PATH.write_text(
            json.dumps(
                evaluation_metrics,
                indent=2,
            )
        )

        config.FAILURE_CASES_PATH.write_text(
            json.dumps(
                failures,
                indent=2,
            )
        )

    # -------------------------------------------------
    # Log evaluation metrics
    # -------------------------------------------------
        mlflow.log_metrics(
            {
               key: value
               for key, value in evaluation_metrics.items()
               if isinstance(value, (int, float))
           }
        )
    
    # -----------------------------
    # Log trained model
    # -----------------------------
        logged_model = mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
        )

    # -------------------------------------------------
    # Register model
    # -------------------------------------------------
        registered_model = mlflow.register_model(
            model_uri=logged_model.model_uri,
            name=config.REGISTERED_MODEL,
        )

        mlflow.log_metric(
            "best_val_f1",
            model_meta["best_val_f1"],
        )

        client.set_registered_model_alias(
            name=config.REGISTERED_MODEL,
            alias=config.PRODUCTION_ALIAS,
            version=registered_model.version,
        )
    # -------------------------------------------------
    # Add metadata
    # -------------------------------------------------
        model_meta.update(
              {
                 "backbone": config.BACKBONE,
                 "freeze_backbone": config.FREEZE_BACKBONE,
                 "batch_size": config.BATCH_SIZE,
                 "learning_rate": config.LEARNING_RATE,
                 "random_seed": config.RANDOM_SEED,
              }
       )

     # -------------------------------------------------
    # Save JSON artifacts
    # -------------------------------------------------
        config.MODEL_META_PATH.write_text(
            json.dumps(model_meta, indent=2)
        )

        config.METRICS_PATH.write_text(
            json.dumps(history, indent=2)
        )

    # -----------------------------
    # Log artifacts to MLflow
    # -----------------------------
        mlflow.log_artifact(str(config.MODEL_META_PATH))
        mlflow.log_artifact(str(config.METRICS_PATH))
        mlflow.log_artifact(
            str(config.ARTIFACT_DIR / "data_quality_report.json")
        )
        mlflow.log_artifact(
            str(config.EVALUATION_PATH)
        )

        mlflow.log_artifact(
            str(evaluation_plot)
        )

        mlflow.log_artifact(
            str(config.FAILURE_CASES_PATH)
        )

    print("\nTraining completed successfully.")

    print(f"Best Validation F1 : {model_meta['best_val_f1']:.4f}")
    print(f"Best Epoch         : {model_meta['best_epoch']}")
    print(f"Registered Model Version : {registered_model.version}")
    print(f"Production Alias         : {config.PRODUCTION_ALIAS}")

    return 0





if __name__ == "__main__":
    raise SystemExit(main())
