"""evaluate.py — Stage 3: evaluation metrics, plots, failure-case analysis. 

Positive class = DEFECT (recall on defects is the headline QC metric). Implement
prediction, imbalance-aware metrics, confusion/ROC plots, and a misclassified-sample list.
"""
from __future__ import annotations

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
)

from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


@torch.no_grad()
def predict(net, loader, return_embeddings: bool = False):
    """
    Run inference over a dataset.

    Parameters
    ----------
    net : nn.Module
        Trained model.

    loader : DataLoader
        Data loader.

    return_embeddings : bool
        Reserved for Stage 4 (embedding drift).

    Returns
    -------
    tuple
        (y_true, y_pred, y_prob_defect)
    """

    net.eval()

    y_true = []
    y_pred = []
    y_prob = []

    for images, labels in loader:

        images = images.to(config.DEVICE)

        outputs = net(images)

        probabilities = F.softmax(outputs, dim=1)

        predictions = probabilities.argmax(dim=1)

        y_true.extend(labels.numpy())

        y_pred.extend(predictions.cpu().numpy())

        y_prob.extend(
            probabilities[:, config.POSITIVE_IDX]
            .cpu()
            .numpy()
        )

    y_true = np.asarray(y_true)

    y_pred = np.asarray(y_pred)

    y_prob = np.asarray(y_prob)

    return y_true, y_pred, y_prob


def compute_metrics(
    y_true,
    y_pred,
    y_prob,
) -> dict:
    """
    Compute evaluation metrics for the defect detector.

    Parameters
    ----------
    y_true : ndarray
        Ground-truth labels.

    y_pred : ndarray
        Predicted labels.

    y_prob : ndarray
        Predicted probability of the defect class.

    Returns
    -------
    dict
        Evaluation metrics.
    """

    metrics = {

        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            pos_label=config.POSITIVE_IDX,
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            pos_label=config.POSITIVE_IDX,
        ),

        "f1": f1_score(
            y_true,
            y_pred,
            pos_label=config.POSITIVE_IDX,
        ),

        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
        ),

        "roc_auc": float(
            roc_auc_score(
                 y_true,
                 y_prob,
            )
        ),    

        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ).tolist(),
    }

    return metrics


def plot_eval(
    y_true,
    y_pred,
    y_prob,
    out: Path | None = None,
) -> Path:
    """
    Generate confusion matrix and ROC curve.

    Parameters
    ----------
    y_true : ndarray
    y_pred : ndarray
    y_prob : ndarray

    out : Path | None

    Returns
    -------
    Path
        Saved evaluation figure.
    """

    if out is None:
        out = config.ARTIFACT_DIR / "model_eval.png"

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    # ----------------------------------------
    # Confusion Matrix
    # ----------------------------------------
    cm = confusion_matrix(
        y_true,
        y_pred,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=config.CLASSES,
    )

    disp.plot(
        ax=axes[0],
        colorbar=False,
    )

    axes[0].set_title("Confusion Matrix")

    # ----------------------------------------
    # ROC Curve
    # ----------------------------------------
    fpr, tpr, _ = roc_curve(
        y_true,
        y_prob,
        pos_label=config.POSITIVE_IDX,
    )

    auc = roc_auc_score(
        y_true,
        y_prob,
    )

    axes[1].plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.3f}",
    )

    axes[1].plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    axes[1].set_xlabel("False Positive Rate")

    axes[1].set_ylabel("True Positive Rate")

    axes[1].set_title("ROC Curve")

    axes[1].legend()

    plt.tight_layout()

    plt.savefig(
        out,
        dpi=300,
    )

    plt.close()

    return out


def failure_cases(
    items,
    y_true,
    y_pred,
    y_prob,
    limit: int = 20,
) -> list[dict]:
    """
    Return misclassified samples for error analysis.

    Parameters
    ----------
    items : list
        Dataset items as (path, label).

    y_true : ndarray
        Ground-truth labels.

    y_pred : ndarray
        Predicted labels.

    y_prob : ndarray
        Defect probabilities.

    limit : int
        Maximum number of failures to return.

    Returns
    -------
    list
        Misclassified sample information.
    """

    failures = []

    for item, actual, predicted, prob in zip(
        items,
        y_true,
        y_pred,
        y_prob,
    ):

        if actual != predicted:

            path, _ = item

            failures.append(
                {
                    "image": str(path),
                    "actual": config.IDX_TO_CLASS[int(actual)],
                    "predicted": config.IDX_TO_CLASS[int(predicted)],
                    "p_defect": round(float(prob), 4),
                }
            )

        if len(failures) >= limit:
            break

    return failures
