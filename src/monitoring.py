"""monitoring.py — Stage 4: statistical + embedding drift + confidence monitoring.

Implement the three drift signals against the clean reference baseline:
  1. statistical drift  — Evidently DataDriftPreset + PSI on image features
  2. embedding drift    — PSI on ResNet-embedding distance-to-centroid distribution
  3. confidence         — mean predicted confidence reference vs current
Use a corrupted copy of clean images as the simulated "current" production batch.
Outputs drift_report.html + drift_summary.json.   Run: python -m src.monitoring

Embedding drift (TODO 4) — see the conceptual walkthrough in
Operations_Monitoring_and_Evidence.ipynb (Stage 4.3):
  1. feature extraction  — penultimate 512-dim ResNet embedding (model.EmbeddingExtractor)
  2. embedding generation — embeddings for reference + current batches
  3. feature-space compare — reduce each to distance-to-reference-centroid (one distribution per batch)
  4. drift calculation   — PSI between the two distance distributions (> ~0.10 => drifted)
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image, ImageEnhance, ImageFilter

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src import data_prep
from src.model import load_model, EmbeddingExtractor
import torch
import torch.nn.functional as F

# -------------------------------------------------
# Optional Evidently support
# -------------------------------------------------
try:
    from evidently import Report
    from evidently.presets import DataDriftPreset

    EVIDENTLY_AVAILABLE = True

except Exception:

    Report = None
    DataDriftPreset = None

    EVIDENTLY_AVAILABLE = False


def psi(reference, current, bins: int = 10) -> float:
    """
    Compute Population Stability Index (PSI) between two 1-D distributions.

    Parameters
    ----------
    reference : array-like
        Reference distribution.

    current : array-like
        Current distribution.

    bins : int
        Number of quantile bins.

    Returns
    -------
    float
        PSI value.
    """

    reference = np.asarray(reference).ravel()
    current = np.asarray(current).ravel()

    # Use evenly spaced bins over the combined range
    minimum = min(reference.min(), current.min())
    maximum = max(reference.max(), current.max())

    if minimum == maximum:
        return 0.0

    edges = np.linspace(
        minimum,
        maximum,
        bins + 1,
    )

    ref_hist, _ = np.histogram(
        reference,
        bins=edges,
    )

    cur_hist, _ = np.histogram(
        current,
        bins=edges,
    )

    ref_pct = ref_hist / ref_hist.sum()
    cur_pct = cur_hist / cur_hist.sum()

    eps = 1e-6

    ref_pct = np.clip(ref_pct, eps, None)
    cur_pct = np.clip(cur_pct, eps, None)

    return float(
        np.sum(
            (cur_pct - ref_pct)
            * np.log(cur_pct / ref_pct)
        )
    )


def corrupt(img: Image.Image) -> Image.Image:
    """
    Simulate production image drift.

    Operations:
    1. Brightness change
    2. Gaussian blur
    3. Rotation
    4. Gaussian noise
    """

    # -----------------------------
    # Brightness
    # -----------------------------
    img = ImageEnhance.Brightness(img).enhance(
        config.DRIFT_SIM["brightness"]
    )

    # -----------------------------
    # Blur
    # -----------------------------
    img = img.filter(
        ImageFilter.GaussianBlur(
            radius=config.DRIFT_SIM["blur_radius"]
        )
    )

    # -----------------------------
    # Rotate
    # -----------------------------
    img = img.rotate(
        config.DRIFT_SIM["rotate"]
    )

    # -----------------------------
    # Gaussian Noise
    # -----------------------------
    arr = np.array(img).astype(np.float32)

    noise = np.random.normal(
        loc=0,
        scale=config.DRIFT_SIM["noise_std"],
        size=arr.shape,
    )

    arr = np.clip(
        arr + noise,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(arr)


def save_simple_html(summary: dict, out_path: Path):
    """
    Generate a simple HTML drift report when Evidently
    is unavailable.
    """

    html = f"""
    <html>
    <head>
        <title>Drift Report</title>
    </head>
    <body>

    <h1>Casting Defect Monitoring Report</h1>

    <table border="1" cellpadding="8">

        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>

        {''.join(
            f"<tr><td>{k}</td><td>{v}</td></tr>"
            for k, v in summary.items()
        )}

    </table>

    </body>
    </html>
    """

    out_path.write_text(
        html,
        encoding="utf-8",
    )


def run() -> dict:
    """
    Run drift monitoring.
    """

    root = data_prep.find_data_root()

    model = load_model()

    model.eval()

    extractor = EmbeddingExtractor(model)

    tf = data_prep.get_transforms(train=False)

    reference_dir = root / "test" / "ok_front"

    reference_paths = sorted(
        reference_dir.glob("*.jpeg")
    )

    reference_features = []
    current_features = []

    reference_embeddings = []
    current_embeddings = []

    reference_confidence = []
    current_confidence = []

    for path in reference_paths:

        img = Image.open(path).convert("L")

        corrupted = corrupt(img)

        reference_features.append(
            data_prep.image_features(img)
        )

        current_features.append(
            data_prep.image_features(corrupted)
        )

        ref_tensor = tf(img).unsqueeze(0)

        cur_tensor = tf(corrupted).unsqueeze(0)

        with torch.no_grad():

            ref_logits = model(ref_tensor)

            cur_logits = model(cur_tensor)

            ref_prob = F.softmax(
                ref_logits,
                dim=1,
            )

            cur_prob = F.softmax(
                cur_logits,
                dim=1,
            )

            reference_confidence.append(
                float(ref_prob.max())
            )

            current_confidence.append(
                float(cur_prob.max())
            )

            reference_embeddings.append(
                extractor(ref_tensor).cpu().numpy().squeeze()
            )

            current_embeddings.append(
                extractor(cur_tensor).cpu().numpy().squeeze()
            )

    reference_df = pd.DataFrame(
        reference_features
    )

    current_df = pd.DataFrame(
        current_features
    )

    statistical_psi = {}

    for col in reference_df.columns:

        statistical_psi[col] = psi(
            reference_df[col],
            current_df[col],
        )

    reference_embeddings = np.asarray(
        reference_embeddings
    )

    np.savez(
        config.REFERENCE_EMBED,
        embeddings=reference_embeddings,
    )

    current_embeddings = np.asarray(
        current_embeddings
    )

    centroid = reference_embeddings.mean(axis=0)

    ref_dist = np.linalg.norm(
        reference_embeddings - centroid,
        axis=1,
    )

    cur_dist = np.linalg.norm(
        current_embeddings - centroid,
        axis=1,
    )

    embedding_psi = psi(
        ref_dist,
        cur_dist,
    )

    ref_conf = float(
        np.mean(reference_confidence)
    )

    cur_conf = float(
        np.mean(current_confidence)
    )

    confidence_drop = ref_conf - cur_conf

    # -------------------------------------------------
    # Overall statistical PSI
    # -------------------------------------------------
    mean_statistical_psi = float(
        np.mean(
            list(statistical_psi.values())
        )
    )

    # -------------------------------------------------
    # Threshold checks
    # -------------------------------------------------
    confidence_alert = (
        confidence_drop >
        config.CONFIDENCE_DROP_THRESHOLD
    )

    statistical_drift = (
        mean_statistical_psi >
        config.PSI_THRESHOLD
    )

    embedding_drift = (
        embedding_psi >
        config.EMBEDDING_DRIFT_THRESHOLD
    )

    retrain_recommended = (
        confidence_alert
        or statistical_drift
        or embedding_drift
    )

    # -------------------------------------------------
    # Summary
    # -------------------------------------------------
    summary = {

        "reference_confidence": round(ref_conf, 4),

        "current_confidence": round(cur_conf, 4),

        "confidence_drop": round(confidence_drop, 4),

        "confidence_alert": confidence_alert,

        "mean_statistical_psi": round(
            mean_statistical_psi,
            4,
        ),

        "embedding_psi": round(
            float(embedding_psi),
            4,
        ),

        "retrain_recommended": retrain_recommended,

        "feature_psi": {
            k: round(v, 4)
            for k, v in statistical_psi.items()
        },
    }

    # -------------------------------------------------
    # Save JSON
    # -------------------------------------------------
    config.DRIFT_SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
        )
    )

    # -------------------------------------------------
    # HTML Report
    # -------------------------------------------------
    if EVIDENTLY_AVAILABLE:

        report = Report(
            metrics=[
                DataDriftPreset(),
            ]
        )

        report.run(
            reference_data=reference_df,
            current_data=current_df,
        )

        report.save_html(
            config.DRIFT_REPORT_PATH
        )

    else:

        print(
            "Evidently unavailable. "
            "Generating fallback report..."
        )

        save_simple_html(
            summary,
            config.DRIFT_REPORT_PATH,
        )

    print("\nMonitoring completed.")

    return summary


if __name__ == "__main__":
    run()
