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

    # Quantile bin edges from the reference distribution
    edges = np.quantile(
        reference,
        np.linspace(0, 1, bins + 1),
    )

    # Ensure strictly increasing edges
    edges = np.unique(edges)

    # Fallback if all values are identical
    if len(edges) < 2:
        return 0.0

    ref_hist, _ = np.histogram(
        reference,
        bins=edges,
    )

    cur_hist, _ = np.histogram(
        current,
        bins=edges,
    )

    ref_pct = ref_hist / len(reference)
    cur_pct = cur_hist / len(current)

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
    # TODO 4: simulate camera/lighting drift (brightness/blur/rotate/noise via config.DRIFT_SIM).
    raise NotImplementedError


def run() -> dict:
    # TODO 4: build reference (clean) + current (corrupted) batches → features, embeddings,
    #         mean confidence. Run Evidently DataDriftPreset + PSI; embedding PSI on
    #         distance-to-centroid; confidence drop. Write drift_summary.json + drift_report.html
    #         and set retrain_recommended from the configured thresholds.
    raise NotImplementedError


if __name__ == "__main__":
    run()
