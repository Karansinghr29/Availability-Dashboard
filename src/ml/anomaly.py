"""
anomaly.py — unsupervised asset ANOMALY detection with Isolation Forest.

Why Isolation Forest?
---------------------
It isolates points via random splits; anomalies need fewer splits to isolate.
Good fit here because:
  * fully unsupervised (no failure labels needed),
  * handles the small, skewed maintenance feature set well,
  * gives a continuous score we can normalise to 0-100 and a binary flag.

Optional Local Outlier Factor (LOF)
-----------------------------------
LOF is computed as a CROSS-CHECK only (density-based, different assumptions).
We report how often IF and LOF agree so the reliability of the anomaly signal is
visible; Isolation Forest remains the primary output surfaced on the dashboard.

Output per asset:
  ml_anomaly_score  — 0-100, higher = more anomalous (normalised IF score)
  ml_anomaly_flag   — True when Isolation Forest labels the asset an outlier
  (module also returns IF/LOF agreement % for logging, not per-asset display)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def detect_anomalies(features: pd.DataFrame, X_scaled: np.ndarray, random_state: int = 42):
    """Return (df, agreement_pct). df has asset_id, ml_anomaly_score, ml_anomaly_flag."""
    n = len(features)
    if n == 0:
        return pd.DataFrame(columns=["asset_id", "ml_anomaly_score", "ml_anomaly_flag"]), None

    iso = IsolationForest(
        n_estimators=200,
        contamination="auto",     # let the model set the outlier threshold, don't force a rate
        random_state=random_state,
    )
    iso.fit(X_scaled)
    if_flag = iso.predict(X_scaled) == -1          # -1 = outlier
    # score_samples: higher = more normal. Invert + min-max to 0-100 (higher = more anomalous).
    raw = iso.score_samples(X_scaled)
    inv = -raw
    lo, hi = inv.min(), inv.max()
    score = np.zeros(n) if hi == lo else (inv - lo) / (hi - lo)
    anomaly_score = np.round(score * 100, 1)

    # LOF cross-check (only if enough neighbours); agreement % for logging
    agreement = None
    if n >= 20:
        k = min(20, n - 1)
        lof = LocalOutlierFactor(n_neighbors=k)          # novelty=False -> fit_predict on the data
        lof_flag = lof.fit_predict(X_scaled) == -1
        agreement = round(100.0 * float((if_flag == lof_flag).mean()), 1)

    df = pd.DataFrame({
        "asset_id": features["asset_id"].astype(str).values,
        "ml_anomaly_score": anomaly_score,
        "ml_anomaly_flag": if_flag,
    })
    return df, agreement
