"""
pipeline.py — single entry point that runs the additive unsupervised ML layer.

Flow (all label-free, all reproducible with random_state=42):
    assets_df + mapped
        -> features.build_feature_frame   (6 features from existing engine output)
        -> StandardScaler                 (GMM & IsolationForest are scale-sensitive)
        -> segmentation.fit_segments      (GMM -> cluster, cluster_prob, risk_segment)
        -> anomaly.detect_anomalies       (Isolation Forest -> anomaly_score, flag)
        -> merged per-asset ML frame

Returns a DataFrame keyed by asset_id with:
    ml_cluster, ml_cluster_prob, ml_risk_segment,
    ml_anomaly_score, ml_anomaly_flag
plus a small ``.attrs['meta']`` dict (n_assets, features, IF/LOF agreement).

Does NOT touch the rule engine, invent labels, or forecast asset failures.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import features as _features
from . import segmentation as _seg
from . import anomaly as _anom

_OUT_COLS = ["asset_id", "ml_cluster", "ml_cluster_prob", "ml_risk_segment",
             "ml_anomaly_score", "ml_anomaly_flag"]


def run_asset_ml(assets_df: pd.DataFrame, mapped: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Run GMM segmentation + Isolation Forest anomaly detection. Empty-safe."""
    f = _features.build_feature_frame(assets_df, mapped)
    empty = pd.DataFrame(columns=_OUT_COLS)
    if f.empty:
        empty.attrs["meta"] = {"n_assets": 0, "features": _features.FEATURE_COLS, "if_lof_agreement": None}
        return empty

    X = StandardScaler().fit_transform(f[_features.FEATURE_COLS].to_numpy(dtype=float))
    seg = _seg.fit_segments(f, X, random_state=random_state)
    anom, agreement = _anom.detect_anomalies(f, X, random_state=random_state)

    out = seg.merge(anom, on="asset_id", how="outer")
    out.attrs["meta"] = {
        "n_assets": int(len(f)),
        "features": _features.FEATURE_COLS,
        "if_lof_agreement": agreement,
    }
    return out
