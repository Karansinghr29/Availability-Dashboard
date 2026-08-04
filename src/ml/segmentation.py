"""
segmentation.py — unsupervised asset risk SEGMENTATION with a Gaussian Mixture Model.

Why GMM (not plain KMeans)?
---------------------------
KMeans assigns hard clusters and assumes spherical, equal-variance groups.
Maintenance features are skewed and correlated (a few very-high-ticket assets),
so GMM is preferred because:
  * it models elliptical clusters with per-cluster covariance, and
  * it returns a soft membership PROBABILITY per asset — which we surface as
    "Cluster Probability" so the manager sees how confident the segment is.
No labels are used; clusters are discovered purely from feature structure.

Mapping clusters -> risk segments
---------------------------------
GMM cluster IDs are arbitrary. We rank clusters by a transparent SEVERITY proxy
(mean of the standardised risk-bearing features) and map the ordered clusters to
Low / Medium / High / Critical. This ordering is deterministic and explainable —
it is NOT a learned label, just a post-hoc naming of discovered clusters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

# risk-bearing features whose HIGHER value = riskier (used only to order clusters)
_SEVERITY_FEATURES = ["ticket_count", "repeat_ticket_count", "recent_30_days",
                      "age_from_allocation", "issue_diversity"]
_SEGMENTS = ["Low", "Medium", "High", "Critical"]


def fit_segments(features: pd.DataFrame, X_scaled: np.ndarray, random_state: int = 42) -> pd.DataFrame:
    """Return per-asset: ml_cluster, ml_cluster_prob, ml_risk_segment.
    Degrades gracefully when there are too few assets for 4 components."""
    n = len(features)
    if n == 0:
        return pd.DataFrame(columns=["asset_id", "ml_cluster", "ml_cluster_prob", "ml_risk_segment"])

    # can't ask for more mixture components than samples; also keep segments meaningful
    k = min(4, n)
    gmm = GaussianMixture(
        n_components=k,
        covariance_type="full",   # per-cluster elliptical covariance (see docstring)
        random_state=random_state,
        n_init=5,                 # several restarts -> stable, reproducible fit
        reg_covar=1e-4,           # guard against singular covariance on small data
    )
    labels = gmm.fit_predict(X_scaled)
    proba = gmm.predict_proba(X_scaled)
    cluster_prob = proba.max(axis=1)

    # order clusters by severity (mean of standardised severity features)
    fx = features.copy()
    sev_cols = [c for c in _SEVERITY_FEATURES if c in fx.columns]
    z = (fx[sev_cols] - fx[sev_cols].mean()) / (fx[sev_cols].std(ddof=0).replace(0, 1))
    fx["_sev"] = z.mean(axis=1)
    fx["_cluster"] = labels
    order = fx.groupby("_cluster")["_sev"].mean().sort_values().index.tolist()  # low->high severity
    seg_names = _SEGMENTS[-k:] if k < 4 else _SEGMENTS  # if <4 clusters, use the top labels
    cluster_to_seg = {c: seg_names[i] for i, c in enumerate(order)}

    return pd.DataFrame({
        "asset_id": features["asset_id"].astype(str).values,
        "ml_cluster": labels.astype(int),
        "ml_cluster_prob": np.round(cluster_prob * 100, 1),          # % membership confidence
        "ml_risk_segment": [cluster_to_seg[c] for c in labels],
    })
