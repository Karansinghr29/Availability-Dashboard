"""
features.py — build the per-asset feature matrix for the unsupervised ML layer.

All features are taken from outputs the RULE engine already computes
(``asset_engine.build``) plus the ticket→asset map. Nothing new is predicted
here and no target/label is created.

Feature set (exactly the one requested):
  ticket_count         — total maintenance tickets pinned to the asset
  repeat_ticket_count  — # issue types that recurred (>=2) for the asset
  recent_30_days       — tickets in the last 30 days
  age_from_allocation  — asset age in months (purchase date where present,
                         else earliest allocation date — the engine's age_months;
                         for 92% of scored assets this IS the allocation age)
  asset_type           — encoded (see note below)
  issue_diversity      — # distinct issue types seen for the asset

asset_type encoding note
------------------------
With only ~251 scored assets, one-hot encoding ~15 types would add many sparse
columns and destabilise GMM covariance estimation. We therefore FREQUENCY-encode
asset_type (how common the type is in the scored population). This keeps the
feature space low-dimensional and stable while still letting the models separate
common vs rare asset types. Documented trade-off, not a hidden choice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "ticket_count",
    "repeat_ticket_count",
    "recent_30_days",
    "age_from_allocation",
    "asset_type_freq",
    "issue_diversity",
]


def build_feature_frame(assets_df: pd.DataFrame, mapped: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by asset_id with FEATURE_COLS (raw, unscaled).
    Empty frame if there are no scored assets."""
    if assets_df is None or assets_df.empty:
        return pd.DataFrame(columns=["asset_id"] + FEATURE_COLS)

    a = assets_df.copy()
    a["asset_id"] = a["asset_id"].astype(str)

    # issue diversity per asset from the ticket→asset map (asset-pinned rows only)
    div = {}
    if mapped is not None and not mapped.empty:
        att = mapped[mapped["asset_id"].astype(str) != ""]
        for aid, g in att.groupby(att["asset_id"].astype(str)):
            div[aid] = int(g["issue_type"].replace("", np.nan).dropna().nunique())

    # frequency encoding of asset_type (share of scored population)
    type_counts = a["asset_type"].astype(str).value_counts()
    n = max(len(a), 1)

    f = pd.DataFrame({
        "asset_id": a["asset_id"].values,
        "ticket_count": pd.to_numeric(a.get("ticket_count"), errors="coerce").fillna(0.0).values,
        "repeat_ticket_count": pd.to_numeric(a.get("repeat_count"), errors="coerce").fillna(0.0).values,
        "recent_30_days": pd.to_numeric(a.get("recent_30d"), errors="coerce").fillna(0.0).values,
        "age_from_allocation": pd.to_numeric(a.get("age_months"), errors="coerce").values,
        "asset_type_freq": (a["asset_type"].astype(str).map(type_counts).fillna(0) / n).values,
        "issue_diversity": a["asset_id"].map(lambda x: div.get(str(x), 0)).values,
    })
    # age missing (no purchase and no allocation) -> median age of the population
    med = np.nanmedian(f["age_from_allocation"]) if np.isfinite(np.nanmedian(f["age_from_allocation"])) else 0.0
    f["age_from_allocation"] = f["age_from_allocation"].fillna(med)
    return f.reset_index(drop=True)
