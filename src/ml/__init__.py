"""
src/ml — ADDITIVE, unsupervised machine-learning layer for the maintenance dashboard.

Design intent (read before extending)
--------------------------------------
The production predictor is the transparent RULE + POISSON engine in
``asset_engine.py``. This package does NOT replace it. It adds two genuinely
unsupervised models that need no labels — chosen precisely because the current
data has NO reliable failure/replacement target:

  * Gaussian Mixture Model (GMM)  -> asset risk SEGMENTATION  (segmentation.py)
  * Isolation Forest (+ optional LOF) -> asset ANOMALY detection (anomaly.py)

Both read ONLY features that already exist in the rule engine's output
(ticket_count, repeat_count, recent_30d, age, asset_type, issue diversity).
No supervised target is invented or fabricated.

Public entry point: ``pipeline.run_asset_ml(assets_df, mapped)``.
"""

from .pipeline import run_asset_ml

__all__ = ["run_asset_ml"]
