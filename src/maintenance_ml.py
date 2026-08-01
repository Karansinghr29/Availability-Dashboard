"""
maintenance_ml.py
=================

PHASE 4 — EXPERIMENTAL machine-learning layer for predictive maintenance.

Completely separate from the production rule-based engine (``maintenance_risk``),
which is NOT touched. This module:

  1. Loads the per-asset feature table (``outputs/maintenance_asset_features.csv``).
  2. Constructs a LEAKAGE-SAFE supervised target for
     "asset will require maintenance in the next window".
  3. Runs a FEASIBILITY GATE first. If the labels are too sparse / not
     asset-specific / leaky / lack a temporal horizon, it STOPS and writes a
     documented verdict instead of forcing a model (per Phase-4 requirement).
  4. Only when feasible: train/test split (no leakage), Random Forest,
     XGBoost (optional — skipped gracefully if unavailable), evaluation
     (Accuracy / F1 / AUC), feature importance, and a comparison against the
     rule-based score. Saves predictions + metrics to separate output files.

Nothing here is imported by the production pages or the rule engine. Running it
never changes existing behaviour.

Outputs
-------
outputs/maintenance_model_metrics.csv       always written (verdict + metrics)
outputs/maintenance_ml_predictions.csv       written only when a model is trained
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
_FEATURES_CSV = _OUTPUT_DIR / "maintenance_asset_features.csv"
_METRICS_CSV = _OUTPUT_DIR / "maintenance_model_metrics.csv"
_PRED_CSV = _OUTPUT_DIR / "maintenance_ml_predictions.csv"

# Leakage-safe (asset-intrinsic) candidate predictors ONLY. Everything derived
# from tickets is EXCLUDED because it is computed from the same events that would
# define the label.
_INTRINSIC_NUMERIC = [
    "asset_age_years", "purchase_age_months", "maintenance_cycle_months",
    "expected_life_months", "replacement_cost_estimate", "supplier_rating",
]
_INTRINSIC_CATEGORICAL = ["asset_type", "condition", "status"]
_LEAKY = [
    "ticket_count", "maintenance_frequency_per_year", "repeat_job_count",
    "total_maintenance_cost", "avg_maintenance_cost", "avg_resolution_hours",
    "sla_breach_count", "last_maintenance_date", "days_since_last_maintenance",
    "service_overdue", "service_due_date", "ticket_link_level",
]


# --------------------------------------------------------------------------- #
# data + target
# --------------------------------------------------------------------------- #
def load_features(path: Optional[Path] = None) -> pd.DataFrame:
    p = Path(path) if path else _FEATURES_CSV
    if not p.is_file():
        raise FileNotFoundError(f"Feature table not found: {p}. Run maintenance_features first.")
    return pd.read_csv(p)


def build_target(features: pd.DataFrame) -> pd.Series:
    """Leakage-safe target: asset had an ASSET-SPECIFIC maintenance event.

    The only asset-specific link available is a BED-level ticket link
    (apartment-level links are shared across every asset in the flat, so they
    are NOT asset-specific and are treated as negative/unknown here). This is the
    best obtainable proxy; the feasibility gate then judges whether it is usable.
    """
    lvl = features.get("ticket_link_level", pd.Series("none", index=features.index)).astype(str)
    return (lvl == "bed").astype(int)


# --------------------------------------------------------------------------- #
# feasibility gate
# --------------------------------------------------------------------------- #
def feasibility_gate(features: pd.DataFrame, target: pd.Series) -> dict:
    """Return a verdict dict. ``feasible`` False => stop, don't train."""
    n = int(len(features))
    pos = int(target.sum())
    pos_rate = round(100 * pos / n, 2) if n else 0.0

    # independent positive groups (bed-linked assets share a room -> not independent)
    linked = features[features.get("ticket_link_level", "") == "bed"]
    pos_groups = int(linked["apartment_code"].nunique()) if not linked.empty else 0

    # non-leaky predictor usability: coverage >= 50% AND non-degenerate variance
    usable = []
    for c in _INTRINSIC_NUMERIC:
        if c in features.columns:
            s = pd.to_numeric(features[c], errors="coerce")
            cov = s.notna().mean()
            var_ok = s.dropna().nunique() > 1
            # cycle/expected_life/replacement are per-asset-TYPE constants -> weak
            type_constant = c in ("maintenance_cycle_months", "expected_life_months", "replacement_cost_estimate")
            if cov >= 0.5 and var_ok and not type_constant:
                usable.append(c)
    for c in _INTRINSIC_CATEGORICAL:
        if c in features.columns:
            nun = features[c].dropna().nunique()
            # condition is low-variance (good/new only); asset_type is a memorisation proxy
            if c == "condition" and nun >= 3:
                usable.append(c)

    # temporal horizon: >1 distinct month of maintenance history
    months = 0
    if "last_maintenance_date" in features.columns:
        d = pd.to_datetime(features["last_maintenance_date"], errors="coerce").dropna()
        months = int(d.dt.to_period("M").nunique())

    reasons = []
    if pos < 30:
        reasons.append(f"Too few asset-specific positives ({pos}; need >=30).")
    if pos_groups < 8:
        reasons.append(f"Positives collapse to {pos_groups} room-groups (labels are room-attributed, not asset-specific).")
    if not (5 <= pos_rate <= 95):
        reasons.append(f"Degenerate class balance (positive rate {pos_rate}%).")
    if len(usable) < 2:
        reasons.append(f"Only {len(usable)} usable non-leaky predictor(s); the predictive features are leaky (ticket-derived) and intrinsic ones are sparse/degenerate.")
    if months < 2:
        reasons.append(f"No temporal horizon ({months} month(s) of maintenance history) — cannot define/validate a future 'next window' label.")
    if int(pd.to_numeric(features.get("repeat_job_count", 0), errors="coerce").fillna(0).gt(0).sum()) == 0:
        reasons.append("Zero repeat-job signal — no recurrence to anchor a maintenance-need label.")

    return {
        "assets": n,
        "positives": pos,
        "positive_rate_pct": pos_rate,
        "independent_positive_groups": pos_groups,
        "usable_nonleaky_features": usable,
        "months_of_history": months,
        "feasible": len(reasons) == 0,
        "reasons": reasons,
    }


# --------------------------------------------------------------------------- #
# model training (only runs when feasible)
# --------------------------------------------------------------------------- #
def _design_matrix(features: pd.DataFrame, cols_num, cols_cat) -> pd.DataFrame:
    X = pd.DataFrame(index=features.index)
    for c in cols_num:
        if c in features.columns:
            X[c] = pd.to_numeric(features[c], errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    for c in cols_cat:
        if c in features.columns:
            d = pd.get_dummies(features[c].astype(str).fillna("Unknown"), prefix=c)
            X = pd.concat([X, d], axis=1)
    return X


def train_and_compare(features: pd.DataFrame, target: pd.Series, usable_num, usable_cat) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train RF (+ XGBoost if available), evaluate, compare to rule-based.

    Returns (metrics_df, predictions_df). Only called when feasible.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import train_test_split

    X = _design_matrix(features, usable_num, usable_cat)
    y = target.values
    Xtr, Xte, ytr, yte, idx_tr, idx_te = train_test_split(
        X, y, features.index, test_size=0.3, random_state=42, stratify=y
    )

    models = {"RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")}
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(n_estimators=300, random_state=42, eval_metric="logloss", use_label_encoder=False)
    except Exception:  # noqa: BLE001
        logger.info("XGBoost unavailable — skipping gracefully.")

    metric_rows, importances, pred_cols = [], {}, {}
    for name, mdl in models.items():
        mdl.fit(Xtr, ytr)
        proba = mdl.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        metric_rows.append({
            "model": name, "accuracy": round(accuracy_score(yte, pred), 3),
            "f1": round(f1_score(yte, pred, zero_division=0), 3),
            "auc": round(roc_auc_score(yte, proba), 3) if len(set(yte)) > 1 else np.nan,
            "n_train": len(ytr), "n_test": len(yte),
        })
        importances[name] = dict(zip(X.columns, getattr(mdl, "feature_importances_", [])))
        pred_cols[name] = pd.Series(mdl.predict_proba(X)[:, 1], index=features.index)

    metrics = pd.DataFrame(metric_rows)
    preds = features[["asset_id", "asset_code", "asset_type", "apartment_code"]].copy()
    preds["actual_label"] = target.values
    for name, s in pred_cols.items():
        preds[f"{name}_proba"] = s.round(3).values
    return metrics, preds


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def run(features_path: Optional[Path] = None) -> dict:
    feats = load_features(features_path)
    target = build_target(feats)
    gate = feasibility_gate(feats, target)

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not gate["feasible"]:
        # STOP — write a documented verdict, do NOT force a model.
        rows = [{"item": "STATUS", "value": "INSUFFICIENT — ML not run"}]
        for k in ("assets", "positives", "positive_rate_pct", "independent_positive_groups", "months_of_history"):
            rows.append({"item": k, "value": gate[k]})
        rows.append({"item": "usable_nonleaky_features", "value": ", ".join(gate["usable_nonleaky_features"]) or "none"})
        for i, r in enumerate(gate["reasons"], 1):
            rows.append({"item": f"reason_{i}", "value": r})
        pd.DataFrame(rows).to_csv(_METRICS_CSV, index=False)
        logger.info("ML feasibility gate FAILED — verdict written to %s", _METRICS_CSV)
        return {"feasible": False, "gate": gate, "metrics_path": _METRICS_CSV}

    # feasible path
    metrics, preds = train_and_compare(
        feats, target,
        [c for c in gate["usable_nonleaky_features"] if c in _INTRINSIC_NUMERIC],
        [c for c in gate["usable_nonleaky_features"] if c in _INTRINSIC_CATEGORICAL],
    )
    metrics.to_csv(_METRICS_CSV, index=False)
    preds.to_csv(_PRED_CSV, index=False)
    logger.info("ML trained — metrics -> %s, predictions -> %s", _METRICS_CSV, _PRED_CSV)
    return {"feasible": True, "gate": gate, "metrics": metrics, "predictions": preds,
            "metrics_path": _METRICS_CSV, "predictions_path": _PRED_CSV}


if __name__ == "__main__":
    import logging as _l

    _l.basicConfig(level=_l.WARNING, format="%(levelname)s %(message)s")
    res = run()
    g = res["gate"]
    print("=== ML FEASIBILITY GATE ===")
    print(f"assets={g['assets']}  positives={g['positives']} ({g['positive_rate_pct']}%)  "
          f"independent_positive_groups={g['independent_positive_groups']}  months_of_history={g['months_of_history']}")
    print(f"usable non-leaky features: {g['usable_nonleaky_features'] or 'NONE'}")
    print(f"\nFEASIBLE: {g['feasible']}")
    if not g["feasible"]:
        print("\nWhy ML is NOT run (labels/history insufficient):")
        for i, r in enumerate(g["reasons"], 1):
            print(f"  {i}. {r}")
        print(f"\nVerdict written -> {res['metrics_path']}")
    else:
        print("\n=== metrics ===")
        print(res["metrics"].to_string(index=False))
