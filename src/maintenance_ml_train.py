"""
maintenance_ml_train.py
=======================

PHASE 4B — Train + evaluate maintenance-prediction models on the LEAKAGE-SAFE
dataset from Phase 4A (``outputs/maintenance_prediction_dataset.csv``).

Does NOT rebuild the dataset. Chronological split only (train = earlier obs,
test = later obs — never random). Median imputation for numeric, "Unknown" for
categorical (fit on TRAIN only). Class imbalance via class_weight /
scale_pos_weight (NO oversampling). Compares every model against the existing
rule-based engine and recommends production only on clear, consistent lift.

Models: Logistic Regression, Random Forest, XGBoost (skipped gracefully if not
installed).

Outputs
-------
outputs/maintenance_model_metrics.csv
outputs/maintenance_feature_importance.csv
outputs/maintenance_predictions.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
_DATASET_CSV = _OUTPUT_DIR / "maintenance_prediction_dataset.csv"
_METRICS_CSV = _OUTPUT_DIR / "maintenance_model_metrics.csv"
_IMPORTANCE_CSV = _OUTPUT_DIR / "maintenance_feature_importance.csv"
_PRED_CSV = _OUTPUT_DIR / "maintenance_predictions.csv"

_LABEL = "maintenance_next_30_days"
_ID_COLS = ["asset_id", "observation_date"]
_CATEGORICAL = ["previous_priority", "previous_issue_type"]
_TEST_FRACTION = 0.30


# --------------------------------------------------------------------------- #
# preprocessing (fit on TRAIN only — no leakage)
# --------------------------------------------------------------------------- #
def _prepare(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    feats = [c for c in train.columns if c not in _ID_COLS + [_LABEL]]
    num = [c for c in feats if c not in _CATEGORICAL]
    cat = [c for c in feats if c in _CATEGORICAL]

    # drop features that are entirely missing in TRAIN (nothing to learn / impute)
    dropped = [c for c in num if train[c].isna().all()]
    num = [c for c in num if c not in dropped]
    if dropped:
        logger.info("Dropped all-NaN-in-train features: %s", dropped)

    # numeric: median imputation from TRAIN
    medians = {c: pd.to_numeric(train[c], errors="coerce").median() for c in num}
    medians = {c: (0.0 if pd.isna(m) else m) for c, m in medians.items()}

    def num_block(df):
        out = pd.DataFrame(index=df.index)
        for c in num:
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(medians[c])
        return out

    # categorical: "Unknown" impute; collapse rare (<2 in train) to keep one-hot sane
    keep_levels = {}
    for c in cat:
        vc = train[c].astype("object").where(train[c].notna(), "Unknown").astype(str).value_counts()
        keep_levels[c] = set(vc[vc >= 2].index) | {"Unknown"}

    def cat_block(df):
        parts = []
        for c in cat:
            s = df[c].astype("object").where(df[c].notna(), "Unknown").astype(str)
            s = s.where(s.isin(keep_levels[c]), "Unknown")
            parts.append(pd.get_dummies(s, prefix=c))
        return pd.concat(parts, axis=1) if parts else pd.DataFrame(index=df.index)

    Xtr = pd.concat([num_block(train), cat_block(train)], axis=1)
    Xte = pd.concat([num_block(test), cat_block(test)], axis=1)
    Xte = Xte.reindex(columns=Xtr.columns, fill_value=0)   # align dummies
    return Xtr, Xte, list(Xtr.columns)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def _metrics(name: str, y_true, proba, thr: float = 0.5) -> dict:
    from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                                 f1_score, roc_auc_score, average_precision_score,
                                 confusion_matrix)
    pred = (np.asarray(proba) >= thr).astype(int)
    both = len(set(y_true)) > 1
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, pred), 3),
        "precision": round(precision_score(y_true, pred, zero_division=0), 3),
        "recall": round(recall_score(y_true, pred, zero_division=0), 3),
        "f1": round(f1_score(y_true, pred, zero_division=0), 3),
        "roc_auc": round(roc_auc_score(y_true, proba), 3) if both else np.nan,
        "pr_auc": round(average_precision_score(y_true, proba), 3) if both else np.nan,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def run(loader=None) -> dict:
    if not _DATASET_CSV.is_file():
        raise FileNotFoundError(f"{_DATASET_CSV} missing — run Phase 4A first.")
    df = pd.read_csv(_DATASET_CSV)
    df = df.sort_values("observation_date").reset_index(drop=True)   # chronological
    cut = int(len(df) * (1 - _TEST_FRACTION))
    train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
    ytr, yte = train[_LABEL].astype(int).values, test[_LABEL].astype(int).values

    Xtr, Xte, cols = _prepare(train, test)
    pos, neg = int(ytr.sum()), int((ytr == 0).sum())
    spw = (neg / pos) if pos else 1.0

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    models = {}
    # LR needs scaling; wrap manually
    scaler = StandardScaler().fit(Xtr)
    models["LogisticRegression"] = ("lr", LogisticRegression(max_iter=2000, class_weight="balanced"))
    models["RandomForest"] = ("rf", RandomForestClassifier(n_estimators=400, random_state=42, class_weight="balanced"))
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = ("xgb", XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,
                                                  random_state=42, eval_metric="logloss", scale_pos_weight=spw))
    except Exception:  # noqa: BLE001
        logger.info("XGBoost not installed — skipping gracefully.")

    metric_rows, importance_rows = [], []
    pred_frame = test[_ID_COLS + [_LABEL]].copy().rename(columns={_LABEL: "actual"})

    for name, (kind, mdl) in models.items():
        if kind == "lr":
            mdl.fit(scaler.transform(Xtr), ytr)
            proba = mdl.predict_proba(scaler.transform(Xte))[:, 1]
            imp = dict(zip(cols, np.abs(mdl.coef_[0])))
        else:
            mdl.fit(Xtr, ytr)
            proba = mdl.predict_proba(Xte)[:, 1]
            imp = dict(zip(cols, getattr(mdl, "feature_importances_", np.zeros(len(cols)))))
        metric_rows.append(_metrics(name, yte, proba))
        pred_frame[f"{name}_proba"] = np.round(proba, 3)
        for fcol, val in sorted(imp.items(), key=lambda t: t[1], reverse=True):
            importance_rows.append({"model": name, "feature": fcol, "importance": round(float(val), 4)})

    # ---- rule-based baseline on the SAME test observations ----
    rule_row = None
    try:
        from data_loader import DataLoader
        import maintenance_risk as MR
        L = loader or DataLoader()
        scores = MR.build_risk_scores(L)
        smap = dict(zip(scores["asset_id"].astype(str), pd.to_numeric(scores["maintenance_risk_score"], errors="coerce")))
        lvlmap = dict(zip(scores["asset_id"].astype(str), scores["risk_level"].astype(str)))
        rb_proba = test["asset_id"].astype(str).map(smap).fillna(0.0).values / 100.0
        rb_pred = test["asset_id"].astype(str).map(lvlmap).isin(["High", "Critical"]).astype(int).values
        pred_frame["RuleBased_risk_score"] = (rb_proba * 100).round(1)
        # metrics: AUC/PR from score ranking; thresholded from High/Critical flag
        rm = _metrics("RuleBased(existing)", yte, rb_proba)
        # override thresholded stats with the engine's own High/Critical decision
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        tn, fp, fn, tp = confusion_matrix(yte, rb_pred, labels=[0, 1]).ravel()
        rm.update({
            "accuracy": round(accuracy_score(yte, rb_pred), 3),
            "precision": round(precision_score(yte, rb_pred, zero_division=0), 3),
            "recall": round(recall_score(yte, rb_pred, zero_division=0), 3),
            "f1": round(f1_score(yte, rb_pred, zero_division=0), 3),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
        rule_row = rm
        metric_rows.append(rm)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Rule-based comparison skipped: %s", exc)

    metrics = pd.DataFrame(metric_rows)
    metrics.insert(1, "n_train", len(ytr))
    metrics.insert(2, "n_test", len(yte))
    metrics.insert(3, "test_positives", int(yte.sum()))

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(_METRICS_CSV, index=False)
    pd.DataFrame(importance_rows).to_csv(_IMPORTANCE_CSV, index=False)
    pred_frame.to_csv(_PRED_CSV, index=False)

    recommendation = _recommend(metrics, rule_row)
    return {"metrics": metrics, "recommendation": recommendation,
            "paths": {"metrics": _METRICS_CSV, "importance": _IMPORTANCE_CSV, "predictions": _PRED_CSV}}


def _recommend(metrics: pd.DataFrame, rule_row) -> str:
    ml = metrics[~metrics["model"].str.contains("RuleBased")]
    if rule_row is None or ml.empty:
        return "Keep the rule-based engine (no valid ML/baseline comparison)."
    # best ML by F1 then PR-AUC
    ml_sorted = ml.sort_values(["f1", "pr_auc"], ascending=False)
    best = ml_sorted.iloc[0]
    rb_f1 = rule_row["f1"]
    rb_pr = rule_row.get("pr_auc")
    rb_pr = rb_pr if (rb_pr is not None and not (isinstance(rb_pr, float) and np.isnan(rb_pr))) else 0.0
    best_pr = best["pr_auc"] if not pd.isna(best["pr_auc"]) else 0.0
    clear = (best["f1"] >= rb_f1 + 0.10) and (best_pr >= rb_pr + 0.05) and (best["recall"] >= rule_row["recall"])
    if clear:
        return (f"Recommend {best['model']} for production — clearly outperforms the rule-based engine "
                f"(F1 {best['f1']} vs {rb_f1}, PR-AUC {best_pr} vs {rb_pr}, recall {best['recall']} vs {rule_row['recall']}).")
    return ("KEEP the rule-based engine as production. No ML model clearly and consistently outperforms it "
            f"(best ML {best['model']}: F1 {best['f1']} vs rule {rb_f1}, PR-AUC {best_pr} vs {rb_pr}). "
            "Given the tiny/noisy test fold, the difference is not decision-grade.")


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING, format="%(levelname)s %(message)s")
    res = run()
    print("=== MODEL METRICS (chronological test fold) ===")
    print(res["metrics"].to_string(index=False))
    print("\n=== RECOMMENDATION ===")
    print(res["recommendation"])
    print("\nsaved:", {k: str(v) for k, v in res["paths"].items()})
