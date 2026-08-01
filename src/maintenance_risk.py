"""
maintenance_risk.py
===================

PHASE 3B — Rule-based Maintenance Risk Score engine (NO machine learning).

Consumes the per-asset feature table (``maintenance_features.build_asset_features``)
and produces, per asset:
    maintenance_risk_score        0-100 (higher = more at risk)
    risk_level                    Low / Medium / High / Critical
    confidence_score              0-100 (how much of the feature set was present)
    top_contributing_factors      the biggest score drivers for this asset
    recommended_action            Monitor / Inspect / Schedule Service / Replace

Design (transparent, deterministic — see ``FACTORS`` below)
----------------------------------------------------------
Each factor maps a feature to a sub-score in [0, 1] (higher = riskier) and a
weight. High-coverage / high-signal fields carry the most weight. A factor whose
feature is missing returns ``None`` (unavailable) and simply drops out — the
score is the weighted average over the AVAILABLE factors only, so missing data
never silently reads as "low risk":

    risk = 100 * Σ(weight_i · sub_i) / Σ(weight_i)      over available factors i
    confidence = 100 * Σ(weight_available) / Σ(weight_all)

Sparse assets (no purchase date, no ticket history) therefore get a score from
whatever IS known plus a LOW confidence, exactly as required.

Reusable: ``score_features(df)`` scores any feature frame; a later ML model can
reuse the same feature table without touching this rule engine.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import pandas as pd

import maintenance_features as MF

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


# --------------------------------------------------------------------------- #
# sub-score helpers (each returns float in [0,1] risk, or None if unavailable)
# --------------------------------------------------------------------------- #
def _f(v):
    x = pd.to_numeric(v, errors="coerce")
    return None if pd.isna(x) else float(x)


def _cycle(row):
    """Shorter maintenance cycle = higher-touch asset = higher baseline risk."""
    c = _f(row.get("maintenance_cycle_months"))
    if c is None:
        return None
    if c <= 3:
        return 1.0
    if c <= 6:
        return 0.7
    if c <= 12:
        return 0.4
    if c <= 24:
        return 0.2
    return 0.1


def _overdue(row):
    """Overdue for scheduled service (needs purchase_date + cycle)."""
    sd = pd.to_datetime(row.get("service_due_date"), errors="coerce")
    if pd.isna(sd):
        return None
    today = pd.Timestamp.today().normalize()
    if sd <= today:
        return 1.0
    if sd <= today + pd.Timedelta(days=30):
        return 0.6
    return 0.1


def _condition(row):
    c = str(row.get("condition") or "").strip().lower()
    if not c:
        return None
    return {"new": 0.0, "good": 0.2, "fair": 0.6, "average": 0.6,
            "poor": 1.0, "bad": 1.0, "damaged": 1.0}.get(c, 0.3)


def _status(row):
    s = str(row.get("status") or "").strip().lower().replace(" ", "_")
    if not s:
        return None
    if s in {"under_maintenance", "maintenance", "repair", "faulty"}:
        return 0.7
    if s in {"retired", "inactive", "disposed", "scrapped", "not_active"}:
        return 0.1
    # active / in_use / allocated / in_service -> in operation = normal wear
    return 0.3


def _ticket_count(row):
    if str(row.get("ticket_link_level") or "none") == "none":
        return None
    n = _f(row.get("ticket_count")) or 0
    if n <= 0:
        return 0.0
    if n == 1:
        return 0.3
    if n <= 3:
        return 0.6
    return 1.0


def _repeat(row):
    if str(row.get("ticket_link_level") or "none") == "none":
        return None
    n = _f(row.get("repeat_job_count")) or 0
    if n <= 0:
        return 0.0
    return 0.6 if n == 1 else 1.0


def _sla(row):
    if str(row.get("ticket_link_level") or "none") == "none":
        return None
    n = _f(row.get("sla_breach_count")) or 0
    if n <= 0:
        return 0.0
    return 0.5 if n == 1 else 1.0


def _freq(row):
    v = _f(row.get("maintenance_frequency_per_year"))
    if v is None:
        return None
    if v <= 0:
        return 0.0
    if v <= 1:
        return 0.3
    if v <= 3:
        return 0.6
    return 1.0


def _cost(row):
    if str(row.get("ticket_link_level") or "none") == "none":
        return None
    v = _f(row.get("total_maintenance_cost"))
    if v is None or v <= 0:
        return 0.0
    if v <= 1000:
        return 0.3
    if v <= 5000:
        return 0.6
    return 1.0


def _resolution(row):
    v = _f(row.get("avg_resolution_hours"))
    if v is None:
        return None
    if v <= 24:
        return 0.2
    if v <= 72:
        return 0.5
    if v <= 168:
        return 0.7
    return 1.0


def _age(row):
    """Age vs expected life when both known, else age buckets. Needs purchase date."""
    yrs = _f(row.get("asset_age_years"))
    if yrs is None:
        return None
    life = _f(row.get("expected_life_months"))
    if life and life > 0:
        ratio = (yrs * 12.0) / life
        if ratio >= 1.0:
            return 1.0
        if ratio >= 0.75:
            return 0.7
        if ratio >= 0.5:
            return 0.4
        return 0.2
    if yrs < 1:
        return 0.1
    if yrs < 3:
        return 0.3
    if yrs < 5:
        return 0.6
    return 1.0


def _warranty(row):
    """Under active warranty = covered = lower risk; expired = mild risk."""
    w = row.get("warranty_active")
    if w is None or (isinstance(w, float) and pd.isna(w)) or str(w).strip() in ("", "nan", "<NA>", "None"):
        return None
    return 0.0 if bool(w) is True else 0.6


# name, weight, sub-score fn, human label
FACTORS: List[Tuple[str, float, Callable, str]] = [
    ("maintenance_cycle", 0.18, _cycle, "Short maintenance cycle"),
    ("service_overdue", 0.14, _overdue, "Overdue for service"),
    ("repeat_jobs", 0.15, _repeat, "Repeat maintenance jobs"),
    ("sla_breaches", 0.10, _sla, "SLA breaches"),
    ("condition", 0.10, _condition, "Poor condition"),
    ("ticket_count", 0.08, _ticket_count, "High ticket volume"),
    ("asset_age", 0.08, _age, "Asset age vs expected life"),
    ("maintenance_frequency", 0.05, _freq, "High maintenance frequency"),
    ("maintenance_cost", 0.05, _cost, "High maintenance cost"),
    ("resolution_time", 0.05, _resolution, "Slow resolution"),
    ("status", 0.05, _status, "Operational status"),
    ("warranty", 0.04, _warranty, "Warranty expired"),
]
_TOTAL_WEIGHT = sum(w for _, w, _, _ in FACTORS)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _level(score: float) -> str:
    if score >= 75:
        return "Critical"
    if score >= 50:
        return "High"
    if score >= 25:
        return "Medium"
    return "Low"


def _action(level: str, row: pd.Series) -> str:
    overdue = _overdue(row) == 1.0
    age_r = _age(row)
    cond = str(row.get("condition") or "").lower()
    end_of_life = (age_r == 1.0) or cond in {"poor", "bad", "damaged"}
    if level == "Critical":
        return "Replace" if end_of_life else "Schedule Service"
    if level == "High":
        return "Schedule Service"
    if level == "Medium":
        return "Schedule Service" if overdue else "Inspect"
    return "Inspect" if overdue else "Monitor"


def score_features(features: pd.DataFrame) -> pd.DataFrame:
    """Score a per-asset feature frame. Pure — no I/O, no loader."""
    if features is None or features.empty:
        return pd.DataFrame()

    out_rows = []
    for _, row in features.iterrows():
        contribs = []          # (name, weight, sub, contribution)
        wsum = 0.0
        risk_acc = 0.0
        for name, weight, fn, label in FACTORS:
            sub = fn(row)
            if sub is None:
                continue
            wsum += weight
            risk_acc += weight * sub
            contribs.append((label, weight * sub))
        score = round(100.0 * risk_acc / wsum, 1) if wsum > 0 else 0.0
        confidence = round(100.0 * wsum / _TOTAL_WEIGHT, 1)
        level = _level(score)
        # top drivers (non-zero contributions, highest first)
        top = [n for n, c in sorted(contribs, key=lambda t: t[1], reverse=True) if c > 0][:3]
        out_rows.append({
            "asset_id": row.get("asset_id"),
            "asset_code": row.get("asset_code"),
            "asset_type": row.get("asset_type"),
            "apartment_code": row.get("apartment_code"),
            "condition": row.get("condition"),
            "status": row.get("status"),
            "maintenance_risk_score": score,
            "risk_level": level,
            "confidence_score": confidence,
            "top_contributing_factors": "; ".join(top) if top else "—",
            "recommended_action": _action(level, row),
            # carry a few key inputs for traceability
            "ticket_count": row.get("ticket_count"),
            "repeat_job_count": row.get("repeat_job_count"),
            "sla_breach_count": row.get("sla_breach_count"),
            "service_overdue": row.get("service_overdue"),
            "maintenance_cycle_months": row.get("maintenance_cycle_months"),
            "asset_age_years": row.get("asset_age_years"),
            "ticket_link_level": row.get("ticket_link_level"),
        })
    res = pd.DataFrame(out_rows)
    return res.sort_values(["maintenance_risk_score", "confidence_score"], ascending=[False, False]).reset_index(drop=True)


def build_risk_scores(loader) -> pd.DataFrame:
    """Build features then score them (convenience)."""
    feats = MF.build_asset_features(loader)
    return score_features(feats)


# --------------------------------------------------------------------------- #
# dashboard aggregates (pure — dashboard section consumes these)
# --------------------------------------------------------------------------- #
_LEVELS = ["Low", "Medium", "High", "Critical"]


def risk_summary(scores: pd.DataFrame) -> dict:
    """KPIs, distributions and the three action tables for the dashboard."""
    empty = {
        "kpis": {}, "distribution": pd.DataFrame(), "by_type": pd.DataFrame(),
        "by_apartment": pd.DataFrame(), "confidence_buckets": pd.DataFrame(),
        "high_risk": pd.DataFrame(), "due_for_service": pd.DataFrame(),
        "replacement": pd.DataFrame(),
    }
    if scores is None or scores.empty:
        return empty

    lvl = scores["risk_level"].astype(str)
    due_mask = scores["service_overdue"].astype(str).str.lower().isin({"true", "1"})
    replace_mask = scores["recommended_action"].astype(str) == "Replace"
    kpis = {
        "critical_assets": int((lvl == "Critical").sum()),
        "high_risk_assets": int((lvl == "High").sum()),
        "due_for_service": int(due_mask.sum()),
        "replacement_candidates": int(replace_mask.sum()),
        "avg_risk_score": round(float(pd.to_numeric(scores["maintenance_risk_score"], errors="coerce").mean()), 1),
        "avg_confidence": round(float(pd.to_numeric(scores["confidence_score"], errors="coerce").mean()), 1),
    }

    distribution = (scores["risk_level"].value_counts()
                    .reindex(_LEVELS, fill_value=0)
                    .rename_axis("risk_level").reset_index(name="count"))

    def _agg(col, top=20):
        s = scores.copy()
        s["_g"] = s[col].map(lambda v: str(v).strip() if str(v).strip() and str(v).lower() != "nan" else "")
        s = s[s["_g"] != ""]
        if s.empty:
            return pd.DataFrame()
        g = (s.groupby("_g")
             .agg(avg_risk_score=("maintenance_risk_score", lambda x: round(pd.to_numeric(x, errors="coerce").mean(), 1)),
                  assets=("asset_id", "count"))
             .reset_index().rename(columns={"_g": col}))
        return g.sort_values("avg_risk_score", ascending=False).head(top)

    by_type = _agg("asset_type")
    by_apartment = _agg("apartment_code")

    conf = pd.to_numeric(scores["confidence_score"], errors="coerce")
    buckets = pd.cut(conf, [-0.01, 25, 50, 75, 100], labels=["0-25", "25-50", "50-75", "75-100"])
    confidence_buckets = buckets.value_counts().reindex(["0-25", "25-50", "50-75", "75-100"], fill_value=0).rename_axis("confidence").reset_index(name="count")

    disp = ["asset_code", "asset_type", "apartment_code", "maintenance_risk_score",
            "confidence_score", "top_contributing_factors", "recommended_action"]
    disp = [c for c in disp if c in scores.columns]
    _rename = {
        "asset_code": "Asset Code", "asset_type": "Asset Type",
        "apartment_code": "Apartment", "maintenance_risk_score": "Risk Score",
        "confidence_score": "Confidence",
        "top_contributing_factors": "Top Contributing Factors",
        "recommended_action": "Recommended Action",
    }

    def _disp(mask):
        return scores[mask][disp].rename(columns=_rename).reset_index(drop=True)

    high_risk = _disp(lvl.isin(["High", "Critical"]))
    due_for_service = _disp(due_mask)
    replacement = _disp(replace_mask)

    return {
        "kpis": kpis, "distribution": distribution, "by_type": by_type,
        "by_apartment": by_apartment, "confidence_buckets": confidence_buckets,
        "high_risk": high_risk, "due_for_service": due_for_service, "replacement": replacement,
    }


def save_risk_scores(loader, path: Optional[Path] = None) -> Path:
    res = build_risk_scores(loader)
    out = Path(path) if path else (_OUTPUT_DIR / "maintenance_risk_scores.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(out, index=False)
    logger.info("Saved maintenance risk scores -> %s (%d assets)", out, len(res))
    return out


if __name__ == "__main__":
    import logging as _l
    from data_loader import DataLoader

    _l.basicConfig(level=_l.WARNING, format="%(levelname)s %(message)s")
    L = DataLoader()
    res = build_risk_scores(L)
    print(f"data_dir: {L.data_dir}")
    print(f"\nScored {len(res)} assets. Factor weights (sum={_TOTAL_WEIGHT:.2f}):")
    for name, w, _, label in FACTORS:
        print(f"  {name:22s} w={w:.2f}  ({label})")
    print("\n=== risk_level distribution ===")
    print(res["risk_level"].value_counts().to_string())
    print("\n=== recommended_action distribution ===")
    print(res["recommended_action"].value_counts().to_string())
    print("\n=== confidence summary ===")
    print(res["confidence_score"].describe()[["min", "50%", "max"]].to_string())
    p = save_risk_scores(L)
    print(f"\nSaved -> {p}")
    print("\n=== top 10 highest-risk assets ===")
    cols = ["asset_code", "asset_type", "apartment_code", "maintenance_risk_score",
            "risk_level", "confidence_score", "recommended_action", "top_contributing_factors"]
    print(res[cols].head(10).to_string(index=False))
