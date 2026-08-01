"""
maintenance_dataset.py
======================

PHASE 4A — Build the LEAKAGE-SAFE asset-level predictive-maintenance dataset.
NO model training here. Dataset only.

Source of truth: ``maintenance_tickets`` (now carries ``asset_id``), joined to
``asset_master`` (purchase/warranty/type) and ``asset_types`` (cycle/expected
life). The old room-attributed approximation is NOT used.

One OBSERVATION per ticket that has a non-null ``asset_id``, taken at the
ticket's ``created_at`` (``observation_date``). Every feature is computed from
history STRICTLY BEFORE ``observation_date``; the label looks only at FUTURE
tickets. Features and label never share information → no leakage.

Right-censoring: observations whose 30-day future window extends past the last
ticket date have an unobservable outcome, so their negative label would be
biased. They are excluded from the final dataset and reported separately.

Target
------
``maintenance_next_30_days`` = 1 if the SAME asset gets another ticket within
(observation_date, observation_date + 30 days]; else 0.

Outputs
-------
outputs/maintenance_prediction_dataset.csv
outputs/maintenance_prediction_dataset_report.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"
_DATASET_CSV = _OUTPUT_DIR / "maintenance_prediction_dataset.csv"
_REPORT_CSV = _OUTPUT_DIR / "maintenance_prediction_dataset_report.csv"

_HORIZON_DAYS = 30

FEATURE_COLUMNS = [
    "asset_id", "observation_date",
    "asset_age_days", "purchase_age_months", "expected_life_months",
    "maintenance_cycle_months", "days_since_last_ticket",
    "previous_ticket_count", "previous_30_day_ticket_count",
    "previous_90_day_ticket_count", "previous_total_maintenance_cost",
    "previous_average_resolution_hours", "previous_issue_type",
    "previous_priority", "service_overdue", "warranty_active",
    "repeat_failure_count",
]


def _naive(series) -> pd.Series:
    d = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return d.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return pd.to_datetime(series, errors="coerce")


def _s(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    t = str(v).strip()
    return "" if t.lower() in ("", "null", "nan", "none", "<na>") else t


# --------------------------------------------------------------------------- #
def build_prediction_dataset(loader, horizon_days: int = _HORIZON_DAYS) -> Dict[str, pd.DataFrame]:
    """Return {'dataset': df, 'report': df}. Leakage-safe, dataset only."""
    tickets = loader.maintenance_tickets()
    if tickets is None or tickets.empty or "asset_id" not in tickets.columns:
        raise FileNotFoundError("maintenance_tickets (with asset_id) not available.")

    t = tickets.copy()
    t["asset_id"] = t["asset_id"].map(_s)
    t = t[t["asset_id"] != ""].copy()                       # asset_id NOT NULL only
    t["created_at"] = _naive(t.get("created_at"))
    t["resolved_at"] = _naive(t.get("resolved_at"))
    t = t[t["created_at"].notna()].copy()
    t["closure_cost"] = pd.to_numeric(t.get("closure_cost"), errors="coerce")
    t["issue_type_id"] = t.get("issue_type_id", "").map(_s)
    t["priority"] = t.get("priority", "").map(lambda v: _s(v).lower())
    t["res_hours"] = (t["resolved_at"] - t["created_at"]).dt.total_seconds() / 3600.0
    t.loc[t["res_hours"] < 0, "res_hours"] = np.nan
    t = t.sort_values("created_at").reset_index(drop=True)

    # asset-intrinsic attributes (static, known at any observation date)
    assets = loader.asset_master()
    types = loader.asset_types()
    a_purchase = {}
    a_warranty = {}
    a_typeid = {}
    if not assets.empty:
        pur = _naive(assets.get("purchase_date"))
        war = _naive(assets.get("warranty_expiry"))
        for i, aid in enumerate(assets.get("id", pd.Series(dtype=str)).map(_s)):
            if aid:
                a_purchase[aid] = pur.iloc[i]
                a_warranty[aid] = war.iloc[i]
                a_typeid[aid] = _s(assets.iloc[i].get("asset_type_id"))
    cyc_map, life_map = {}, {}
    if not types.empty:
        cyc = pd.to_numeric(types.get("maintenance_cycle_months"), errors="coerce")
        life = pd.to_numeric(types.get("expected_life_months"), errors="coerce")
        for i, tid in enumerate(types.get("id", pd.Series(dtype=str)).map(_s)):
            if tid:
                cyc_map[tid] = cyc.iloc[i]
                life_map[tid] = life.iloc[i]

    max_date = t["created_at"].max()
    horizon = pd.Timedelta(days=horizon_days)
    per_asset = {aid: g for aid, g in t.groupby("asset_id")}

    rows, censored = [], 0
    for _, tk in t.iterrows():
        aid = tk["asset_id"]
        obs = tk["created_at"]
        g = per_asset[aid]
        prior = g[g["created_at"] < obs]                    # STRICTLY before obs
        future = g[(g["created_at"] > obs) & (g["created_at"] <= obs + horizon)]

        # right-censoring: incomplete future window -> unobservable label
        if obs + horizon > max_date:
            censored += 1
            continue

        tid = a_typeid.get(aid, "")
        pdte = a_purchase.get(aid, pd.NaT)
        wexp = a_warranty.get(aid, pd.NaT)
        cyc = cyc_map.get(tid, np.nan)
        life = life_map.get(tid, np.nan)
        age_days = (obs - pdte).days if pd.notna(pdte) else np.nan

        if not prior.empty:
            last = prior.iloc[-1]
            days_since = (obs - prior["created_at"].max()).days
            prev_issue = _s(last.get("issue_type_id")) or np.nan
            prev_prio = _s(last.get("priority")) or np.nan
            avg_res = prior["res_hours"].mean()
            prev_cost = prior["closure_cost"].sum(min_count=1)
        else:
            days_since = np.nan
            prev_issue = np.nan
            prev_prio = np.nan
            avg_res = np.nan
            prev_cost = 0.0

        # service overdue as-of obs (needs purchase_date + cycle)
        if pd.notna(pdte) and pd.notna(cyc):
            overdue = int((pdte + pd.DateOffset(months=int(cyc))) <= obs)
        else:
            overdue = np.nan
        warranty_active = int(wexp >= obs) if pd.notna(wexp) else np.nan
        repeat_fail = int((prior["issue_type_id"] == tk["issue_type_id"]).sum()) if (not prior.empty and _s(tk["issue_type_id"])) else 0

        rows.append({
            "asset_id": aid,
            "observation_date": obs,
            "asset_age_days": age_days,
            "purchase_age_months": round(age_days / 30.44, 1) if pd.notna(pdte) else np.nan,
            "expected_life_months": life,
            "maintenance_cycle_months": cyc,
            "days_since_last_ticket": days_since,
            "previous_ticket_count": int(len(prior)),
            "previous_30_day_ticket_count": int((prior["created_at"] >= obs - pd.Timedelta(days=30)).sum()) if not prior.empty else 0,
            "previous_90_day_ticket_count": int((prior["created_at"] >= obs - pd.Timedelta(days=90)).sum()) if not prior.empty else 0,
            "previous_total_maintenance_cost": float(prev_cost) if pd.notna(prev_cost) else 0.0,
            "previous_average_resolution_hours": round(float(avg_res), 1) if pd.notna(avg_res) else np.nan,
            "previous_issue_type": prev_issue,
            "previous_priority": prev_prio,
            "service_overdue": overdue,
            "warranty_active": warranty_active,
            "repeat_failure_count": repeat_fail,
            "maintenance_next_30_days": int(len(future) > 0),
        })

    dataset = pd.DataFrame(rows, columns=FEATURE_COLUMNS + ["maintenance_next_30_days"])
    report = _validate(dataset, tickets_total=len(tickets), asset_id_rows=len(t), censored=censored, horizon=horizon_days)
    return {"dataset": dataset, "report": report}


def _validate(dataset: pd.DataFrame, tickets_total: int, asset_id_rows: int, censored: int, horizon: int) -> pd.DataFrame:
    n = len(dataset)
    pos = int(dataset["maintenance_next_30_days"].sum()) if n else 0
    neg = n - pos
    rows = [
        ("horizon_days", horizon),
        ("tickets_total", tickets_total),
        ("tickets_with_asset_id", asset_id_rows),
        ("observations_excluded_right_censored", censored),
        ("total_observations", n),
        ("positive_labels", pos),
        ("negative_labels", neg),
        ("positive_rate_pct", round(100 * pos / n, 2) if n else 0.0),
        ("class_balance_neg_per_pos", round(neg / pos, 2) if pos else None),
        ("assets_represented", int(dataset["asset_id"].nunique()) if n else 0),
        ("date_range_start", str(dataset["observation_date"].min()) if n else ""),
        ("date_range_end", str(dataset["observation_date"].max()) if n else ""),
    ]
    for col in FEATURE_COLUMNS:
        if col in ("asset_id", "observation_date"):
            continue
        miss = int(dataset[col].isna().sum()) if n else 0
        rows.append((f"missing__{col}", f"{miss} ({round(100*miss/n,1) if n else 0}%)"))
    return pd.DataFrame(rows, columns=["metric", "value"])


def save(loader, horizon_days: int = _HORIZON_DAYS) -> Dict[str, Path]:
    out = build_prediction_dataset(loader, horizon_days)
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out["dataset"].to_csv(_DATASET_CSV, index=False)
    out["report"].to_csv(_REPORT_CSV, index=False)
    logger.info("Saved dataset (%d obs) -> %s ; report -> %s", len(out["dataset"]), _DATASET_CSV, _REPORT_CSV)
    return {"dataset": _DATASET_CSV, "report": _REPORT_CSV}


if __name__ == "__main__":
    import logging as _l
    from data_loader import DataLoader

    _l.basicConfig(level=_l.WARNING, format="%(levelname)s %(message)s")
    L = DataLoader()
    out = build_prediction_dataset(L)
    ds, rep = out["dataset"], out["report"]
    print("=== VALIDATION REPORT ===")
    print(rep.to_string(index=False))
    print("\n=== dataset head ===")
    print(ds.head(6).to_string(index=False))
    paths = save(L)
    print(f"\nSaved -> {paths['dataset']}\n         {paths['report']}")
