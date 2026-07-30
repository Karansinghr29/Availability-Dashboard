"""
maintenance_features.py
=======================

PHASE 3 — Predictive Maintenance: FEATURE ENGINEERING ONLY.

Builds a clean, one-row-per-asset feature table from the integrated maintenance
data. NO model is trained here and NO scoring is applied — this module only
produces the feature matrix that a later rule-based scorer AND/OR an ML model
will consume. It is intentionally split from any prediction logic so both can
reuse the same features.

Pure transform layer: reads only through ``DataLoader`` + the ``maintenance``
model. Imports nothing from the availability pages and is imported by none of
them, so existing functionality is untouched.

Features per asset
------------------
Identity / location : asset_id, asset_code, asset_type, apartment_code, bed_code,
                      condition, status
Age                 : purchase_date, asset_age_days, asset_age_years,
                      purchase_age_months, has_purchase_date
Warranty            : warranty_expiry, warranty_active
Maintenance cycle   : maintenance_cycle_months, expected_life_months,
                      replacement_cost_estimate, service_due_date, service_overdue
Vendor performance  : supplier_id, supplier_name, supplier_rating
Ticket history      : ticket_count, maintenance_frequency_per_year, repeat_job_count,
                      total_maintenance_cost, avg_maintenance_cost,
                      avg_resolution_hours, sla_breach_count,
                      last_maintenance_date, days_since_last_maintenance
Provenance          : ticket_link_level (bed / apartment / none)

Tickets carry no asset_id — they link to assets only through the room
(apartment/bed) via the maintenance cost-lines. That link is sparse, so
ticket-derived features are 0/NaN for most assets by design; ``feature_coverage``
reports exactly how many assets have real ticket history.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

import maintenance as M

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _s(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    t = str(v).strip()
    return "" if t.lower() in ("", "null", "nan", "none", "<na>") else t


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _naive_dt(series) -> pd.Series:
    d = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return d.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return pd.to_datetime(series, errors="coerce")


# --------------------------------------------------------------------------- #
# per-ticket enrichment (built once, reused per asset)
# --------------------------------------------------------------------------- #
def _ticket_table(loader, model: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """One row per ticket with room keys + cost/resolution/repeat/SLA features."""
    life = model["ticket_lifecycle"]
    costs = model["maintenance_costs"]
    cl = loader.maintenance_cost_lines()
    est = loader.ticket_cost_estimates()
    issue = loader.issue_types()

    if life.empty:
        return pd.DataFrame(columns=[
            "ticket_id", "apartment_id", "bed_id", "maintenance_type",
            "opened_at", "resolved_at", "resolution_hours", "total_cost",
            "is_repeat", "sla_hours", "sla_breach",
        ])

    t = life[["ticket_id", "opened_at", "resolved_at", "resolution_hours"]].copy()
    t["opened_at"] = _naive_dt(t["opened_at"])
    t["resolved_at"] = _naive_dt(t["resolved_at"])

    # room + maintenance_type from cost-lines (the only asset/room bridge)
    if not cl.empty and "ticket_id" in cl.columns:
        room = (cl.assign(_t=cl["ticket_id"].map(_s))
                .groupby("_t")
                .agg(apartment_id=("apartment_id", lambda s: _s(s.iloc[0])),
                     bed_id=("bed_id", lambda s: _s(s.iloc[0])),
                     maintenance_type=("maintenance_type", lambda s: _s(s.iloc[0]))))
        t = t.merge(room, left_on=t["ticket_id"].map(_s), right_index=True, how="left")
    for c in ("apartment_id", "bed_id", "maintenance_type"):
        if c not in t.columns:
            t[c] = ""

    # cost per ticket
    if not costs.empty and "total_cost" in costs.columns:
        cmap = dict(zip(costs["ticket_id"].map(_s), _num(costs["total_cost"])))
        t["total_cost"] = t["ticket_id"].map(lambda x: cmap.get(_s(x)))
    else:
        t["total_cost"] = np.nan

    # repeat-job flag from cost estimates
    if not est.empty and "ticket_id" in est.columns and "repeat_job_alert" in est.columns:
        rep = est.assign(_t=est["ticket_id"].map(_s))
        rep["_r"] = rep["repeat_job_alert"].astype(str).str.strip().str.lower().isin({"true", "t", "1", "yes"})
        rmap = rep.groupby("_t")["_r"].any().to_dict()
        t["is_repeat"] = t["ticket_id"].map(lambda x: bool(rmap.get(_s(x), False)))
    else:
        t["is_repeat"] = False

    # SLA hours by issue type (map maintenance_type -> issue_types.name), breach test
    sla_map = {}
    if not issue.empty and {"name", "sla_hours"} <= set(issue.columns):
        for n, h in zip(issue["name"], _num(issue["sla_hours"])):
            if _s(n):
                sla_map[_s(n).lower()] = h

    def _sla(mt):
        m = _s(mt).lower()
        if not m:
            return np.nan
        if m in sla_map:
            return sla_map[m]
        for k, v in sla_map.items():          # loose contains match (AC Issues ~ AC)
            if k in m or m in k:
                return v
        return np.nan

    t["sla_hours"] = t["maintenance_type"].map(_sla)
    t["sla_breach"] = (_num(t["resolution_hours"]) > _num(t["sla_hours"]))
    t.drop(columns=[c for c in t.columns if c == "key_0"], errors="ignore", inplace=True)
    return t


# --------------------------------------------------------------------------- #
# per-asset feature table
# --------------------------------------------------------------------------- #
def build_asset_features(loader, as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Return one feature row per asset. No scoring, no model — features only."""
    model = M.build_maintenance_model(loader)
    assets = model["assets"]
    if assets.empty:
        return assets
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()

    # asset-type attributes (expected life + replacement cost)
    types = loader.asset_types()
    life_map = dict(zip(types.get("id", pd.Series(dtype=str)).map(_s), _num(types.get("expected_life_months")))) if not types.empty else {}
    repl_map = dict(zip(types.get("id", pd.Series(dtype=str)).map(_s), _num(types.get("replacement_cost_estimate")))) if not types.empty else {}

    tickets = _ticket_table(loader, model)
    # index tickets by room key for fast per-asset lookup
    by_bed: Dict[str, pd.DataFrame] = {k: g for k, g in tickets.groupby(tickets["bed_id"].map(_s))} if not tickets.empty else {}
    by_apt: Dict[str, pd.DataFrame] = {k: g for k, g in tickets.groupby(tickets["apartment_id"].map(_s))} if not tickets.empty else {}

    rows = []
    for _, a in assets.iterrows():
        aid = _s(a.get("asset_id"))
        bed = _s(a.get("bed_id"))
        apt = _s(a.get("apartment_id"))
        pdate = pd.to_datetime(a.get("purchase_date"), errors="coerce")
        age_days = (as_of - pdate).days if pd.notna(pdate) else np.nan
        age_years = round(age_days / 365.25, 2) if pd.notna(pdate) else np.nan
        cyc = _num(pd.Series([a.get("maintenance_cycle_months")])).iloc[0]
        tid = _s(a.get("asset_type_id"))

        # linked tickets: bed match (specific) else apartment match.
        link = "none"
        lt = pd.DataFrame()
        if bed and bed in by_bed and not by_bed[bed].empty:
            lt, link = by_bed[bed], "bed"
        elif apt and apt in by_apt and not by_apt[apt].empty:
            lt, link = by_apt[apt], "apartment"

        ticket_count = int(len(lt))
        repeat_count = int(lt["is_repeat"].sum()) if ticket_count else 0
        total_cost = float(_num(lt["total_cost"]).sum()) if ticket_count else 0.0
        avg_cost = round(total_cost / ticket_count, 2) if ticket_count else np.nan
        res = _num(lt["resolution_hours"]).dropna() if ticket_count else pd.Series(dtype=float)
        avg_res = round(float(res.mean()), 1) if not res.empty else np.nan
        sla_breaches = int(lt["sla_breach"].sum()) if ticket_count else 0
        last_dt = lt["opened_at"].max() if ticket_count else pd.NaT
        days_since = (as_of - pd.Timestamp(last_dt).normalize()).days if pd.notna(last_dt) else np.nan
        freq = round(ticket_count / age_years, 3) if (pd.notna(age_years) and age_years and age_years > 0) else np.nan

        rows.append({
            # identity / location
            "asset_id": aid,
            "asset_code": _s(a.get("asset_code")),
            "asset_type": _s(a.get("asset_type")),
            "apartment_code": _s(a.get("apartment_code")),
            "bed_code": _s(a.get("bed_code")),
            "condition": _s(a.get("condition")),
            "status": _s(a.get("status")),
            # age / purchase age
            "purchase_date": pdate,
            "has_purchase_date": bool(pd.notna(pdate)),
            "asset_age_days": age_days,
            "asset_age_years": age_years,
            "purchase_age_months": round(age_days / 30.44, 1) if pd.notna(pdate) else np.nan,
            # warranty
            "warranty_expiry": pd.to_datetime(a.get("warranty_expiry"), errors="coerce"),
            "warranty_active": a.get("warranty_active"),
            # maintenance cycle / life
            "maintenance_cycle_months": cyc,
            "expected_life_months": life_map.get(tid, np.nan),
            "replacement_cost_estimate": repl_map.get(tid, np.nan),
            "service_due_date": pd.to_datetime(a.get("service_due_date"), errors="coerce"),
            "service_overdue": bool(pd.notna(a.get("service_due_date")) and pd.to_datetime(a.get("service_due_date"), errors="coerce") <= as_of),
            # vendor performance
            "supplier_id": _s(a.get("supplier_id")),
            "supplier_name": _s(a.get("supplier_name")),
            "supplier_rating": _num(pd.Series([a.get("supplier_rating") if "supplier_rating" in a.index else np.nan])).iloc[0] if "supplier_rating" in a.index else np.nan,
            # ticket history (sparse — room-linked)
            "ticket_count": ticket_count,
            "maintenance_frequency_per_year": freq,
            "repeat_job_count": repeat_count,
            "total_maintenance_cost": total_cost,
            "avg_maintenance_cost": avg_cost,
            "avg_resolution_hours": avg_res,
            "sla_breach_count": sla_breaches,
            "last_maintenance_date": last_dt,
            "days_since_last_maintenance": days_since,
            "ticket_link_level": link,
        })

    feats = pd.DataFrame(rows)
    # supplier rating from vendor master (assets lacked it) — enrich once.
    vend = loader.vendors()
    if not vend.empty and "id" in vend.columns and "vendor_rating" in vend.columns:
        rmap = dict(zip(vend["id"].map(_s), _num(vend["vendor_rating"])))
        feats["supplier_rating"] = feats["supplier_id"].map(lambda x: rmap.get(_s(x), np.nan))
    return feats


# --------------------------------------------------------------------------- #
# coverage report (honesty about feature sparsity)
# --------------------------------------------------------------------------- #
def feature_coverage(features: pd.DataFrame) -> dict:
    if features is None or features.empty:
        return {"assets": 0}
    n = len(features)

    def pct(mask):
        return round(100 * int(mask.sum()) / n, 1)

    return {
        "assets": int(n),
        "with_purchase_date": pct(features["has_purchase_date"]),
        "with_warranty_active_known": pct(features["warranty_active"].notna()),
        "with_maintenance_cycle": pct(features["maintenance_cycle_months"].notna()),
        "with_any_ticket_link": pct(features["ticket_count"] > 0),
        "linked_by_bed": pct(features["ticket_link_level"] == "bed"),
        "linked_by_apartment": pct(features["ticket_link_level"] == "apartment"),
        "with_supplier_rating": pct(features["supplier_rating"].notna()),
        "with_resolution_time": pct(features["avg_resolution_hours"].notna()),
    }


def save_asset_features(loader, path: Optional[Path] = None) -> Path:
    feats = build_asset_features(loader)
    out = Path(path) if path else (_OUTPUT_DIR / "maintenance_asset_features.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out, index=False)
    logger.info("Saved asset feature table -> %s (%d assets, %d features)", out, len(feats), feats.shape[1])
    return out


if __name__ == "__main__":
    import logging as _l
    from data_loader import DataLoader

    _l.basicConfig(level=_l.WARNING, format="%(levelname)s %(message)s")
    L = DataLoader()
    feats = build_asset_features(L)
    print(f"data_dir: {L.data_dir}")
    print(f"\nAsset feature table: {feats.shape[0]} assets x {feats.shape[1]} features\n")
    print("columns:", list(feats.columns))
    print("\n=== coverage ===")
    for k, v in feature_coverage(feats).items():
        print(f"  {k}: {v}")
    p = save_asset_features(L)
    print(f"\nSaved -> {p}")
    print("\n=== sample (assets with ticket history) ===")
    sample = feats[feats["ticket_count"] > 0].head(8)
    cols = ["asset_code", "asset_type", "apartment_code", "asset_age_years",
            "ticket_count", "repeat_job_count", "total_maintenance_cost",
            "avg_resolution_hours", "sla_breach_count", "ticket_link_level"]
    print(sample[[c for c in cols if c in sample.columns]].to_string(index=False))
