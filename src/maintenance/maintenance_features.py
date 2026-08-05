"""Phase 2 — Multi-grain maintenance feature engineering (no ML).

Consumes the Phase-1 ``outputs/ticket_asset_mapping.csv`` plus the raw tables
via ``DataLoader`` and produces four feature tables at different grains, then a
validation report. NO model is fit here; NO dashboard page is modified.

Grains
------
A) Asset-level    — Verified + High confidence mappings ONLY (reliable set).
B) Ticket-level   — all 1505 tickets.
C) Tenant-level   — grouped by tenant_id.
D) Technician-lvl — grouped by ticket_resolutions.resolved_by.

Data reality (drives coverage — see validation report)
------------------------------------------------------
* Cost lives in ``ticket_resolutions.total_cost``; only ~14 tickets have a
  cost > 0, so cost-derived columns are mostly empty by construction.
* ``maintenance_items.default_cost`` covers ~11 issue types, so ``expected_cost``
  and its gaps are sparse.
* ``resolved_by`` has ~4 distinct technicians; ``vendor_id`` is absent.
Coverage is reported, never hidden.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]        # project root (Availability_AI)
_OUT = _ROOT / "outputs"
_DM = 30.44                                          # days per month
_RELIABLE = {"Verified", "High"}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _s(v) -> str:
    t = "" if v is None else str(v).strip()
    return "" if t.lower() in ("", "null", "nan", "none", "<na>", "nat") else t


def _dt(series) -> pd.Series:
    d = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return d.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return pd.to_datetime(series, errors="coerce")


def _num(series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _now() -> pd.Timestamp:
    return pd.Timestamp.now().normalize()


# --------------------------------------------------------------------------- #
# input assembly: one enriched per-ticket frame that every grain reuses
# --------------------------------------------------------------------------- #
def _load_mapping(loader) -> pd.DataFrame:
    """Phase-1 mapping: prefer the committed CSV, else rebuild it."""
    csv = _OUT / "ticket_asset_mapping.csv"
    if csv.exists():
        m = pd.read_csv(csv, dtype=str)
    else:
        from ticket_asset_mapping import build_ticket_asset_mapping
        m = build_ticket_asset_mapping(loader).astype(str)
    for c in ("ticket_id", "mapped_asset_id", "mapping_confidence"):
        if c in m:
            m[c] = m[c].map(_s)
    return m


def _expected_cost_by_issue(loader) -> Dict[str, float]:
    """issue_type_id -> expected cost (median of positive default_cost)."""
    mi = loader.maintenance_items()
    if mi.empty or "issue_type_id" not in mi.columns:
        return {}
    mi = mi.copy()
    mi["_it"] = mi["issue_type_id"].map(_s)
    mi["_dc"] = _num(mi.get("default_cost"))
    pos = mi[(mi["_it"] != "") & (mi["_dc"] > 0)]
    return pos.groupby("_it")["_dc"].median().to_dict()


def _resolution_cost(loader) -> pd.DataFrame:
    """Per-ticket cost + technician from ticket_resolutions (ticket_id key)."""
    r = loader.ticket_resolutions()
    if r.empty:
        return pd.DataFrame(columns=["ticket_id", "cost", "resolved_by"])
    r = r.copy()
    r["ticket_id"] = r["ticket_id"].map(_s)
    r["cost"] = _num(r.get("total_cost"))
    r["resolved_by"] = r.get("resolved_by").map(_s)
    # a ticket could have >1 resolution row: sum cost, keep last resolver
    agg = r.groupby("ticket_id").agg(
        cost=("cost", "sum"),
        resolved_by=("resolved_by", "last"),
    ).reset_index()
    return agg


def build_enriched_tickets(loader) -> pd.DataFrame:
    """The shared per-ticket spine feeding all four grains."""
    t = loader.maintenance_tickets().copy()
    mapping = _load_mapping(loader)
    exp = _expected_cost_by_issue(loader)
    cost = _resolution_cost(loader)

    df = pd.DataFrame({
        "ticket_id": t["id"].map(_s),
        "tenant_id": t["tenant_id"].map(_s),
        "bed_id": t["bed_id"].map(_s),
        "issue_type_id": t["issue_type_id"].map(_s),
        "created_at": _dt(t.get("created_at")),
        "resolved_at": _dt(t.get("resolved_at")),
        "tenant_approved": t["tenant_approved"].map(_s).str.lower(),
        "rejection_reason": t.get("tenant_rejection_reason").map(_s),
    })

    # resolution hours (clip physically-impossible negatives to NaN)
    dh = (df["resolved_at"] - df["created_at"]).dt.total_seconds() / 3600.0
    df["resolution_hours"] = dh.where(dh >= 0)

    # mapping join (asset id + confidence + issue label + asset type)
    mcols = mapping[["ticket_id", "mapped_asset_id", "mapping_confidence",
                     "issue_type", "mapped_asset_type"]].copy()
    df = df.merge(mcols, on="ticket_id", how="left")
    df["mapping_confidence"] = df["mapping_confidence"].fillna("Unmapped")
    df["mapped_asset_id"] = df["mapped_asset_id"].fillna("")

    # cost + technician
    df = df.merge(cost, on="ticket_id", how="left")
    df["has_cost"] = df["cost"].notna() & (df["cost"] > 0)

    # expected cost + difference
    df["expected_cost"] = df["issue_type_id"].map(exp)
    df["cost_difference"] = np.where(
        df["has_cost"] & df["expected_cost"].notna(),
        df["cost"] - df["expected_cost"], np.nan,
    )

    # rejection flag
    df["is_rejected"] = df["tenant_approved"].eq("false")

    # repeat flag: same (bed, issue_type) seen more than once across all tickets
    key = df["bed_id"] + "|" + df["issue_type_id"]
    df["repeat_issue_flag"] = (key.map(key.value_counts()) > 1) & (df["bed_id"] != "")

    # tenant volume (broadcast)
    df["tenant_ticket_count"] = df["tenant_id"].map(df["tenant_id"].value_counts())
    return df


# --------------------------------------------------------------------------- #
# A) asset-level (Verified + High only)
# --------------------------------------------------------------------------- #
def asset_features(df: pd.DataFrame, loader) -> pd.DataFrame:
    rel = df[df["mapping_confidence"].isin(_RELIABLE) & (df["mapped_asset_id"] != "")].copy()
    if rel.empty:
        return pd.DataFrame()

    # purchase date from asset_master (authoritative)
    am = loader.asset_master()
    pur = dict(zip(am["id"].map(_s), _dt(am.get("purchase_date"))))
    now = _now()

    out = []
    for aid, g in rel.groupby("mapped_asset_id"):
        issue_counts = g["issue_type"].replace("", np.nan).dropna().value_counts()
        first, last = g["created_at"].min(), g["created_at"].max()
        span_m = max((last - first).days / _DM, 1.0) if pd.notna(first) and pd.notna(last) else 1.0
        costed = g[g["has_cost"]]
        pd_ = pur.get(aid, pd.NaT)
        conf = g["mapping_confidence"].value_counts()
        out.append({
            "asset_id": aid,
            "asset_type": g["mapped_asset_type"].dropna().replace("", np.nan).dropna().iloc[0]
            if g["mapped_asset_type"].replace("", np.nan).notna().any() else "",
            "purchase_date": pd_,
            "asset_age_days": (now - pd_.normalize()).days if pd.notna(pd_) else np.nan,
            "total_tickets": len(g),
            "distinct_issue_types": int(issue_counts.shape[0]),
            "repeat_issue_count": int((issue_counts - 1).clip(lower=0).sum()),
            "max_same_issue_repeats": int(issue_counts.max()) if not issue_counts.empty else 0,
            "avg_resolution_hours": round(g["resolution_hours"].mean(), 2),
            "median_resolution_hours": round(g["resolution_hours"].median(), 2),
            "costed_ticket_count": int(len(costed)),
            "total_cost": round(costed["cost"].sum(), 2) if len(costed) else 0.0,
            "avg_cost": round(costed["cost"].mean(), 2) if len(costed) else np.nan,
            "expected_cost_gap": round(costed["cost_difference"].sum(), 2)
            if costed["cost_difference"].notna().any() else np.nan,
            "last_ticket_date": last,
            "tickets_per_month": round(len(g) / span_m, 3),
            "mapping_confidence_summary": "|".join(
                f"{k}:{int(v)}" for k, v in conf.items()),
        })
    return pd.DataFrame(out).sort_values("total_tickets", ascending=False)


# --------------------------------------------------------------------------- #
# B) ticket-level (all tickets)
# --------------------------------------------------------------------------- #
def ticket_features(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "ticket_id": df["ticket_id"],
        "tenant_id": df["tenant_id"],
        "bed_id": df["bed_id"],
        "issue_type": df["issue_type"].fillna(""),
        "resolution_hours": df["resolution_hours"].round(2),
        "cost": df["cost"].where(df["has_cost"]),
        "expected_cost": df["expected_cost"],
        "cost_difference": df["cost_difference"],
        "repeat_issue_flag": df["repeat_issue_flag"].astype(int),
        "tenant_ticket_count": df["tenant_ticket_count"],
        "asset_mapping_confidence": df["mapping_confidence"],
    })


# --------------------------------------------------------------------------- #
# C) tenant-level
# --------------------------------------------------------------------------- #
def tenant_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["tenant_id"] != ""]
    out = []
    for tid, g in d.groupby("tenant_id"):
        costed = g[g["has_cost"]]
        n = len(g)
        rej = int(g["is_rejected"].sum())
        out.append({
            "tenant_id": tid,
            "total_tickets": n,
            "avg_resolution_hours": round(g["resolution_hours"].mean(), 2),
            "rejection_count": rej,
            "rejection_rate": round(rej / n, 3) if n else 0.0,
            "total_maintenance_cost": round(costed["cost"].sum(), 2),
            "cost_per_ticket": round(costed["cost"].sum() / n, 2) if n else 0.0,
        })
    return pd.DataFrame(out).sort_values("total_tickets", ascending=False)


# --------------------------------------------------------------------------- #
# D) technician-level (resolved_by)
# --------------------------------------------------------------------------- #
def technician_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df["resolved_by"].fillna("") != ""]
    if d.empty:
        return pd.DataFrame()
    out = []
    for tech, g in d.groupby("resolved_by"):
        costed = g[g["has_cost"]]
        out.append({
            "technician_id": tech,
            "tickets_closed": len(g),
            "avg_resolution_time": round(g["resolution_hours"].mean(), 2),
            "average_cost": round(costed["cost"].mean(), 2) if len(costed) else np.nan,
            "rejected_count": int(g["is_rejected"].sum()),
        })
    return pd.DataFrame(out).sort_values("tickets_closed", ascending=False)


# --------------------------------------------------------------------------- #
# validation report
# --------------------------------------------------------------------------- #
def _validate(tables: Dict[str, pd.DataFrame], enriched: pd.DataFrame) -> pd.DataFrame:
    recs = []

    def add(metric, value):
        recs.append({"metric": metric, "value": value})

    for name, tbl in tables.items():
        add(f"rows.{name}", len(tbl))
    # cost coverage
    add("cost.tickets_with_resolution", int(enriched["cost"].notna().sum()))
    add("cost.tickets_cost_gt0", int(enriched["has_cost"].sum()))
    add("cost.pct_cost_gt0", round(100 * enriched["has_cost"].mean(), 2))
    add("cost.expected_cost_available", int(enriched["expected_cost"].notna().sum()))
    add("cost.cost_diff_computable", int(enriched["cost_difference"].notna().sum()))
    # resolution coverage
    add("time.resolution_hours_available", int(enriched["resolution_hours"].notna().sum()))
    # missing-value pct per column, per table
    for name, tbl in tables.items():
        if tbl.empty:
            continue
        for col in tbl.columns:
            miss = tbl[col].isna().mean()
            if miss > 0:
                add(f"missing_pct.{name}.{col}", round(100 * miss, 2))
    # distributions on key numeric features
    rh = enriched["resolution_hours"].dropna()
    for q, lbl in [(.5, "p50"), (.9, "p90"), (.99, "p99")]:
        add(f"dist.resolution_hours.{lbl}", round(rh.quantile(q), 2) if len(rh) else np.nan)
    add("dist.resolution_hours.max", round(rh.max(), 2) if len(rh) else np.nan)
    cc = enriched.loc[enriched["has_cost"], "cost"]
    for q, lbl in [(.5, "p50"), (.9, "p90"), (1.0, "max")]:
        add(f"dist.cost_gt0.{lbl}", round(cc.quantile(q), 2) if len(cc) else np.nan)
    if "total_tickets" in tables["asset"].columns and not tables["asset"].empty:
        at = tables["asset"]["total_tickets"]
        add("dist.asset_tickets.max", int(at.max()))
        add("dist.asset_tickets.mean", round(at.mean(), 2))
    if not tables["tenant"].empty:
        add("dist.tenant_tickets.max", int(tables["tenant"]["total_tickets"].max()))
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_all(loader=None, out_dir: Optional[str] = None
              ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    if loader is None:
        from data_loader import DataLoader
        loader = DataLoader()

    enriched = build_enriched_tickets(loader)
    tables = {
        "asset": asset_features(enriched, loader),
        "ticket": ticket_features(enriched),
        "tenant": tenant_features(enriched),
        "technician": technician_features(enriched),
    }
    report = _validate(tables, enriched)

    base = Path(out_dir) if out_dir else _OUT
    base.mkdir(parents=True, exist_ok=True)
    paths = {
        "asset": base / "maintenance_asset_features.csv",
        "ticket": base / "maintenance_ticket_features.csv",
        "tenant": base / "maintenance_tenant_features.csv",
        "technician": base / "maintenance_technician_features.csv",
    }
    for k, tbl in tables.items():
        tbl.to_csv(paths[k], index=False)
    report.to_csv(base / "maintenance_features_validation.csv", index=False)
    return tables, report


def _print_summary(tables: Dict[str, pd.DataFrame], report: pd.DataFrame) -> None:
    print("=" * 60)
    print("PHASE 2 — MAINTENANCE FEATURE TABLES (validation)")
    print("=" * 60)
    for k, tbl in tables.items():
        print(f"  {k:<11} : {len(tbl):>5} rows x {tbl.shape[1]} cols")
    print("-" * 60)
    for _, r in report.iterrows():
        if str(r["metric"]).startswith(("cost.", "time.", "dist.")):
            print(f"  {r['metric']:<34} {r['value']}")
    print("=" * 60)


def main() -> None:
    tables, report = build_all()
    _print_summary(tables, report)
    print(f"\nWrote 4 feature CSVs + maintenance_features_validation.csv -> {_OUT}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # expose src/
    main()
