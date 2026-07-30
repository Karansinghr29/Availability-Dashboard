"""
maintenance.py
==============

Unified MAINTENANCE / ASSET / TICKET / VENDOR data model for Availability_AI.

This module is a pure TRANSFORM layer (like ``preprocessing`` for rooms). It
reads ONLY through ``DataLoader`` accessors — never opens a file — and produces
clean processed DataFrames plus a relationship audit. It adds NO business rules
to the existing availability pipeline and is imported by nothing in the current
dashboard, so every existing page keeps working unchanged.

Source tables (signature-detected by DataLoader, no filenames hardcoded):
    asset_master, asset_types, asset_allocations, maintenance_items,
    maintenance_item_purchases, vendors, issue_types, ticket_logs,
    ticket_resolutions, ticket_cost_estimates, ticket_purchases,
    maintenance_cost_lines  (+ apartment_master / beds_master_uuid for text keys)

Processed DataFrames (build_maintenance_model):
    vendors, asset_types, assets, asset_allocations, ticket_lifecycle,
    maintenance_tickets, maintenance_costs, purchases

Ticket lifecycle is RECONSTRUCTED from ticket_logs + ticket_resolutions because
the ticket header table is not part of the export.

NO synthetic data is generated — only real database exports are used.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"

# Log statuses that mean the ticket is finished.
_CLOSED_STATUSES = {"closed", "resolved", "completed", "done"}


# --------------------------------------------------------------------------- #
# Small helpers (no business logic)
# --------------------------------------------------------------------------- #
def _s(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    t = str(v).strip()
    return "" if t.lower() in ("", "null", "nan", "none", "<na>") else t


def _id_map(df: pd.DataFrame, key: str, value: str) -> Dict[str, str]:
    """{key -> value} from a lookup frame, blanks dropped, last wins."""
    if df is None or df.empty or key not in df.columns or value not in df.columns:
        return {}
    out: Dict[str, str] = {}
    for k, v in zip(df[key], df[value]):
        ks = _s(k)
        if ks:
            out[ks] = v
    return out


def _dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


# --------------------------------------------------------------------------- #
# Processed DataFrames
# --------------------------------------------------------------------------- #
def build_vendors(loader) -> pd.DataFrame:
    """Clean vendor master (identity + rating + status)."""
    v = loader.vendors().copy()
    if v.empty:
        return v
    keep = [c for c in [
        "id", "vendor_name", "contact_person", "phone", "email",
        "status", "vendor_rating", "gst_number", "pan_number",
    ] if c in v.columns]
    out = v[keep].copy()
    if "vendor_rating" in out.columns:
        out["vendor_rating"] = pd.to_numeric(out["vendor_rating"], errors="coerce")
    return out.reset_index(drop=True)


def build_asset_types(loader) -> pd.DataFrame:
    """Asset type catalog (expected life + maintenance cycle)."""
    t = loader.asset_types().copy()
    if t.empty:
        return t
    for c in ("expected_life_months", "maintenance_cycle_months", "replacement_cost_estimate"):
        if c in t.columns:
            t[c] = pd.to_numeric(t[c], errors="coerce")
    return t.reset_index(drop=True)


def _current_allocation_map(loader) -> Dict[str, dict]:
    """asset_id -> latest {apartment_id, bed_id, allocated_date} from allocations."""
    a = loader.asset_allocations().copy()
    if a.empty or "asset_id" not in a.columns:
        return {}
    a["_ad"] = _dt(a.get("allocated_date"))
    a = a.sort_values("_ad").drop_duplicates("asset_id", keep="last")
    out = {}
    for _, r in a.iterrows():
        aid = _s(r.get("asset_id"))
        if aid:
            out[aid] = {
                "apartment_id": _s(r.get("apartment_id")),
                "bed_id": _s(r.get("bed_id")),
                "allocated_date": r.get("_ad"),
            }
    return out


def build_assets(loader, as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Asset register: asset_master ⋈ asset_types ⋈ vendors ⋈ latest allocation.

    Derived (from real fields only): ``asset_age_days`` (purchase_date),
    ``warranty_active`` (warranty_expiry), ``service_due_date`` (purchase_date +
    maintenance_cycle_months). NULL stays NULL when the source lacks the value.
    """
    a = loader.asset_master().copy()
    if a.empty:
        return a
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()

    types = loader.asset_types()
    type_name = _id_map(types, "id", "name")
    type_cycle = _id_map(types, "id", "maintenance_cycle_months")
    type_life = _id_map(types, "id", "expected_life_months")
    vendor_name = _id_map(loader.vendors(), "id", "vendor_name")
    apt_code = _id_map(loader.apartment_master(), "id", "apartment_code")
    bed_code = _id_map(loader.beds_master_uuid(), "id", "bed_code")
    alloc = _current_allocation_map(loader)

    a["purchase_date"] = _dt(a.get("purchase_date"))
    a["warranty_expiry"] = _dt(a.get("warranty_expiry"))
    rows = []
    for _, r in a.iterrows():
        aid = _s(r.get("id"))
        tid = _s(r.get("asset_type_id"))
        pdte = r.get("purchase_date")
        cyc = pd.to_numeric(type_cycle.get(tid), errors="coerce")
        life = pd.to_numeric(type_life.get(tid), errors="coerce")
        service_due = (pdte + pd.DateOffset(months=int(cyc))) if (pd.notna(pdte) and pd.notna(cyc)) else pd.NaT
        eol = (pdte + pd.DateOffset(months=int(life))) if (pd.notna(pdte) and pd.notna(life)) else pd.NaT
        wexp = r.get("warranty_expiry")
        al = alloc.get(aid, {})
        rows.append({
            "asset_id": aid,
            "asset_code": _s(r.get("asset_code")),
            "asset_type_id": tid,
            "asset_type": type_name.get(tid),
            "brand": _s(r.get("brand")),
            "model": _s(r.get("model")),
            "condition": _s(r.get("condition")),
            "status": _s(r.get("status")),
            "purchase_date": pdte,
            "purchase_price": pd.to_numeric(r.get("purchase_price"), errors="coerce"),
            "supplier_id": _s(r.get("supplier_id")),
            "supplier_name": vendor_name.get(_s(r.get("supplier_id"))),
            "warranty_expiry": wexp,
            "warranty_active": (bool(pd.notna(wexp) and wexp >= as_of) if pd.notna(wexp) else pd.NA),
            "asset_age_days": ((as_of - pdte).days if pd.notna(pdte) else pd.NA),
            "maintenance_cycle_months": cyc,
            "service_due_date": service_due,
            "expected_end_of_life": eol,
            "apartment_id": al.get("apartment_id"),
            "apartment_code": apt_code.get(al.get("apartment_id")) if al.get("apartment_id") else None,
            "bed_id": al.get("bed_id"),
            "bed_code": bed_code.get(al.get("bed_id")) if al.get("bed_id") else None,
            "allocated_date": al.get("allocated_date"),
        })
    return pd.DataFrame(rows)


def build_asset_allocations(loader) -> pd.DataFrame:
    """Asset allocation log enriched with asset_code + apartment/bed text keys."""
    a = loader.asset_allocations().copy()
    if a.empty:
        return a
    asset_code = _id_map(loader.asset_master(), "id", "asset_code")
    apt_code = _id_map(loader.apartment_master(), "id", "apartment_code")
    bed_code = _id_map(loader.beds_master_uuid(), "id", "bed_code")
    a["allocated_date"] = _dt(a.get("allocated_date"))
    a["asset_code"] = a.get("asset_id").map(lambda x: asset_code.get(_s(x)))
    a["apartment_code"] = a.get("apartment_id").map(lambda x: apt_code.get(_s(x)))
    a["bed_code"] = a.get("bed_id").map(lambda x: bed_code.get(_s(x)))
    return a.reset_index(drop=True)


def _ticket_universe(loader) -> set:
    """All distinct ticket_ids seen across every ticket-linked table."""
    ids = set()
    for acc in ("ticket_logs", "ticket_resolutions", "ticket_cost_estimates",
                "ticket_purchases", "maintenance_cost_lines"):
        df = getattr(loader, acc)()
        if not df.empty and "ticket_id" in df.columns:
            ids |= {_s(x) for x in df["ticket_id"] if _s(x)}
    return ids


def build_ticket_lifecycle(loader) -> pd.DataFrame:
    """Reconstruct per-ticket lifecycle from ticket_logs (+ ticket_resolutions).

    Header table is absent, so: ``opened_at`` = earliest log; ``current_status``
    = status of latest log; ``closed_at`` = earliest log whose new_status is a
    closed state; ``resolved_at`` / resolution cost from ticket_resolutions;
    ``resolution_hours`` = (resolved_at|closed_at − opened_at). ``status_changes``
    counts log rows (activity volume).
    """
    logs = loader.ticket_logs().copy()
    res = loader.ticket_resolutions().copy()
    universe = _ticket_universe(loader)
    if not universe:
        return pd.DataFrame(columns=[
            "ticket_id", "opened_at", "current_status", "closed_at", "resolved_at",
            "is_resolved", "resolution_hours", "status_changes", "resolution_total_cost",
        ])

    opened, current, closed, changes = {}, {}, {}, {}
    if not logs.empty and "ticket_id" in logs.columns:
        logs["_ts"] = _dt(logs.get("created_at"))
        logs["_ns"] = logs.get("new_status").map(_s)
        for tid, g in logs.groupby(logs["ticket_id"].map(_s)):
            if not tid:
                continue
            g = g.sort_values("_ts")
            opened[tid] = g["_ts"].min()
            current[tid] = g["_ns"].replace("", pd.NA).dropna().iloc[-1] if g["_ns"].replace("", pd.NA).notna().any() else pd.NA
            changes[tid] = int(len(g))
            cl = g[g["_ns"].str.lower().isin(_CLOSED_STATUSES)]
            if not cl.empty:
                closed[tid] = cl["_ts"].min()

    resolved_at, res_cost = {}, {}
    if not res.empty and "ticket_id" in res.columns:
        res["_ra"] = _dt(res.get("resolved_at"))
        res["_tc"] = pd.to_numeric(res.get("total_cost"), errors="coerce")
        for tid, g in res.groupby(res["ticket_id"].map(_s)):
            if not tid:
                continue
            resolved_at[tid] = g["_ra"].max()
            res_cost[tid] = g["_tc"].sum(min_count=1)

    rows = []
    for tid in sorted(universe):
        op = opened.get(tid, pd.NaT)
        end = resolved_at.get(tid) if pd.notna(resolved_at.get(tid, pd.NaT)) else closed.get(tid, pd.NaT)
        hrs = ((end - op).total_seconds() / 3600.0) if (pd.notna(op) and pd.notna(end)) else pd.NA
        rows.append({
            "ticket_id": tid,
            "opened_at": op,
            "current_status": current.get(tid, pd.NA),
            "closed_at": closed.get(tid, pd.NaT),
            "resolved_at": resolved_at.get(tid, pd.NaT),
            "is_resolved": bool(pd.notna(resolved_at.get(tid, pd.NaT)) or pd.notna(closed.get(tid, pd.NaT))),
            "resolution_hours": (round(hrs, 1) if hrs is not pd.NA and pd.notna(hrs) else pd.NA),
            "status_changes": changes.get(tid, 0),
            "resolution_total_cost": res_cost.get(tid, pd.NA),
        })
    return pd.DataFrame(rows)


def build_maintenance_tickets(loader) -> pd.DataFrame:
    """Per-ticket header = lifecycle ⋈ cost-line room/type ⋈ resolution.

    Room/type/tenant come from maintenance_cost_lines (100% linked to Q83/Q85/Q82).
    One row per ticket_id.
    """
    life = build_ticket_lifecycle(loader)
    if life.empty:
        return life
    cl = loader.maintenance_cost_lines().copy()
    apt_code = _id_map(loader.apartment_master(), "id", "apartment_code")
    bed_code = _id_map(loader.beds_master_uuid(), "id", "bed_code")

    line_meta: Dict[str, dict] = {}
    if not cl.empty and "ticket_id" in cl.columns:
        for tid, g in cl.groupby(cl["ticket_id"].map(_s)):
            if not tid:
                continue
            r0 = g.iloc[0]
            mtypes = sorted({_s(x) for x in g.get("maintenance_type", []) if _s(x)})
            line_meta[tid] = {
                "maintenance_type": ", ".join(mtypes) if mtypes else pd.NA,
                "apartment_id": _s(r0.get("apartment_id")),
                "bed_id": _s(r0.get("bed_id")),
                "tenant_id": _s(r0.get("tenant_id")),
                "line_cost": pd.to_numeric(g.get("actual_cost"), errors="coerce").sum(min_count=1),
            }

    def meta(tid, k):
        return line_meta.get(tid, {}).get(k, pd.NA)

    out = life.copy()
    out["maintenance_type"] = out["ticket_id"].map(lambda t: meta(t, "maintenance_type"))
    out["apartment_id"] = out["ticket_id"].map(lambda t: meta(t, "apartment_id"))
    out["bed_id"] = out["ticket_id"].map(lambda t: meta(t, "bed_id"))
    out["tenant_id"] = out["ticket_id"].map(lambda t: meta(t, "tenant_id"))
    out["apartment_code"] = out["apartment_id"].map(lambda x: apt_code.get(_s(x)) if _s(x) else None)
    out["bed_code"] = out["bed_id"].map(lambda x: bed_code.get(_s(x)) if _s(x) else None)
    out["line_cost"] = out["ticket_id"].map(lambda t: meta(t, "line_cost"))
    return out.reset_index(drop=True)


def build_maintenance_costs(loader) -> pd.DataFrame:
    """Per-ticket cost rollup from every cost source (long-safe, one row/ticket).

    Sources: maintenance_cost_lines.actual_cost, ticket_resolutions.total_cost,
    ticket_cost_estimates.total, ticket_purchases.actual_cost.
    """
    universe = sorted(_ticket_universe(loader))
    if not universe:
        return pd.DataFrame(columns=["ticket_id", "line_cost", "resolution_cost", "estimate_cost", "purchase_cost", "total_cost"])

    def per_ticket(df, col):
        if df.empty or "ticket_id" not in df.columns or col not in df.columns:
            return {}
        s = pd.to_numeric(df[col], errors="coerce")
        return df.assign(_v=s).groupby(df["ticket_id"].map(_s))["_v"].sum(min_count=1).to_dict()

    line = per_ticket(loader.maintenance_cost_lines(), "actual_cost")
    reso = per_ticket(loader.ticket_resolutions(), "total_cost")
    est = per_ticket(loader.ticket_cost_estimates(), "total")
    pur = per_ticket(loader.ticket_purchases(), "actual_cost")

    rows = []
    for tid in universe:
        lc, rc, ec, pc = line.get(tid), reso.get(tid), est.get(tid), pur.get(tid)
        vals = [x for x in (lc, rc, pc) if x is not None and pd.notna(x)]
        rows.append({
            "ticket_id": tid,
            "line_cost": lc, "resolution_cost": rc,
            "estimate_cost": ec, "purchase_cost": pc,
            # Prefer resolution total; else sum of line/purchase; else estimate.
            "total_cost": (rc if (rc is not None and pd.notna(rc)) else (sum(vals) if vals else ec)),
        })
    return pd.DataFrame(rows)


def build_purchases(loader) -> pd.DataFrame:
    """Maintenance-item purchases ⋈ item master ⋈ vendor (stock receipts)."""
    p = loader.maintenance_item_purchases().copy()
    if p.empty:
        return p
    item_name = _id_map(loader.maintenance_items(), "id", "item_name")
    p["purchased_on"] = _dt(p.get("purchased_on"))
    for c in ("quantity_received", "quantity_available", "purchase_cost_per_unit"):
        if c in p.columns:
            p[c] = pd.to_numeric(p[c], errors="coerce")
    p["item_name"] = p.get("maintenance_item_id").map(lambda x: item_name.get(_s(x)))
    if {"quantity_received", "purchase_cost_per_unit"} <= set(p.columns):
        p["total_purchase_cost"] = p["quantity_received"] * p["purchase_cost_per_unit"]
    return p.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Relationship audit
# --------------------------------------------------------------------------- #
def relationship_audit(loader) -> pd.DataFrame:
    """Validate every maintenance FK: matched / unmatched / missing / dup keys.

    Parent id-sets come from the referenced master. For ticket FKs the parent is
    the reconstructed ticket universe (header table absent) — flagged in notes.
    """
    universe = _ticket_universe(loader)

    def load(acc):
        return getattr(loader, acc)()

    # (child_acc, child_col, parent_ids_or_acc, parent_col_or_note, label)
    edges = [
        ("asset_allocations", "asset_id", ("asset_master", "id"), "assets"),
        ("asset_allocations", "apartment_id", ("apartment_master", "id"), "apartments Q85"),
        ("asset_allocations", "bed_id", ("beds_master_uuid", "id"), "beds Q83"),
        ("asset_master", "asset_type_id", ("asset_types", "id"), "asset_types"),
        ("asset_master", "supplier_id", ("vendors", "id"), "vendors"),
        ("maintenance_items", "asset_type_id", ("asset_types", "id"), "asset_types"),
        ("maintenance_items", "issue_type_id", ("issue_types", "id"), "issue_types"),
        ("maintenance_item_purchases", "maintenance_item_id", ("maintenance_items", "id"), "items"),
        ("ticket_logs", "ticket_id", ("__universe__", ""), "ticket header MISSING (reconstructed)"),
        ("ticket_resolutions", "ticket_id", ("__universe__", ""), "ticket header MISSING (reconstructed)"),
        ("ticket_cost_estimates", "ticket_id", ("__universe__", ""), "ticket header MISSING (reconstructed)"),
        ("ticket_purchases", "ticket_id", ("__universe__", ""), "ticket header MISSING (reconstructed)"),
        ("maintenance_cost_lines", "ticket_id", ("__universe__", ""), "ticket header MISSING (reconstructed)"),
        ("ticket_purchases", "cost_estimate_id", ("ticket_cost_estimates", "id"), "cost_estimates"),
        ("ticket_purchases", "vendor_id", ("vendors", "id"), "vendors"),
        ("ticket_resolutions", "vendor_id", ("vendors", "id"), "vendors"),
        ("maintenance_cost_lines", "apartment_id", ("apartment_master", "id"), "apartments Q85"),
        ("maintenance_cost_lines", "bed_id", ("beds_master_uuid", "id"), "beds Q83"),
        ("maintenance_cost_lines", "tenant_id", ("tenant_master", "id"), "tenants Q82"),
        ("asset_types", "category_id", (None, None), "asset_categories MISSING"),
    ]

    rows = []
    for child_acc, child_col, (parent_acc, parent_col), note in edges:
        child = load(child_acc)
        if child.empty or child_col not in child.columns:
            rows.append({
                "child_table": child_acc, "child_column": child_col,
                "parent_table": note, "child_rows": 0, "fk_present": 0,
                "fk_missing": 0, "matched": 0, "unmatched": 0,
                "parent_duplicate_keys": 0, "note": "child empty or column absent",
            })
            continue
        fk = child[child_col].map(_s)
        present = fk[fk != ""]
        missing = int((fk == "").sum())

        if parent_acc == "__universe__":
            parent_ids = universe
            dup = 0
        elif parent_acc is None:
            parent_ids = set()
            dup = 0
        else:
            pdf = load(parent_acc)
            parent_ids = {_s(x) for x in pdf.get(parent_col, pd.Series([], dtype=str)) if _s(x)}
            dup = int(pdf[parent_col].map(_s).replace("", pd.NA).dropna().duplicated().sum()) if (not pdf.empty and parent_col in pdf.columns) else 0

        matched = int(present.isin(parent_ids).sum()) if parent_ids else 0
        unmatched = int(len(present) - matched)
        rows.append({
            "child_table": child_acc, "child_column": child_col,
            "parent_table": note, "child_rows": int(len(child)),
            "fk_present": int(len(present)), "fk_missing": missing,
            "matched": matched, "unmatched": unmatched,
            "parent_duplicate_keys": dup,
            "note": ("parent table absent" if parent_acc is None
                     else ("reconstructed universe" if parent_acc == "__universe__" else "")),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def build_maintenance_model(loader) -> Dict[str, pd.DataFrame]:
    """Return every processed maintenance DataFrame in one dict."""
    return {
        "vendors": build_vendors(loader),
        "asset_types": build_asset_types(loader),
        "assets": build_assets(loader),
        "asset_allocations": build_asset_allocations(loader),
        "ticket_lifecycle": build_ticket_lifecycle(loader),
        "maintenance_tickets": build_maintenance_tickets(loader),
        "maintenance_costs": build_maintenance_costs(loader),
        "purchases": build_purchases(loader),
    }


def save_relationship_audit(loader, path: Optional[Path] = None) -> Path:
    """Write the FK relationship audit to outputs/maintenance_relationship_audit.csv."""
    audit = relationship_audit(loader)
    out = Path(path) if path else (_OUTPUT_DIR / "maintenance_relationship_audit.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out, index=False)
    logger.info("Saved maintenance relationship audit -> %s (%d edges)", out, len(audit))
    return out


if __name__ == "__main__":
    import logging as _l
    from data_loader import DataLoader

    _l.basicConfig(level=_l.WARNING, format="%(levelname)s %(message)s")
    L = DataLoader()
    model = build_maintenance_model(L)
    print(f"data_dir: {L.data_dir}\n")
    print("=== processed maintenance dataframes ===")
    for name, df in model.items():
        print(f"  {name:22s} rows={len(df):5d} cols={df.shape[1]}")
    p = save_relationship_audit(L)
    print(f"\nRelationship audit saved -> {p}")
    print(relationship_audit(L).to_string(index=False))
