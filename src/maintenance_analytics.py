"""
maintenance_analytics.py
========================

DESCRIPTIVE analytics for the Maintenance Intelligence dashboard page.

Pure compute layer (no Streamlit, no I/O). Consumes the processed maintenance
model (``maintenance.build_maintenance_model``) plus a few raw loader accessors,
and returns tidy DataFrames / scalar dicts the page renders directly.

NO predictive analytics here — Phase 2 is descriptive only. NO business logic in
the existing availability pipeline is touched; this module is imported only by
the new dashboard page.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

import maintenance as M


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _counts(series: pd.Series, name_col: str, count_col: str = "count") -> pd.DataFrame:
    s = series.map(lambda v: str(v).strip() if v is not None and str(v).strip() and str(v).lower() != "nan" else "Unknown")
    out = s.value_counts(dropna=False).rename_axis(name_col).reset_index(name=count_col)
    return out


def _naive_dt(series: pd.Series) -> pd.Series:
    """Parse to tz-naive datetimes (handles mixed tz-aware/naive columns)."""
    d = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return d.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return pd.to_datetime(series, errors="coerce")


def _month(series: pd.Series) -> pd.Series:
    d = _naive_dt(series)
    return d.dt.to_period("M").astype(str).replace("NaT", np.nan)


# --------------------------------------------------------------------------- #
# 1. KPI cards
# --------------------------------------------------------------------------- #
def kpi_cards(model: Dict[str, pd.DataFrame], loader) -> dict:
    tickets = model["maintenance_tickets"]
    costs = model["maintenance_costs"]
    assets = model["assets"]
    vendors = model["vendors"]
    purchases = model["purchases"]

    total_tickets = int(len(tickets))
    resolved = int(tickets["is_resolved"].sum()) if "is_resolved" in tickets and total_tickets else 0
    open_tickets = total_tickets - resolved
    res_hours = _num(tickets.get("resolution_hours")) if total_tickets else pd.Series(dtype=float)
    avg_res_h = round(float(res_hours.mean()), 1) if res_hours.notna().any() else None
    total_cost = float(_num(costs.get("total_cost")).sum()) if not costs.empty else 0.0

    active_vendors = int((vendors.get("status", pd.Series(dtype=str)).astype(str).str.lower() == "active").sum()) if not vendors.empty else 0
    if not active_vendors and not vendors.empty:
        active_vendors = int(len(vendors))

    total_purchase_value = float(_num(purchases.get("total_purchase_cost")).sum()) if not purchases.empty else 0.0
    low_stock = 0
    if not purchases.empty:
        mi = loader.maintenance_items()
        minmap = {}
        if not mi.empty and "id" in mi.columns and "minimum_stock_level" in mi.columns:
            minmap = dict(zip(mi["id"].astype(str), _num(mi["minimum_stock_level"])))
        if "maintenance_item_id" in purchases.columns:
            qa = _num(purchases.get("quantity_available"))
            for iid, avail in zip(purchases["maintenance_item_id"].astype(str), qa):
                mn = minmap.get(iid)
                if pd.notna(avail) and mn is not None and pd.notna(mn) and avail < mn:
                    low_stock += 1

    return {
        "total_tickets": total_tickets,
        "open_tickets": open_tickets,
        "resolved_tickets": resolved,
        "resolution_rate_pct": round(100 * resolved / total_tickets, 1) if total_tickets else 0.0,
        "avg_resolution_hours": avg_res_h,
        "total_maintenance_cost": total_cost,
        "total_assets": int(len(assets)),
        "assets_in_service": int((assets.get("status", pd.Series(dtype=str)).astype(str).str.lower().isin({"active", "in_use", "allocated", "in-service"})).sum()) if not assets.empty else 0,
        "active_vendors": active_vendors,
        "total_purchase_value": total_purchase_value,
        "low_stock_items": low_stock,
    }


# --------------------------------------------------------------------------- #
# 2. Ticket analytics
# --------------------------------------------------------------------------- #
def ticket_analytics(model: Dict[str, pd.DataFrame], loader) -> dict:
    t = model["maintenance_tickets"]
    out = {"by_status": pd.DataFrame(), "by_type": pd.DataFrame(),
           "over_time": pd.DataFrame(), "top_apartments": pd.DataFrame(),
           "resolution": {}}
    if t.empty:
        return out
    out["by_status"] = _counts(t.get("current_status"), "status")
    out["by_type"] = _counts(t.get("maintenance_type"), "maintenance_type").head(15)
    ot = t.assign(month=_month(t.get("opened_at")))
    out["over_time"] = (ot.dropna(subset=["month"]).groupby("month").size()
                        .rename("tickets").reset_index())
    apt = _counts(t.get("apartment_code"), "apartment_code")
    apt = apt[apt["apartment_code"] != "Unknown"].head(15)
    out["top_apartments"] = apt
    rh = _num(t.get("resolution_hours")).dropna()
    out["resolution"] = {
        "avg_hours": round(float(rh.mean()), 1) if not rh.empty else None,
        "median_hours": round(float(rh.median()), 1) if not rh.empty else None,
        "max_hours": round(float(rh.max()), 1) if not rh.empty else None,
        "resolved_with_time": int(len(rh)),
    }
    # avg resolution hours by type
    if not rh.empty:
        tt = t.assign(_rh=_num(t.get("resolution_hours")))
        by = (tt.dropna(subset=["_rh"]).groupby(tt["maintenance_type"].fillna("Unknown"))["_rh"]
              .mean().round(1).rename("avg_resolution_hours").reset_index())
        out["resolution_by_type"] = by.sort_values("avg_resolution_hours", ascending=False)
    else:
        out["resolution_by_type"] = pd.DataFrame()
    return out


# --------------------------------------------------------------------------- #
# 3. Asset analytics
# --------------------------------------------------------------------------- #
def asset_analytics(model: Dict[str, pd.DataFrame]) -> dict:
    a = model["assets"]
    out = {"by_type": pd.DataFrame(), "by_condition": pd.DataFrame(),
           "by_status": pd.DataFrame(), "by_apartment": pd.DataFrame(),
           "age_buckets": pd.DataFrame(), "service_due": pd.DataFrame(),
           "warranty": pd.DataFrame()}
    if a.empty:
        return out
    out["by_type"] = _counts(a.get("asset_type"), "asset_type").head(20)
    out["by_condition"] = _counts(a.get("condition"), "condition")
    out["by_status"] = _counts(a.get("status"), "status")
    apt = _counts(a.get("apartment_code"), "apartment_code")
    out["by_apartment"] = apt[apt["apartment_code"] != "Unknown"].head(20)
    # age buckets (years)
    age_years = _num(a.get("asset_age_days")) / 365.25
    buckets = pd.cut(age_years, [-0.01, 1, 3, 5, 100],
                     labels=["<1y", "1-3y", "3-5y", "5y+"])
    out["age_buckets"] = buckets.value_counts().rename_axis("age").reset_index(name="count")
    # warranty active vs expired/unknown
    w = a.get("warranty_active")
    wv = w.map(lambda x: "Active" if x is True else ("Expired" if x is False else "Unknown"))
    out["warranty"] = wv.value_counts().rename_axis("warranty").reset_index(name="count")
    # service due within 30 days (or overdue) — descriptive list only
    if "service_due_date" in a.columns:
        sd = pd.to_datetime(a["service_due_date"], errors="coerce")
        today = pd.Timestamp.today().normalize()
        due = a[sd.notna() & (sd <= today + pd.Timedelta(days=30))].copy()
        due = due.assign(service_due_date=sd[sd.notna() & (sd <= today + pd.Timedelta(days=30))])
        cols = [c for c in ["asset_code", "asset_type", "apartment_code", "condition", "service_due_date"] if c in due.columns]
        out["service_due"] = due[cols].sort_values("service_due_date").head(50)
    return out


# --------------------------------------------------------------------------- #
# 4. Vendor analytics
# --------------------------------------------------------------------------- #
def vendor_analytics(model: Dict[str, pd.DataFrame], loader) -> dict:
    vendors = model["vendors"]
    out = {"master": vendors, "spend": pd.DataFrame(), "tickets": pd.DataFrame(), "ratings": pd.DataFrame()}

    # Spend by vendor_name (cost-lines are name-keyed; item purchases too).
    cl = loader.maintenance_cost_lines()
    frames = []
    if not cl.empty and {"vendor_name", "actual_cost"} <= set(cl.columns):
        frames.append(cl[["vendor_name", "actual_cost"]].rename(columns={"actual_cost": "amount"}))
    pur = model["purchases"]
    if not pur.empty and {"vendor_name", "total_purchase_cost"} <= set(pur.columns):
        frames.append(pur[["vendor_name", "total_purchase_cost"]].rename(columns={"total_purchase_cost": "amount"}))
    if frames:
        sp = pd.concat(frames, ignore_index=True)
        sp["vendor_name"] = sp["vendor_name"].map(lambda v: str(v).strip() if v is not None and str(v).strip() and str(v).lower() != "nan" else "Unknown")
        sp["amount"] = _num(sp["amount"])
        out["spend"] = (sp.groupby("vendor_name")["amount"].sum().round(2)
                        .rename("total_spend").reset_index().sort_values("total_spend", ascending=False))

    # ticket count per vendor (name)
    if not cl.empty and "vendor_name" in cl.columns:
        out["tickets"] = _counts(cl["vendor_name"], "vendor_name").rename(columns={"count": "cost_lines"})

    # ratings from master
    if not vendors.empty and "vendor_rating" in vendors.columns:
        r = vendors[["vendor_name", "vendor_rating", "status"]].copy()
        r["vendor_rating"] = _num(r["vendor_rating"])
        out["ratings"] = r.sort_values("vendor_rating", ascending=False)
    return out


# --------------------------------------------------------------------------- #
# 5. Purchase & inventory analytics
# --------------------------------------------------------------------------- #
def purchase_inventory_analytics(model: Dict[str, pd.DataFrame], loader) -> dict:
    p = model["purchases"]
    out = {"by_item": pd.DataFrame(), "over_time": pd.DataFrame(),
           "low_stock": pd.DataFrame(), "by_vendor": pd.DataFrame(), "stock": pd.DataFrame()}
    if p.empty:
        return out
    p = p.copy()
    p["total_purchase_cost"] = _num(p.get("total_purchase_cost"))
    # spend + qty by item
    by_item = (p.assign(item=p["item_name"].fillna("Unknown"))
               .groupby("item")
               .agg(total_cost=("total_purchase_cost", "sum"),
                    qty_received=("quantity_received", lambda s: _num(s).sum()))
               .round(2).reset_index().sort_values("total_cost", ascending=False))
    out["by_item"] = by_item.head(20)
    # purchases over time
    ot = p.assign(month=_month(p.get("purchased_on")))
    out["over_time"] = (ot.dropna(subset=["month"]).groupby("month")["total_purchase_cost"]
                        .sum().round(2).rename("purchase_cost").reset_index())
    # by vendor
    if "vendor_name" in p.columns:
        out["by_vendor"] = (p.assign(vendor=p["vendor_name"].fillna("Unknown"))
                            .groupby("vendor")["total_purchase_cost"].sum().round(2)
                            .rename("purchase_cost").reset_index().sort_values("purchase_cost", ascending=False))
    # stock vs minimum level (low stock)
    mi = loader.maintenance_items()
    if not mi.empty and "id" in mi.columns:
        minmap = dict(zip(mi["id"].astype(str), _num(mi.get("minimum_stock_level"))))
        namemap = dict(zip(mi["id"].astype(str), mi.get("item_name")))
        stock = (p.assign(_iid=p.get("maintenance_item_id").astype(str))
                 .groupby("_iid")
                 .agg(available=("quantity_available", lambda s: _num(s).sum())).reset_index())
        stock["item_name"] = stock["_iid"].map(namemap)
        stock["minimum_stock_level"] = stock["_iid"].map(minmap)
        stock["below_minimum"] = stock["available"] < stock["minimum_stock_level"]
        out["stock"] = stock[["item_name", "available", "minimum_stock_level", "below_minimum"]]
        out["low_stock"] = stock[stock["below_minimum"]][["item_name", "available", "minimum_stock_level"]]
    return out


# --------------------------------------------------------------------------- #
# 6. Maintenance cost analytics
# --------------------------------------------------------------------------- #
def cost_analytics(model: Dict[str, pd.DataFrame], loader) -> dict:
    costs = model["maintenance_costs"]
    tickets = model["maintenance_tickets"]
    out = {"total": 0.0, "by_type": pd.DataFrame(), "by_apartment": pd.DataFrame(),
           "over_time": pd.DataFrame(), "parts_vs_labour": pd.DataFrame(),
           "per_ticket": pd.DataFrame()}
    if costs.empty:
        return out
    c = costs.merge(
        tickets[["ticket_id", "maintenance_type", "apartment_code", "opened_at"]],
        on="ticket_id", how="left",
    )
    c["total_cost"] = _num(c["total_cost"])
    out["total"] = round(float(c["total_cost"].sum()), 2)
    out["by_type"] = (c.assign(t=c["maintenance_type"].fillna("Unknown"))
                      .groupby("t")["total_cost"].sum().round(2)
                      .rename("total_cost").reset_index().sort_values("total_cost", ascending=False))
    apt = (c.assign(a=c["apartment_code"].fillna("Unknown"))
           .groupby("a")["total_cost"].sum().round(2)
           .rename("total_cost").reset_index())
    out["by_apartment"] = apt[apt["a"] != "Unknown"].sort_values("total_cost", ascending=False).head(20)
    ot = c.assign(month=_month(c["opened_at"]))
    out["over_time"] = (ot.dropna(subset=["month"]).groupby("month")["total_cost"]
                        .sum().round(2).rename("cost").reset_index())
    out["per_ticket"] = c[["ticket_id", "total_cost"]].dropna().sort_values("total_cost", ascending=False).head(20)
    # parts vs labour from resolutions
    res = loader.ticket_resolutions()
    if not res.empty:
        parts = float(_num(res.get("total_parts_cost")).sum())
        labour = float(_num(res.get("total_labour_cost")).sum())
        out["parts_vs_labour"] = pd.DataFrame(
            {"component": ["Parts", "Labour"], "cost": [round(parts, 2), round(labour, 2)]}
        )
    return out


# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
_ALL = "All"


def _distinct(series) -> list:
    if series is None:
        return []
    vals = {str(v).strip() for v in series if v is not None and str(v).strip() and str(v).lower() != "nan"}
    return sorted(vals)


def filter_options(model: Dict[str, pd.DataFrame], loader) -> dict:
    """Distinct filter values (from the FULL model) + date bounds."""
    assets = model["assets"]
    tickets = model["maintenance_tickets"]
    purchases = model["purchases"]
    props = _distinct(loader.property_master().get("property_name")) if not loader.property_master().empty else []
    apts = _distinct(loader.apartment_master().get("apartment_code"))
    issue_types = _distinct(loader.issue_types().get("name"))
    # fall back to the maintenance_type vocabulary when issue_types names don't map
    issue_types = issue_types or _distinct(tickets.get("maintenance_type"))
    vendors = _distinct(loader.vendors().get("vendor_name"))

    dates = []
    for df, col in [(tickets, "opened_at"), (assets, "purchase_date"), (purchases, "purchased_on")]:
        if not df.empty and col in df.columns:
            d = _naive_dt(df[col]).dropna()
            if not d.empty:
                dates += [d.min(), d.max()]
    dmin = min(dates) if dates else None
    dmax = max(dates) if dates else None
    return {
        "property": [_ALL] + props,
        "apartment": [_ALL] + apts,
        "asset_type": [_ALL] + _distinct(assets.get("asset_type")),
        "issue_type": [_ALL] + issue_types,
        "vendor": [_ALL] + vendors,
        "date_min": dmin,
        "date_max": dmax,
    }


def _apartments_for_property(loader, property_name: str) -> set:
    apts = loader.apartment_master()
    props = loader.property_master()
    if apts.empty or props.empty:
        return set()
    pid = {str(i) for i, n in zip(props.get("id", []), props.get("property_name", [])) if str(n).strip() == property_name}
    codes = {str(c).strip() for p, c in zip(apts.get("property_id", []), apts.get("apartment_code", [])) if str(p) in pid}
    return codes


def apply_filters(model: Dict[str, pd.DataFrame], loader, filters: Optional[dict]) -> Dict[str, pd.DataFrame]:
    """Return an in-memory filtered copy of the model. ``All``/None = no filter."""
    if not filters:
        return model
    f = {k: (None if v in (None, _ALL, "") else v) for k, v in filters.items()}
    out = {k: v.copy() for k, v in model.items()}

    apt_scope = None
    if f.get("property"):
        apt_scope = _apartments_for_property(loader, f["property"])
    if f.get("apartment"):
        apt_scope = ({f["apartment"]} if apt_scope is None else (apt_scope & {f["apartment"]}))

    d0 = pd.to_datetime(f.get("date_from")) if f.get("date_from") else None
    d1 = pd.to_datetime(f.get("date_to")) if f.get("date_to") else None

    def by_apt(df):
        return df[df["apartment_code"].astype(str).str.strip().isin(apt_scope)] if (apt_scope is not None and "apartment_code" in df.columns) else df

    def by_date(df, col):
        if col not in df.columns or (d0 is None and d1 is None):
            return df
        dd = _naive_dt(df[col]).dt.normalize()  # compare on date only (inclusive end-of-day)
        # Rows with NO date are never excluded by a date filter — otherwise the
        # default full-span range would silently drop every undated row (e.g.
        # assets missing purchase_date) and distort the KPIs / data-quality view.
        in_range = pd.Series(True, index=df.index)
        if d0 is not None:
            in_range &= dd >= pd.Timestamp(d0).normalize()
        if d1 is not None:
            in_range &= dd <= pd.Timestamp(d1).normalize()
        return df[dd.isna() | in_range]

    # tickets
    t = out["maintenance_tickets"]
    t = by_apt(t)
    if f.get("issue_type") and "maintenance_type" in t.columns:
        t = t[t["maintenance_type"].astype(str).str.contains(str(f["issue_type"]), case=False, na=False)]
    t = by_date(t, "opened_at")
    out["maintenance_tickets"] = t
    keep_tickets = set(t["ticket_id"]) if "ticket_id" in t.columns else None

    # costs follow tickets
    if keep_tickets is not None and "ticket_id" in out["maintenance_costs"].columns:
        out["maintenance_costs"] = out["maintenance_costs"][out["maintenance_costs"]["ticket_id"].isin(keep_tickets)]
    out["ticket_lifecycle"] = by_date(by_apt(out["ticket_lifecycle"]) if "apartment_code" in out["ticket_lifecycle"].columns else out["ticket_lifecycle"], "opened_at")

    # assets
    a = out["assets"]
    a = by_apt(a)
    if f.get("asset_type") and "asset_type" in a.columns:
        a = a[a["asset_type"].astype(str).str.strip() == f["asset_type"]]
    if f.get("vendor") and "supplier_name" in a.columns:
        a = a[a["supplier_name"].astype(str).str.strip() == f["vendor"]]
    a = by_date(a, "purchase_date")
    out["assets"] = a
    out["asset_allocations"] = by_apt(out["asset_allocations"])

    # purchases
    p = out["purchases"]
    if f.get("vendor") and "vendor_name" in p.columns:
        p = p[p["vendor_name"].astype(str).str.strip() == f["vendor"]]
    p = by_date(p, "purchased_on")
    out["purchases"] = p

    if f.get("vendor") and not out["vendors"].empty and "vendor_name" in out["vendors"].columns:
        out["vendors"] = out["vendors"][out["vendors"]["vendor_name"].astype(str).str.strip() == f["vendor"]]
    return out


# --------------------------------------------------------------------------- #
# Data quality (surfaced, not hidden)
# --------------------------------------------------------------------------- #
def _invalid_vendor(name) -> bool:
    s = str(name).strip()
    if not s or s.lower() in ("null", "nan", "none"):
        return True
    if len(s) < 3:                      # junk like "fh"
        return True
    if not any(ch.isalpha() for ch in s):
        return True
    return False


def data_quality(model: Dict[str, pd.DataFrame], loader) -> dict:
    assets = model["assets"]
    tickets = model["maintenance_tickets"]
    purchases = model["purchases"]

    no_pdate = assets[assets["purchase_date"].isna()] if not assets.empty and "purchase_date" in assets.columns else assets.iloc[0:0]
    no_warr = assets[assets["warranty_active"].isna()] if not assets.empty and "warranty_active" in assets.columns else assets.iloc[0:0]
    no_apt = tickets[tickets["apartment_code"].isna() | (tickets["apartment_code"].astype(str).str.strip() == "")] if not tickets.empty else tickets
    unk_type = tickets[tickets["maintenance_type"].isna() | (tickets["maintenance_type"].astype(str).str.strip().isin(["", "Unknown"]))] if not tickets.empty else tickets

    # invalid vendor names across the name-keyed sources
    vnames = set()
    for acc, col in [("maintenance_cost_lines", "vendor_name"), ("maintenance_item_purchases", "vendor_name")]:
        df = getattr(loader, acc)()
        if not df.empty and col in df.columns:
            vnames |= {str(v).strip() for v in df[col]}
    invalid_vendors = sorted({v for v in vnames if _invalid_vendor(v)})

    # outlier purchase costs (IQR fence on total_purchase_cost)
    outliers = purchases.iloc[0:0]
    if not purchases.empty and "total_purchase_cost" in purchases.columns:
        v = _num(purchases["total_purchase_cost"]).dropna()
        if len(v) >= 4:
            q1, q3 = v.quantile(0.25), v.quantile(0.75)
            fence = q3 + 3 * (q3 - q1)
            outliers = purchases[_num(purchases["total_purchase_cost"]) > fence]

    return {
        "assets_without_purchase_date": int(len(no_pdate)),
        "assets_without_warranty": int(len(no_warr)),
        "tickets_without_apartment": int(len(no_apt)),
        "tickets_unknown_type": int(len(unk_type)),
        "invalid_vendor_count": int(len(invalid_vendors)),
        "invalid_vendor_names": invalid_vendors,
        "outlier_purchase_count": int(len(outliers)),
        "tables": {
            "assets_without_purchase_date": no_pdate[[c for c in ["asset_code", "asset_type", "apartment_code", "condition"] if c in no_pdate.columns]].head(200),
            "assets_without_warranty": no_warr[[c for c in ["asset_code", "asset_type", "apartment_code", "purchase_date"] if c in no_warr.columns]].head(200),
            "tickets_unknown_type": unk_type[[c for c in ["ticket_id", "current_status", "apartment_code", "opened_at"] if c in unk_type.columns]].head(200),
            "outlier_purchases": outliers[[c for c in ["item_name", "vendor_name", "quantity_received", "purchase_cost_per_unit", "total_purchase_cost"] if c in outliers.columns]].head(200),
        },
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def build_base(loader) -> dict:
    """Build the (expensive) maintenance model + filter options once."""
    model = M.build_maintenance_model(loader)
    return {"model": model, "options": filter_options(model, loader)}


def compute_all(loader, filters: Optional[dict] = None, base: Optional[dict] = None) -> dict:
    """Compute every section. Reuses ``base`` (build_base) when supplied, else
    builds the model. Applies ``filters`` (in-memory) before computing."""
    if base is not None:
        model = base["model"]
        options = base["options"]
    else:
        model = M.build_maintenance_model(loader)
        options = filter_options(model, loader)
    fmodel = apply_filters(model, loader, filters)
    return {
        "model": model,
        "fmodel": fmodel,
        "options": options,
        "dq": data_quality(fmodel, loader),
        "kpis": kpi_cards(fmodel, loader),
        "tickets": ticket_analytics(fmodel, loader),
        "assets": asset_analytics(fmodel),
        "vendors": vendor_analytics(fmodel, loader),
        "purchases": purchase_inventory_analytics(fmodel, loader),
        "costs": cost_analytics(fmodel, loader),
    }
