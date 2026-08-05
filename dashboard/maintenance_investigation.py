"""Maintenance Investigation — read-only dashboard page (Phase 4).

Renders the Phase-3 investigation risk scores (tickets / assets / tenants /
technicians) with KPIs, filters, ranked tables and per-entity drill-down.

Read-only: reads the ``outputs/*_risk.csv`` and ``*_features.csv`` artefacts and
uses the existing ``DataLoader`` only for display lookups + drill-down detail.
It never recomputes analytics and never mutates any other page's state.

Wired into app.py additively (one PAGES entry); no existing page is modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[1]
_OUT = _ROOT / "outputs"

DISCLAIMER = (
    "This identifies maintenance anomalies requiring investigation. "
    "It does not prove fraud."
)

_LEVEL_ORDER = ["Critical", "High", "Medium", "Low"]
_LEVEL_COLOR = {"Critical": "#b00020", "High": "#d97706",
                "Medium": "#2563eb", "Low": "#6b7280"}


# --------------------------------------------------------------------------- #
# cached IO
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _csv(name: str) -> pd.DataFrame:
    p = _OUT / name
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, dtype=str)


@st.cache_resource(show_spinner=False)
def _loader():
    from data_loader import DataLoader
    return DataLoader()


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


@st.cache_data(show_spinner=False)
def _lookup_maps() -> Dict[str, Dict[str, str]]:
    """id -> human label maps for tenants / beds / assets / issue types."""
    ld = _loader()

    def _s(v) -> str:
        t = "" if v is None else str(v).strip()
        return "" if t.lower() in ("", "nan", "none", "null", "<na>") else t

    def _mk(df, key, val):
        if df.empty or key not in df or val not in df:
            return {}
        return {_s(k): _s(v) for k, v in zip(df[key], df[val]) if _s(k)}

    tn = ld.tenant_master()
    tenant = _mk(tn, "id", "full_name") if "full_name" in tn.columns else _mk(tn, "id", "name")
    beds = ld.beds_master_uuid()
    bed = _mk(beds, "id", "bed_code")
    am = ld.asset_master()
    asset = _mk(am, "id", "asset_code")
    it = ld.issue_types()
    issue = _mk(it, "id", "name")
    return {"tenant": tenant, "bed": bed, "asset": asset, "issue": issue}


# --------------------------------------------------------------------------- #
# enriched views (risk + features + mapping + labels)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def _ticket_view() -> pd.DataFrame:
    risk = _csv("maintenance_ticket_risk.csv")
    mp = _csv("ticket_asset_mapping.csv")
    if risk.empty:
        return pd.DataFrame()
    maps = _lookup_maps()
    df = risk.rename(columns={"entity_id": "ticket_id"})
    if not mp.empty:
        keep = ["ticket_id", "ticket_number", "tenant_id", "apartment_code",
                "bed_id", "issue_type", "created_at", "mapped_asset_id"]
        df = df.merge(mp[[c for c in keep if c in mp.columns]], on="ticket_id", how="left")
    df["risk_score"] = _num(df["risk_score"])
    df["created_at"] = pd.to_datetime(df.get("created_at"), errors="coerce")
    df["tenant"] = df.get("tenant_id", "").map(lambda x: maps["tenant"].get(str(x), "") or str(x)[:8])
    df["bed_code"] = df.get("bed_id", "").map(lambda x: maps["bed"].get(str(x), ""))
    df["room_bed"] = (df.get("apartment_code", "").fillna("") + " / " + df["bed_code"].fillna("")).str.strip(" /")
    df["asset"] = df.get("mapped_asset_id", "").map(lambda x: maps["asset"].get(str(x), "") if str(x) else "")
    return df.sort_values("risk_score", ascending=False)


@st.cache_data(show_spinner=False)
def _asset_view() -> pd.DataFrame:
    risk = _csv("maintenance_asset_risk.csv")
    feat = _csv("maintenance_asset_features.csv")
    if risk.empty:
        return pd.DataFrame()
    maps = _lookup_maps()
    df = risk.rename(columns={"entity_id": "asset_id"})
    if not feat.empty:
        keep = ["asset_id", "asset_type", "total_tickets", "repeat_issue_count",
                "max_same_issue_repeats", "tickets_per_month", "asset_age_days"]
        df = df.merge(feat[[c for c in keep if c in feat.columns]], on="asset_id", how="left")
    df["risk_score"] = _num(df["risk_score"])
    df["asset_code"] = df["asset_id"].map(lambda x: maps["asset"].get(str(x), str(x)[:8]))
    return df.sort_values("risk_score", ascending=False)


@st.cache_data(show_spinner=False)
def _tenant_view() -> pd.DataFrame:
    risk = _csv("maintenance_tenant_risk.csv")
    feat = _csv("maintenance_tenant_features.csv")
    if risk.empty:
        return pd.DataFrame()
    maps = _lookup_maps()
    df = risk.rename(columns={"entity_id": "tenant_id"})
    if not feat.empty:
        keep = ["tenant_id", "total_tickets", "rejection_rate",
                "rejection_count", "avg_resolution_hours"]
        df = df.merge(feat[[c for c in keep if c in feat.columns]], on="tenant_id", how="left")
    df["risk_score"] = _num(df["risk_score"])
    df["tenant"] = df["tenant_id"].map(lambda x: maps["tenant"].get(str(x), str(x)[:8]))
    return df.sort_values("risk_score", ascending=False)


@st.cache_data(show_spinner=False)
def _tech_view() -> pd.DataFrame:
    risk = _csv("maintenance_technician_risk.csv")
    feat = _csv("maintenance_technician_features.csv")
    if risk.empty:
        return pd.DataFrame()
    df = risk.rename(columns={"entity_id": "technician_id"})
    if not feat.empty:
        keep = ["technician_id", "tickets_closed", "avg_resolution_time",
                "average_cost", "rejected_count"]
        df = df.merge(feat[[c for c in keep if c in feat.columns]], on="technician_id", how="left")
    df["risk_score"] = _num(df["risk_score"])
    return df.sort_values("risk_score", ascending=False)


# --------------------------------------------------------------------------- #
# filtering
# --------------------------------------------------------------------------- #
def _apply_filters(tv: pd.DataFrame, levels, issue_types, date_range) -> pd.DataFrame:
    df = tv
    if levels:
        df = df[df["risk_level"].isin(levels)]
    if issue_types:
        df = df[df.get("issue_type", "").isin(issue_types)]
    if date_range and "created_at" in df.columns:
        start, end = date_range
        d = df["created_at"]
        mask = d.isna() | ((d.dt.date >= start) & (d.dt.date <= end))
        df = df[mask]
    return df


# --------------------------------------------------------------------------- #
# drill-down
# --------------------------------------------------------------------------- #
def _drill_ticket(ticket_id: str) -> None:
    ld = _loader()
    maps = _lookup_maps()
    t = ld.maintenance_tickets()
    row = t[t["id"].astype(str) == str(ticket_id)]
    if row.empty:
        st.info("Ticket detail not found.")
        return
    r = row.iloc[0]

    def g(c):
        v = r.get(c)
        return "" if v is None or str(v).strip().lower() in ("", "nan", "none", "null", "nat") else str(v)

    st.markdown(f"**Ticket {g('ticket_number') or ticket_id[:8]}** — {maps['issue'].get(g('issue_type_id'), g('issue_type_id'))}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Status", g("status") or "—")
    c2.metric("Priority", g("priority") or "—")
    c3.metric("Tenant approved", g("tenant_approved") or "—")

    # timeline
    st.markdown("**Timeline**")
    tl = []
    for label, col in [("Created", "created_at"), ("SLA deadline", "sla_deadline"),
                       ("Resolved", "resolved_at"), ("Closed", "closed_at")]:
        v = pd.to_datetime(g(col), errors="coerce")
        if pd.notna(v):
            tl.append({"event": label, "when": v})
    if tl:
        st.dataframe(pd.DataFrame(tl).sort_values("when"), hide_index=True, use_container_width=True)
    else:
        st.caption("No timestamps available.")

    if g("description"):
        st.markdown(f"**Description:** {g('description')[:400]}")
    if g("tenant_rejection_reason"):
        st.warning(f"Rejection reason: {g('tenant_rejection_reason')[:300]}")

    # resolution + cost
    try:
        res = ld.ticket_resolutions()
        rr = res[res["ticket_id"].astype(str) == str(ticket_id)]
        if not rr.empty:
            rd = rr.iloc[0]
            st.markdown("**Resolution & cost**")
            cost = _num(pd.Series([rd.get("total_cost")])).iloc[0]
            cc = st.columns(3)
            cc[0].metric("Total cost", f"₹{cost:,.0f}" if pd.notna(cost) else "—")
            cc[1].metric("Parts", f"₹{_num(pd.Series([rd.get('total_parts_cost')])).iloc[0]:,.0f}")
            cc[2].metric("Labour", f"₹{_num(pd.Series([rd.get('total_labour_cost')])).iloc[0]:,.0f}")
            if str(rd.get("closure_summary")).strip() not in ("", "nan", "None"):
                st.caption(f"Closure: {str(rd.get('closure_summary'))[:300]}")
    except Exception:  # noqa: BLE001
        pass

    # related tickets (same bed) + issue history
    bed = g("bed_id")
    if bed:
        rel = t[t["bed_id"].astype(str) == bed].copy()
        if len(rel) > 1:
            rel["issue"] = rel["issue_type_id"].astype(str).map(lambda x: maps["issue"].get(x, ""))
            rel["created"] = pd.to_datetime(rel["created_at"], errors="coerce")
            st.markdown(f"**Related tickets at this bed ({len(rel)})**")
            st.dataframe(
                rel[["ticket_number", "issue", "status", "created"]]
                .sort_values("created", ascending=False).head(30),
                hide_index=True, use_container_width=True,
            )


# --------------------------------------------------------------------------- #
# table styling
# --------------------------------------------------------------------------- #
def _style(df: pd.DataFrame, cols):
    show = df[[c for c in cols if c in df.columns]].copy()
    if "risk_level" in show.columns:
        def _c(v):
            return f"color:{_LEVEL_COLOR.get(v, '#000')};font-weight:600"
        return show.style.map(_c, subset=["risk_level"])
    return show


# --------------------------------------------------------------------------- #
# main render
# --------------------------------------------------------------------------- #
def render() -> None:
    st.header("Maintenance Investigation")
    st.warning(DISCLAIMER)

    tv = _ticket_view()
    av = _asset_view()
    nv = _tenant_view()
    cv = _tech_view()

    if tv.empty and av.empty:
        st.error(
            "Risk artefacts not found in outputs/. Run the maintenance pipeline "
            "first: ticket_asset_mapping → maintenance_features → anomaly_detection."
        )
        return

    # ---- KPIs ---- #
    def _hc(df):
        return int(df["risk_level"].isin(["High", "Critical"]).sum()) if not df.empty else 0

    k = st.columns(5)
    k[0].metric("Tickets analyzed", len(tv))
    k[1].metric("High + Critical tickets", _hc(tv))
    k[2].metric("High-risk assets", _hc(av))
    k[3].metric("High-risk tenants", _hc(nv))
    k[4].metric("High-risk technicians", _hc(cv))

    # ---- filters ---- #
    with st.expander("Filters", expanded=False):
        fc = st.columns(4)
        levels = fc[0].multiselect("Risk level", _LEVEL_ORDER, default=["Critical", "High"])
        issue_opts = sorted([x for x in tv.get("issue_type", pd.Series(dtype=str)).dropna().unique() if x])
        issue_sel = fc[1].multiselect("Issue type", issue_opts)
        asset_opts = sorted([x for x in av.get("asset_type", pd.Series(dtype=str)).dropna().unique() if x])
        asset_sel = fc[2].multiselect("Asset type", asset_opts)
        dts = tv["created_at"].dropna() if "created_at" in tv.columns else pd.Series([], dtype="datetime64[ns]")
        date_range = None
        if len(dts):
            lo, hi = dts.min().date(), dts.max().date()
            date_range = fc[3].date_input("Ticket date range", value=(lo, hi),
                                          min_value=lo, max_value=hi)
            if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
                date_range = None

    tab_t, tab_a, tab_n, tab_c = st.tabs(
        ["Suspicious Tickets", "Suspicious Assets", "Suspicious Tenants", "Technicians"])

    # ===== A) tickets ===== #
    with tab_t:
        ft = _apply_filters(tv, levels, issue_sel, date_range)
        st.caption(f"{len(ft)} tickets after filters")
        cols = ["risk_score", "ticket_number", "tenant", "room_bed", "asset",
                "issue_type", "risk_level", "flags", "explanation"]
        st.dataframe(_style(ft, cols), hide_index=True, use_container_width=True, height=420)
        if not ft.empty:
            opts = ft.head(200).assign(
                lbl=lambda d: d["risk_score"].round(0).astype("Int64").astype(str)
                + " · " + d["ticket_number"].fillna(d["ticket_id"].str[:8])
                + " · " + d["risk_level"])
            pick = st.selectbox("Drill down into ticket", opts["lbl"], key="drill_ticket")
            tid = opts.loc[opts["lbl"] == pick, "ticket_id"].iloc[0]
            with st.container(border=True):
                _drill_ticket(tid)

    # ===== B) assets ===== #
    with tab_a:
        fa = av[av["risk_level"].isin(levels)] if levels else av
        if asset_sel:
            fa = fa[fa["asset_type"].isin(asset_sel)]
        st.caption(f"{len(fa)} assets after filters")
        cols = ["asset_code", "asset_type", "risk_score", "total_tickets",
                "repeat_issue_count", "risk_level", "flags", "explanation"]
        st.dataframe(_style(fa, cols), hide_index=True, use_container_width=True, height=420)
        if not fa.empty:
            maps = _lookup_maps()
            opts = fa.head(200).copy()
            opts["lbl"] = (opts["risk_score"].round(0).astype("Int64").astype(str)
                           + " · " + opts["asset_code"].astype(str) + " · " + opts["risk_level"])
            pick = st.selectbox("Drill down into asset", opts["lbl"], key="drill_asset")
            aid = opts.loc[opts["lbl"] == pick, "asset_id"].iloc[0]
            with st.container(border=True):
                _asset_drill(aid, maps)

    # ===== C) tenants ===== #
    with tab_n:
        fn = nv[nv["risk_level"].isin(levels)] if levels else nv
        st.caption(f"{len(fn)} tenants after filters")
        cols = ["tenant", "total_tickets", "rejection_rate", "risk_score",
                "risk_level", "flags", "explanation"]
        st.dataframe(_style(fn, cols), hide_index=True, use_container_width=True, height=420)
        if not fn.empty:
            opts = fn.head(200).copy()
            opts["lbl"] = (opts["risk_score"].round(0).astype("Int64").astype(str)
                           + " · " + opts["tenant"].astype(str) + " · " + opts["risk_level"])
            pick = st.selectbox("Drill down into tenant", opts["lbl"], key="drill_tenant")
            tid = opts.loc[opts["lbl"] == pick, "tenant_id"].iloc[0]
            with st.container(border=True):
                _tenant_drill(tid)

    # ===== D) technicians (rules only) ===== #
    with tab_c:
        st.caption("Rules-based only — technician sample size is too small (n≈4) "
                   "for a reliable anomaly model.")
        cols = ["technician_id", "tickets_closed", "avg_resolution_time",
                "average_cost", "rejected_count", "risk_score", "risk_level",
                "flags", "explanation"]
        st.dataframe(_style(cv, cols), hide_index=True, use_container_width=True)


def _asset_drill(asset_id: str, maps: Dict) -> None:
    ld = _loader()
    t = ld.maintenance_tickets()
    mp = _csv("ticket_asset_mapping.csv")
    ids = set()
    if not mp.empty:
        ids = set(mp.loc[mp["mapped_asset_id"].astype(str) == str(asset_id), "ticket_id"].astype(str))
        ids |= set(mp.loc[mp["ticket_id"].isin(  # verified direct links
            t.loc[t["asset_id"].astype(str) == str(asset_id), "id"].astype(str)), "ticket_id"].astype(str))
    hist = t[t["id"].astype(str).isin(ids)].copy()
    st.markdown(f"**Asset {maps['asset'].get(str(asset_id), str(asset_id)[:8])} — issue history ({len(hist)})**")
    if hist.empty:
        st.caption("No mapped tickets.")
        return
    hist["issue"] = hist["issue_type_id"].astype(str).map(lambda x: maps["issue"].get(x, ""))
    hist["created"] = pd.to_datetime(hist["created_at"], errors="coerce")
    st.dataframe(
        hist[["ticket_number", "issue", "status", "created"]]
        .sort_values("created", ascending=False).head(40),
        hide_index=True, use_container_width=True)
    ic = hist["issue"].value_counts()
    if not ic.empty:
        st.caption("Issue breakdown: " + ", ".join(f"{k}×{v}" for k, v in ic.head(6).items()))


def _tenant_drill(tenant_id: str) -> None:
    ld = _loader()
    maps = _lookup_maps()
    t = ld.maintenance_tickets()
    hist = t[t["tenant_id"].astype(str) == str(tenant_id)].copy()
    st.markdown(f"**Tenant {maps['tenant'].get(str(tenant_id), str(tenant_id)[:8])} — tickets ({len(hist)})**")
    if hist.empty:
        st.caption("No tickets.")
        return
    hist["issue"] = hist["issue_type_id"].astype(str).map(lambda x: maps["issue"].get(x, ""))
    hist["created"] = pd.to_datetime(hist["created_at"], errors="coerce")
    hist["approved"] = hist["tenant_approved"].astype(str)
    st.dataframe(
        hist[["ticket_number", "issue", "status", "approved", "created"]]
        .sort_values("created", ascending=False).head(40),
        hide_index=True, use_container_width=True)
    rej = (hist["tenant_approved"].astype(str).str.lower() == "false").sum()
    st.caption(f"Rejected/not-approved: {int(rej)} of {len(hist)}")
