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
    try:
        return pd.read_csv(p, dtype=str)
    except pd.errors.EmptyDataError:   # empty/header-less artefact (e.g. 0 alerts)
        return pd.DataFrame()


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
# Verification Queue (deterministic business rule; separate from ML anomaly)
# --------------------------------------------------------------------------- #
_SEV_COLOR = {"Critical": "#b00020", "High": "#d97706", "Medium": "#ca8a04"}
_SEV_EMOJI = {"Critical": "🔴", "High": "🟠", "Medium": "🟡"}
_SEV_ORDER = {"Critical": 0, "High": 1, "Medium": 2}
_REVIEW_OPTIONS = ["Pending Review", "Verified Genuine",
                   "Verified Fake", "Not Enough Evidence", "Asset Replaced"]
_FAKE_REASONS = ["Duplicate Ticket", "No Actual Issue", "Tenant Misuse",
                 "Staff Error", "System Error", "Other"]
_REVIEW_PATH = _OUT / "maintenance_verification_reviews.csv"
_REVIEW_LEDGER = _OUT / "maintenance_review_history.csv"
_LEDGER_COLS = ["ticket_id", "asset_id", "previous_status", "new_status",
                "fake_reason", "reviewed_by", "reviewed_at"]


_HIST_QUEUE = "maintenance_historical_audit_queue.csv"
_FUT_QUEUE = "maintenance_future_verification_queue.csv"


@st.cache_data(show_spinner=False)
def _queue_view(fname: str) -> pd.DataFrame:
    df = _csv(fname)
    if df.empty:
        return df
    for c in ["faster_percentage", "current_resolution_hours",
              "asset_historical_median", "severity", "recent_repeat_count_90d",
              "days_since_previous_repair"]:
        if c in df.columns:
            df[c] = _num(df[c])
    df["current_repair_date"] = pd.to_datetime(df.get("current_repair_date"), errors="coerce")
    df["_ord"] = df["severity_level"].map(_SEV_ORDER).fillna(9)
    return df.sort_values(["_ord", "faster_percentage"], ascending=[True, False]) \
             .reset_index(drop=True)


def _load_reviews() -> pd.DataFrame:
    if _REVIEW_PATH.exists():
        try:
            return pd.read_csv(_REVIEW_PATH, dtype=str)
        except Exception:  # noqa: BLE001
            pass
    return pd.DataFrame(columns=["ticket_id", "asset_code", "severity_level",
                                 "faster_percentage", "review_status", "fake_reason",
                                 "reviewed_by", "notes", "reviewed_at"])


def _append_review_ledger(ticket_id: str, asset_id: str, previous_status: str,
                          new_status: str, fake_reason: str, reviewed_by: str,
                          reviewed_at: str) -> None:
    """APPEND-ONLY permanent review-change ledger. Never updates/deletes rows.
    Audit-only — has no effect on baselines or verification decisions."""
    row = pd.DataFrame([{
        "ticket_id": str(ticket_id), "asset_id": str(asset_id),
        "previous_status": previous_status or "(none)", "new_status": new_status,
        "fake_reason": fake_reason if new_status == "Verified Fake" else "",
        "reviewed_by": reviewed_by, "reviewed_at": reviewed_at,
    }])[_LEDGER_COLS]
    _OUT.mkdir(parents=True, exist_ok=True)
    row.to_csv(_REVIEW_LEDGER, mode="a", header=not _REVIEW_LEDGER.exists(), index=False)


def _save_review(ticket_id: str, asset_code: str, severity_level: str,
                 faster_percentage, status: str, notes: str,
                 fake_reason: str = "", reviewed_by: str = "",
                 asset_id: str = "") -> None:
    rv = _load_reviews()
    prev = rv.loc[rv["ticket_id"].astype(str) == str(ticket_id), "review_status"]
    previous_status = str(prev.iloc[0]) if len(prev) else ""
    rv = rv[rv["ticket_id"].astype(str) != str(ticket_id)]
    reviewed_at = pd.Timestamp.now().isoformat(timespec="seconds")
    new = pd.DataFrame([{
        "ticket_id": str(ticket_id), "asset_code": asset_code,
        "severity_level": severity_level, "faster_percentage": faster_percentage,
        "review_status": status,
        "fake_reason": fake_reason if status == "Verified Fake" else "",
        "reviewed_by": reviewed_by, "notes": notes,
        "reviewed_at": reviewed_at,
    }])
    out = pd.concat([rv, new], ignore_index=True)
    out.to_csv(_REVIEW_PATH, index=False)
    # permanent change ledger: append a row only when the status actually changed
    if status != previous_status:
        _append_review_ledger(ticket_id, asset_id, previous_status, status,
                              fake_reason, reviewed_by, reviewed_at)
    _load_reviews_cached.clear()


@st.cache_data(show_spinner=False)
def _load_reviews_cached() -> Dict[str, dict]:
    rv = _load_reviews()
    return {str(r["ticket_id"]): dict(r) for _, r in rv.iterrows()}


def _fmt_dur(h) -> str:
    h = _num(pd.Series([h])).iloc[0]
    if pd.isna(h):
        return "—"
    return f"{int(round(h * 60))} min" if h < 1 else f"{h:.1f} h"


def _rebuild_historical() -> str:
    """Regenerate the Phase-1 Historical Repair Audit queue (reviewed repairs drop out)."""
    try:
        from maintenance.verification_queue import build_historical_audit
        build_historical_audit(loader=_loader())
        _queue_view.clear(); _fake_metrics.clear()
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


def _rebuild_future() -> str:
    """Regenerate the Phase-2 Future Verification queue + refresh baseline store."""
    try:
        from maintenance.verification_queue import build_future_verification
        build_future_verification(loader=_loader())
        _queue_view.clear(); _fake_metrics.clear()
        return "ok"
    except Exception as exc:  # noqa: BLE001
        return f"error: {exc}"


@st.cache_data(show_spinner=False)
def _fake_metrics() -> Dict[str, dict]:
    """asset_id -> fake-repair audit metrics (count, last date, %, business note)."""
    df = _csv("maintenance_asset_fake_metrics.csv")
    if df.empty:
        return {}
    return {str(r["asset_id"]): dict(r) for _, r in df.iterrows()}


def _rebuild_asset_baseline(asset_id: str):
    """Targeted: rebuild ONLY this asset's baseline row (not every asset)."""
    try:
        from maintenance.verification_queue import update_asset_baseline
        return update_asset_baseline(asset_id, loader=_loader())
    except Exception:  # noqa: BLE001
        return None


def _render_historical_audit() -> None:
    st.subheader("Historical Repair Audit Findings")
    st.caption(
        "**Phase 1 — historical repair analysis.** Historical repairs that were "
        "completed **faster than the asset's typical historical repair duration**, "
        "based on available historical records. These are **observations from old "
        "data only** — the system does not conclude anything and recommends no "
        "action. The owner decides what to do after reviewing them; marking a "
        "repair *Verified Genuine* builds the clean baseline."
    )
    if st.button("↻ Refresh findings", key="hist_refresh"):
        res = _rebuild_historical()
        st.success("Findings refreshed." if res == "ok" else res); st.rerun()
    vq = _queue_view(_HIST_QUEUE)
    if vq.empty:
        st.info("No historical repair observations found. Click **Refresh findings** "
                "to (re)generate the analysis from the historical data.")
        return
    _render_historical_findings(vq)


def _render_future_verification() -> None:
    st.subheader("Future Verification Queue")
    st.caption(
        "**Phase 2 — day-to-day operation.** Newly completed repairs compared "
        "against each asset's **clean (Verified Genuine) historical baseline**. "
        "If a repair is much faster than that asset normally takes, it appears "
        "here for verification. Investigation aid only — never fraud."
    )
    if st.button("↻ Refresh queue", key="fut_refresh"):
        res = _rebuild_future()
        st.success("Queue refreshed." if res == "ok" else res); st.rerun()
    vq = _queue_view(_FUT_QUEUE)
    if vq.empty:
        st.info("No new repairs require verification. New completed repairs appear "
                "here once the queue is refreshed.")
        return
    _render_queue_section(vq, "fut", _REVIEW_OPTIONS, _rebuild_future)


def _blank(v) -> str:
    """NaN/None/empty -> '' for display (never show 'nan')."""
    s = "" if v is None else str(v)
    return "" if s.strip().lower() in ("", "nan", "none", "nat", "<na>") else s


_HIST_DECISIONS = ["Verified Genuine", "Verified Fake", "Not Enough Evidence"]


def _render_historical_findings(vq: pd.DataFrame) -> None:
    """Phase-1: pure historical analysis table (the seven business columns only).
    The review system stays in the background (a collapsed optional form) so it
    does not dominate the page."""
    def _fp(v):
        x = _num(pd.Series([v])).iloc[0]
        return f"{x:.1f}%" if pd.notna(x) else ""

    st.caption(f"{len(vq)} historical repair observations across "
               f"{int(vq['asset_id'].nunique())} assets — analysis of old data only.")

    # ---- the analysis table: exactly the seven business columns ---- #
    disp = vq.copy()
    disp["Repair Date"] = pd.to_datetime(
        disp["current_repair_date"], errors="coerce").dt.strftime("%d %b %Y")
    disp["Actual Repair Duration"] = disp["current_resolution_hours"].map(_fmt_dur)
    disp["Typical Historical Repair Duration"] = disp["asset_historical_median"].map(_fmt_dur)
    disp["Faster %"] = disp["faster_percentage"].map(_fp)
    disp["Observation"] = ("Completed much faster than this asset's typical "
                           "historical repair duration.")
    view = disp.rename(columns={"asset_id": "Asset ID", "asset_code": "Asset Code"})[
        ["Asset ID", "Asset Code", "Repair Date", "Actual Repair Duration",
         "Typical Historical Repair Duration", "Faster %", "Observation"]]
    st.dataframe(view, hide_index=True, use_container_width=True, height=440)

    # ---- background review recorder (collapsed; optional; does not dominate) ---- #
    with st.expander("Record an owner decision (optional — used only to build clean "
                     "future baselines)", expanded=False):
        reviews = _load_reviews_cached()
        opts = vq.copy()
        opts["_lbl"] = (opts["asset_code"].astype(str) + " · "
                        + pd.to_datetime(opts["current_repair_date"], errors="coerce")
                        .dt.strftime("%d %b %Y").fillna("") + " · "
                        + opts["faster_percentage"].map(_fp) + " faster")
        pick = st.selectbox("Record", opts["_lbl"], key="hist_pick")
        row = opts.loc[opts["_lbl"] == pick].iloc[0]
        tid = str(row["ticket_id"])
        cur = _blank(reviews.get(tid, {}).get("review_status"))
        cur = cur if cur in _HIST_DECISIONS else ""
        dopts = [""] + _HIST_DECISIONS
        c1, c2 = st.columns([2, 3])
        decision = c1.selectbox("Decision", dopts,
                                index=dopts.index(cur) if cur in dopts else 0,
                                key=f"hd_{tid}",
                                format_func=lambda x: "— choose —" if x == "" else x)
        reviewed_by = c2.text_input(
            "Reviewed by", value=_blank(reviews.get(tid, {}).get("reviewed_by")),
            key=f"hby_{tid}", placeholder="name / id")
        notes = c2.text_input("Notes", value=_blank(reviews.get(tid, {}).get("notes")),
                              key=f"hn_{tid}")
        fake_reason = ""
        if decision == "Verified Fake":
            prev = _blank(reviews.get(tid, {}).get("fake_reason"))
            fidx = _FAKE_REASONS.index(prev) if prev in _FAKE_REASONS else 0
            fake_reason = c1.selectbox("Fake reason", _FAKE_REASONS, index=fidx,
                                       key=f"hfr_{tid}")
        if st.button("Save decision", key=f"hs_{tid}"):
            if decision == "":
                st.error("Choose a decision before saving."); st.stop()
            if decision == "Verified Fake" and not fake_reason:
                st.error("Select a fake reason."); st.stop()
            _save_review(tid, row["asset_code"], row.get("severity_level", ""),
                         row.get("faster_percentage", ""), decision, notes,
                         fake_reason=fake_reason, reviewed_by=reviewed_by,
                         asset_id=row.get("asset_id", ""))
            _rebuild_asset_baseline(row.get("asset_id", ""))
            _rebuild_historical()
            st.success(f"Recorded: {decision}."); st.rerun()


def _render_queue_section(vq: pd.DataFrame, prefix: str, review_options: list,
                          on_rebuild, assess_label: str = "Verification Status",
                          count_label: str = "Repairs to verify") -> None:
    reviews = _load_reviews_cached()
    # KPIs
    k = st.columns(4)
    k[0].metric(count_label, len(vq))
    k[1].metric("Critical", int((vq["severity_level"] == "Critical").sum()))
    k[2].metric("High", int((vq["severity_level"] == "High").sum()))
    reviewed = sum(1 for t in vq["ticket_id"].astype(str)
                   if reviews.get(t, {}).get("review_status", "Pending Review")
                   not in ("", "Pending Review"))
    k[3].metric("Reviewed", f"{reviewed}/{len(vq)}")

    # ---- overview table (color-coded severity) ---- #
    disp = vq.copy()
    disp["Faster %"] = disp["faster_percentage"].map(lambda v: f"{v:.1f}%")
    disp["Current"] = disp["current_resolution_hours"].map(_fmt_dur)
    disp["Historical median"] = disp["asset_historical_median"].map(_fmt_dur)
    disp["Current Repair Date"] = disp["current_repair_date"].dt.strftime("%d %b %Y")
    disp["Status"] = disp["ticket_id"].astype(str).map(
        lambda t: reviews.get(t, {}).get("review_status") or "Pending Review")
    if "baseline_history_type" not in disp.columns:
        disp["baseline_history_type"] = ""
    view = disp.rename(columns={
        "asset_code": "Asset Code", "baseline_confidence": "Baseline Confidence",
        "baseline_history_type": "Baseline History",
        "recent_repeat_count_90d": "Recent Repairs (90d)",
        "severity_level": "Severity", "verification_status": assess_label,
    })[["Asset Code", "Current Repair Date", "Current", "Historical median",
        "Faster %", "Baseline Confidence", "Baseline History", "Recent Repairs (90d)",
        "Severity", assess_label, "Status"]]

    def _sev_style(v):
        return f"background-color:{_SEV_COLOR.get(v,'')};color:white;font-weight:600"
    st.dataframe(view.style.map(_sev_style, subset=["Severity"]),
                 hide_index=True, use_container_width=True, height=380)

    # ---- expandable rows: drill-down + manager review ---- #
    st.markdown("##### Review alerts")
    sev_filter = st.multiselect("Show severity", ["Critical", "High", "Medium"],
                                default=["Critical", "High", "Medium"],
                                key=f"{prefix}_sev")
    shown = vq[vq["severity_level"].isin(sev_filter)] if sev_filter else vq
    for _, r in shown.iterrows():
        tid = str(r["ticket_id"])
        cur = reviews.get(tid, {}).get("review_status") or "Pending Review"
        emoji = _SEV_EMOJI.get(r["severity_level"], "")
        title = (f"{emoji} {r['asset_code']} · {r['faster_percentage']:.1f}% faster "
                 f"· {r['severity_level']} · [{cur}]")
        with st.expander(title, expanded=False):
            c1, c2, c3 = st.columns(3)
            c1.metric("Current repair", _fmt_dur(r["current_resolution_hours"]))
            c2.metric("Asset historical median", _fmt_dur(r["asset_historical_median"]))
            c3.metric("Faster than normal", f"{r['faster_percentage']:.1f}%")
            meta = st.columns(3)
            meta[0].markdown(f"**Baseline:** {r.get('baseline_source','')} "
                             f"({r.get('baseline_confidence','')} confidence, "
                             f"n={r.get('asset_history_sample_size','')}) · "
                             f"**{r.get('baseline_history_type','')}** history "
                             f"(verified={int(r.get('verified_history_size') or 0)}, "
                             f"raw={int(r.get('raw_history_size') or 0)})")
            pr = r.get("previous_repair_date")
            meta[1].markdown(f"**Previous repair:** "
                             f"{str(pr)[:10] if str(pr) not in ('nan','NaT','') else '—'}"
                             + (f" ({r['days_since_previous_repair']:.0f} d ago)"
                                if pd.notna(r.get('days_since_previous_repair')) else ""))
            meta[2].markdown(f"**Recent repairs (90d):** "
                             f"{int(r.get('recent_repeat_count_90d') or 0)}")

            # audit-only fake-repair note (does NOT affect the decision)
            fm = _fake_metrics().get(str(r.get("asset_id", "")))
            if fm:
                note = str(fm.get("business_note") or "").strip()
                fc = fm.get("fake_repair_count", 0)
                fp = fm.get("fake_repair_percentage", 0)
                if note:
                    st.warning(f"⚠ {note} ({fc} fake repairs, {fp}% of this asset's "
                               "repairs) — audit signal only.")
                elif str(fc) not in ("", "0", "nan"):
                    st.caption(f"Audit: {fc} fake repair(s) on record ({fp}%).")

            st.markdown("**Repair timeline** (historical durations → current):")
            st.code(str(r.get("repair_timeline", "")), language=None)
            st.info(r.get("explanation", ""))

            # ---- manager review ---- #
            rc1, rc2 = st.columns([2, 3])
            idx = review_options.index(cur) if cur in review_options else 0
            status = rc1.radio("Manager review", review_options, index=idx,
                               key=f"{prefix}_rev_{tid}")
            reviewed_by = rc2.text_input(
                "Reviewed by", value=reviews.get(tid, {}).get("reviewed_by", "") or "",
                key=f"{prefix}_rby_{tid}", placeholder="manager name / id")
            notes = rc2.text_input("Notes (optional)",
                                   value=reviews.get(tid, {}).get("notes", "") or "",
                                   key=f"{prefix}_note_{tid}")
            # fake_reason required when marking Verified Fake
            fake_reason = ""
            if status == "Verified Fake":
                prev_fr = reviews.get(tid, {}).get("fake_reason", "") or ""
                fr_idx = _FAKE_REASONS.index(prev_fr) if prev_fr in _FAKE_REASONS else 0
                fake_reason = rc1.selectbox("Fake reason (required)", _FAKE_REASONS,
                                            index=fr_idx, key=f"{prefix}_frs_{tid}")
            if st.button("Save review", key=f"{prefix}_save_{tid}"):
                if status == "Verified Fake" and not fake_reason:
                    st.error("Select a fake_reason before saving a Verified Fake.")
                    st.stop()
                _save_review(tid, r["asset_code"], r["severity_level"],
                             r["faster_percentage"], status, notes,
                             fake_reason=fake_reason, reviewed_by=reviewed_by,
                             asset_id=r.get("asset_id", ""))
                # TARGETED: any review change rebuilds only THIS asset's baseline,
                # so a Verified Fake (or downgrade from Genuine) immediately drops
                # that repair from the verified history and the baseline.
                row = _rebuild_asset_baseline(r.get("asset_id", ""))
                cnt = int(row["verified_repair_count"]) if row else 0
                on_rebuild()   # refresh this phase's queue (reviewed drop from Phase 1)
                if status == "Asset Replaced":
                    st.success(f"Saved: {status}. Old verified history archived; "
                               f"baseline for {r['asset_code']} reset (version 1).")
                elif status == "Verified Fake":
                    st.success(f"Saved: {status}. Repair removed from verified history; "
                               f"baseline for {r['asset_code']} rebuilt "
                               f"(verified repairs = {cnt}).")
                elif status == "Verified Genuine":
                    st.success(f"Saved: {status}. Rebuilt baseline for asset "
                               f"{r['asset_code']} (verified repairs = {cnt}).")
                else:
                    st.success(f"Saved: {status}. Baseline for {r['asset_code']} "
                               f"refreshed (verified repairs = {cnt}).")
                st.rerun()

    st.caption(
        f"Review decisions are stored in `{_REVIEW_PATH.name}` (ticket_id, status, "
        "notes, timestamp) as labels for future supervised models. Note: on "
        "ephemeral hosting (e.g. Streamlit Cloud) this file resets on redeploy — "
        "commit it or use a database for durable labels."
    )


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

    tab_h, tab_f, tab_t, tab_a, tab_n, tab_c = st.tabs(
        ["🧹 Historical Repair Audit", "🔎 Future Verification Queue",
         "Suspicious Tickets", "Suspicious Assets", "Suspicious Tenants",
         "Technicians"])

    # ===== Phase 1 — Historical Repair Audit (one-time cleanup) ===== #
    with tab_h:
        _render_historical_audit()
    # ===== Phase 2 — Future Verification Queue (production) ===== #
    with tab_f:
        _render_future_verification()

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
