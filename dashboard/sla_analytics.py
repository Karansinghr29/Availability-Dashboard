"""SLA Performance — operations dashboard page (standalone).

Reads the deterministic SLA tables from the sla_analytics module. Completely
independent of the verification / historical-analysis pages.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner="Building SLA analytics…")
def _sla_tables() -> dict:
    from data_loader import DataLoader
    from sla.sla_analytics import build_sla_analytics
    return dict(build_sla_analytics(loader=DataLoader()))


def _kpi_map(kpis: pd.DataFrame) -> dict:
    return {r["metric"]: r["value"] for _, r in kpis.iterrows()}


def _mlabel(m: str) -> str:
    try:
        return pd.to_datetime(str(m) + "-01").strftime("%b %Y")
    except Exception:  # noqa: BLE001
        return str(m)


def render() -> None:
    st.header("SLA Performance Analytics")
    st.caption(
        "Operations dashboard. Per ticket: **SLA deadline = created_at + SLA hours**, "
        "**Resolution = resolved_at − created_at**, **Closure = closed_at − created_at**. "
        "Descriptive performance reporting only — not fraud detection or verification."
    )
    if st.button("↻ Refresh SLA analytics"):
        _sla_tables.clear()
        st.rerun()

    t = _sla_tables()
    monthly = t["monthly_summary"]
    k = _kpi_map(t["kpis"])
    cov = _kpi_map(t["coverage"])

    if int(k.get("total_tickets", 0)) == 0:
        st.info("No SLA-evaluable tickets (need created_at and resolved_at or closed_at).")
        return

    # ---- operational alert (PRIMARY = SLA vs Closure) ---- #
    if str(k.get("operational_alert")).lower() == "true":
        st.error(f"⚠ {k.get('operational_alert_message')}")
    elif str(k.get("month_exceeded_message")):
        st.warning(str(k.get("month_exceeded_message")))
    else:
        st.success(str(k.get("operational_alert_message")))
    if str(k.get("resolution_alert")).lower() == "true" and str(k.get("resolution_alert_message")):
        st.info(str(k.get("resolution_alert_message")))
    st.caption(
        "Primary comparison: **configured SLA vs actual closure time**. Resolution "
        "time is shown alongside to locate the delay (before vs after resolution). "
        "`resolved_by` identifies the resolver/technician where available; "
        "`closed_at` is the final closure time and is **not** attributed to a person."
    )

    # ==== 1) Overall Performance (all evaluable tickets) ==== #
    st.subheader("Overall Performance")
    cov_c = st.columns(3)
    cov_c[0].metric("Total Maintenance Tickets", int(cov["total_tickets"]))
    cov_c[1].metric("SLA Evaluable Tickets", int(cov["evaluable_tickets"]))
    cov_c[2].metric("Not Evaluable", int(cov["excluded_total"]))
    st.caption("Averages below are **overall / across all "
               f"{int(cov['evaluable_tickets'])} evaluable tickets** — not a single month.")
    o = st.columns(5)
    o[0].metric("Avg SLA (overall)", f"{k['avg_sla_hours']} h")
    o[1].metric("Avg Resolution (tracked)", f"{k['avg_resolution_hours']} h")
    o[2].metric("Avg Closure (overall)", f"{k['avg_closure_hours']} h")
    o[3].metric("Avg Delay (overall)", f"{k['avg_delay_hours']} h")
    o[4].metric("SLA Breach % (overall)", f"{k['breach_pct']}%")
    ntr = int(cov.get("resolution_tracked_tickets", 0))
    st.caption(
        f"**Avg Resolution: {k['avg_resolution_hours']} h** — based on **{ntr} tracked "
        f"resolution tickets** (resolved_at > created_at). Resolution tracking "
        f"coverage: **{ntr} / {int(cov['total_tickets'])}**. "
        "resolved_at == created_at is treated as untracked, not a 0-hour resolution."
    )

    # ==== 1b) Delay Breakdown: SLA -> Resolution -> Closure (SECONDARY diagnostic) ==== #
    st.subheader("Delay Breakdown: SLA → Resolution → Closure")
    st.caption(
        "**Secondary diagnostic** — does **not** change the primary SLA verdict "
        "(which stays **SLA vs Closure**). Ticket lifecycle: "
        "**Created → SLA Deadline → Resolved → Closed**. Use it to see **where** the "
        "delay happens: a large **Resolution Delay** means the wait is **before** "
        "resolution; a large **Post-Resolution Closure** means the wait is **after** "
        "resolution (ticket resolved but not closed). Resolution-based figures use "
        "**tracked** records only (resolved_at > created_at); untracked rows show **—**."
    )
    st.caption(
        "Example — SLA 24h, Resolved 40h, Closed 100h → Resolution Delay **+16h**, "
        "Post-Resolution Closure **60h**, Total Closure **100h**, Closure Delay "
        "**+76h**, Primary SLA Status **SLA Breached**."
    )
    b = st.columns(6)
    b[0].metric("Avg SLA", f"{k['avg_sla_hours']} h")
    b[1].metric("Avg Tracked Resolution", f"{k['avg_resolution_hours']} h")
    b[2].metric("Avg Resolution Delay", f"{k.get('avg_resolution_delay_hours', 0)} h")
    b[3].metric("Avg Post-Resolution Closure", f"{k.get('avg_post_resolution_closure_hours', 0)} h")
    b[4].metric("Avg Closure", f"{k['avg_closure_hours']} h")
    b[5].metric("Avg Closure Delay", f"{k['avg_delay_hours']} h")   # = mean closure_delay (primary)

    if not monthly.empty:
        st.markdown("**Month-wise delay breakdown** (tracked resolution only; **—** where no tracked data)")
        bd = monthly.copy()
        bd["Month"] = bd["month"].map(_mlabel)
        for c in ("avg_resolution_hours", "avg_resolution_delay", "avg_post_resolution_closure"):
            bd[c] = bd[c].map(lambda v: "—" if pd.isna(v) else str(v))
        bd = bd.rename(columns={
            "avg_sla_hours": "Avg SLA", "avg_resolution_hours": "Avg Resolution",
            "avg_resolution_delay": "Avg Resolution Delay",
            "avg_post_resolution_closure": "Avg Post-Resolution Closure",
            "avg_closure_hours": "Avg Closure", "avg_closure_delay": "Avg Closure Delay"})[
            ["Month", "Avg SLA", "Avg Resolution", "Avg Resolution Delay",
             "Avg Post-Resolution Closure", "Avg Closure", "Avg Closure Delay"]]
        st.dataframe(bd, hide_index=True, use_container_width=True)

    if not monthly.empty:
        last = monthly.iloc[-1]
        prev = monthly.iloc[-2] if len(monthly) >= 2 else None

        # ==== 2) Latest Month Performance (with vs-previous deltas) ==== #
        st.subheader(f"Latest Month Performance — {_mlabel(last['month'])}")

        def _hv(v, unit):   # "—" for untracked/NaN months
            return "—" if pd.isna(v) else f"{v}{unit}"

        def _d(cur, col, unit, inv=True):
            if prev is None or pd.isna(cur) or pd.isna(prev[col]):
                return None, None
            ch = round(float(cur) - float(prev[col]), 1)
            return f"{ch:+.1f}{unit}", ("inverse" if inv else "normal")

        m = st.columns(7)
        dt, _ = (f"{int(last['tickets']-prev['tickets']):+d}", "normal") if prev is not None else (None, None)
        m[0].metric("Tickets", int(last["tickets"]), delta=dt, delta_color="off")
        for i, (lab, col, unit) in enumerate([
                ("Avg SLA", "avg_sla_hours", "h"), ("Avg Resolution", "avg_resolution_hours", "h"),
                ("Avg Closure", "avg_closure_hours", "h"),
                ("Avg Res Delay", "avg_resolution_delay", "h"),
                ("Avg Clo Delay", "avg_closure_delay", "h"),
                ("SLA Breach %", "breach_pct", "%")], start=1):
            dv, dc = _d(last[col], col, unit)
            m[i].metric(lab, _hv(last[col], unit), delta=dv, delta_color=(dc or "off"))

        # ==== 3) Current vs Previous Month (explicit numbers) ==== #
        if prev is not None:
            st.subheader(f"Current vs Previous Month — {_mlabel(last['month'])} vs {_mlabel(prev['month'])}")
            rows = []
            for lab, col, unit in [
                    ("Avg SLA", "avg_sla_hours", "h"), ("Avg Resolution", "avg_resolution_hours", "h"),
                    ("Avg Closure", "avg_closure_hours", "h"),
                    ("Avg Resolution Delay", "avg_resolution_delay", "h"),
                    ("Avg Closure Delay", "avg_closure_delay", "h"),
                    ("SLA Breach %", "breach_pct", "%")]:
                c_, p_ = last[col], prev[col]
                ch = "—" if (pd.isna(c_) or pd.isna(p_)) else f"{float(c_)-float(p_):+.1f}{unit}"
                rows.append({"Metric": lab,
                             _mlabel(last["month"]): _hv(c_, unit),
                             _mlabel(prev["month"]): _hv(p_, unit),
                             "Change": ch})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        else:
            st.caption("Only one month of data — no previous month to compare.")

        # ==== 4) Monthly Performance Table ==== #
        st.subheader("Monthly Performance Table")
        show = monthly.copy()
        show["Month"] = show["month"].map(_mlabel)
        # tracked resolution is blank ('—') for months with no tracked records
        # (cast whole column to str so it stays Arrow-serialisable)
        for c in ("avg_resolution_hours", "avg_resolution_delay"):
            show[c] = show[c].map(lambda v: "—" if pd.isna(v) else str(v))
        show = show.rename(columns={
            "tickets": "Tickets", "avg_sla_hours": "Avg SLA",
            "avg_resolution_hours": "Avg Resolution", "avg_closure_hours": "Avg Closure",
            "avg_resolution_delay": "Avg Resolution Delay", "avg_closure_delay": "Avg Closure Delay",
            "breach_pct": "SLA Breach %"})[
            ["Month", "Tickets", "Avg SLA", "Avg Resolution", "Avg Closure",
             "Avg Resolution Delay", "Avg Closure Delay", "SLA Breach %"]]
        st.dataframe(show, hide_index=True, use_container_width=True)

        # ==== 5) Monthly Trends (presentation only — separate scales) ==== #
        # Values are taken directly from monthly_summary; nothing is aggregated,
        # filtered, or recalculated here. Charts are split so a large Closure
        # scale never compresses SLA/Resolution, and hours never share an axis
        # with percentages.
        st.subheader("Monthly Trends")
        mi = monthly.copy()
        mi["Month"] = mi["month"].map(_mlabel)     # readable month labels; order preserved
        mi = mi.set_index("Month")

        # --- Chart 1: SLA vs Resolution vs Closure — one panel each (own scale) ---
        st.markdown("**Chart 1 — Monthly SLA vs Resolution vs Closure** (each on its own hours scale)")
        st.caption("Avg SLA (hours)")
        st.line_chart(mi[["avg_sla_hours"]].rename(columns={"avg_sla_hours": "Avg SLA (h)"}), height=180)
        st.caption("Avg Resolution — tracked records only (hours)")
        st.line_chart(mi[["avg_resolution_hours"]].rename(columns={"avg_resolution_hours": "Avg Resolution (h)"}), height=180)
        st.caption("Avg Closure (hours)")
        st.line_chart(mi[["avg_closure_hours"]].rename(columns={"avg_closure_hours": "Avg Closure (h)"}), height=180)

        # --- Chart 2: Delays (hours) separate from Breach (%) ---
        st.markdown("**Chart 2 — Delay (hours) and SLA Breach (%)** (separate axes)")
        st.caption("Avg Resolution Delay vs Avg Closure Delay (hours)")
        st.line_chart(mi[["avg_resolution_delay", "avg_closure_delay"]].rename(columns={
            "avg_resolution_delay": "Avg Resolution Delay (h)",
            "avg_closure_delay": "Avg Closure Delay (h)"}), height=200)
        st.caption("SLA Breach % (closure-based, primary)")
        st.line_chart(mi[["breach_pct"]].rename(columns={"breach_pct": "SLA Breach %"}), height=200)

    # ---- delay distribution ---- #
    st.subheader("Delay distribution")
    dd = t["delay_distribution"]
    if not dd.empty:
        st.bar_chart(dd.set_index("delay_bucket")["tickets"], height=220)

    # ---- breakdowns ---- #
    st.subheader("Breakdowns")

    def _show(df: pd.DataFrame, first: str):
        if df.empty:
            st.caption("No data.")
            return
        v = df.rename(columns={
            first: first.replace("_", " ").title(), "tickets": "Tickets",
            "within_sla": "Within SLA", "breached": "Breached", "breach_pct": "Breach %",
            "avg_sla_hours": "Avg SLA (h)", "avg_resolution_hours": "Avg Resolution (h)",
            "avg_closure_hours": "Avg Closure (h)", "avg_delay_hours": "Avg Delay (h)"})
        st.dataframe(v, hide_index=True, use_container_width=True, height=340)

    ta, tx, tt, ti = st.tabs(["Apartment", "Asset Type", "Technician", "Issue Type"])
    with ta:
        _show(t["apartment"], "apartment")
    with tx:
        _show(t["asset_type"], "asset_type")
    with tt:
        st.caption("Technician = ticket `assigned_to`.")
        _show(t["technician"], "technician")
    with ti:
        _show(t["issue_type"], "issue_type")

    # ---- ticket details ---- #
    st.subheader("Ticket details")
    st.caption("Every evaluable ticket — SLA → Resolved → Closed. "
               "`Technician/Resolver` = resolved_by (where available); closure has no attributed actor.")
    det = t["ticket_details"].copy()
    stf = st.multiselect("SLA status", ["Within SLA", "SLA Breached", ""],
                         default=["Within SLA", "SLA Breached", ""], key="sla_status_filter",
                         format_func=lambda x: "Pending closure" if x == "" else x)
    if stf:
        det = det[det["sla_status"].fillna("").isin(stf)]
    det = det.rename(columns={
        "ticket": "Ticket", "technician_resolver": "Technician/Resolver",
        "created_at": "Created At", "sla": "SLA", "sla_deadline": "SLA Deadline",
        "resolved_at": "Resolved At", "resolution_hours": "Resolution Hours",
        "resolution_delay": "Resolution Delay",
        "closed_at": "Closed At", "closure_hours": "Closure Hours",
        "post_resolution_closure_hours": "Post-Resolution Closure Time",
        "closure_delay": "Closure Delay", "sla_status": "SLA Status"})[
        ["Ticket", "Technician/Resolver", "Created At", "SLA", "SLA Deadline",
         "Resolved At", "Resolution Hours", "Resolution Delay", "Closed At",
         "Closure Hours", "Post-Resolution Closure Time", "Closure Delay", "SLA Status"]]
    st.dataframe(det, hide_index=True, use_container_width=True, height=420)

    # ---- coverage ---- #
    with st.expander("Coverage (why any tickets are excluded)", expanded=False):
        st.write(
            f"- Total tickets: **{int(cov['total_tickets'])}**\n"
            f"- With created_at: {int(cov['with_created_at'])}\n"
            f"- With resolved_at: {int(cov['with_resolved_at'])}\n"
            f"- With closed_at: {int(cov['with_closed_at'])}\n"
            f"- **Evaluable (created + resolved or closed): {int(cov['evaluable_tickets'])}**\n"
            f"- Excluded: {int(cov['excluded_total'])} "
            f"(missing created_at: {int(cov['excluded_missing_created_at'])}, "
            f"missing both resolved_at and closed_at: "
            f"{int(cov['excluded_missing_resolved_and_closed'])})"
        )
