"""
Availability AI — Management & Recommendation Dashboard (PHASE 7)
=================================================================

Live data only. Every page rebuilds analytics from DataLoader (Q35–Q39)
production CSVs via existing src modules — never from ``outputs/*.csv``.

  * Inventory          — ``build_room_inventory`` + ``prepare_inventory``
  * Occupancy History  — ``build_occupancy_history``
  * Blocked Rooms      — ``detect_blocked_rooms`` (+ Inactive annotation)
  * Revenue Leakage    — ``portfolio_kpis`` / ``leakage_by_apartment`` on live blocked
  * Recommendations    — existing Recommendation Engine on live inventory

Every page rebuilds from the live data source on load (no manual refresh).

Run:  streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------- #
# Paths — make the analytics package importable.
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

st.set_page_config(page_title="Availability AI", layout="wide")


# --------------------------------------------------------------------------- #
# Live data access — rebuild from DataLoader; never read outputs/*.csv.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Building live analytics from production CSVs…")
def get_live_analytics():
    """Load Q35–Q39 via DataLoader and rebuild all dashboard dataframes once.

    Returns
    -------
    loader, inventory, occupancy_history, blocked
    """
    from blocked_room_detector import detect_blocked_rooms
    from data_loader import DataLoader
    from inventory_views import attach_apartment_status_to_blocked, prepare_inventory
    from preprocessing import build_occupancy_history, build_room_inventory

    loader = DataLoader()
    bookings = loader.bookings()

    tenants = None
    try:
        tenants = loader.tenants()
    except Exception:  # noqa: BLE001
        tenants = None

    beds_master = None
    try:
        beds_master = loader.beds_master()
    except Exception:  # noqa: BLE001
        beds_master = None

    transfers = None
    try:
        transfers = loader.transfers()
    except Exception:  # noqa: BLE001
        transfers = None

    # Bed Map = the single live current-occupancy source for every page.
    bed_map = None
    try:
        bed_map = loader.bed_map()
    except Exception:  # noqa: BLE001
        bed_map = None

    inventory = prepare_inventory(
        build_room_inventory(
            bookings, tenants=tenants, beds_master=beds_master, bed_map=bed_map
        )
    )
    occupancy_history = build_occupancy_history(bookings, transfers=transfers)
    blocked = detect_blocked_rooms(
        inventory, occupancy_history=occupancy_history, bed_map=bed_map
    )
    blocked = attach_apartment_status_to_blocked(blocked, inventory)
    return loader, inventory, occupancy_history, blocked


def get_engine():
    """Compatibility: loader + prepared inventory from the live analytics bundle."""
    loader, inventory, _, _ = get_live_analytics()
    return loader, inventory


def _shared_inventory() -> pd.DataFrame | None:
    """Single source of truth: live prepare_inventory() from DataLoader."""
    try:
        _, inv, _, _ = get_live_analytics()
        if inv is not None and not inv.empty:
            return inv
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to build live inventory: {exc}")
    return None


def _shared_blocked() -> pd.DataFrame | None:
    """Live blocked-rooms dataframe (Inactive → ₹0 leakage)."""
    try:
        _, _, _, blocked = get_live_analytics()
        if blocked is not None:
            return blocked
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to build live blocked rooms: {exc}")
    return None


def _shared_occupancy_history() -> pd.DataFrame | None:
    """Live occupancy history from bookings (Q35)."""
    try:
        _, _, history, _ = get_live_analytics()
        if history is not None:
            return history
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to build live occupancy history: {exc}")
    return None


def _portfolio_from_blocked(blocked: pd.DataFrame):
    """KPIs from an already-annotated live blocked table."""
    from revenue_analytics import leakage_by_apartment, portfolio_kpis

    by_apt = leakage_by_apartment(blocked)
    return portfolio_kpis(blocked, by_apt), by_apt


def _status_badge(status) -> str:
    """Colour badge for Active / Inactive / Live / Not-Active / Unknown."""
    s = str(status or "Unknown").strip()
    n = s.lower().replace("_", "-")
    if n in {"active", "live"}:
        label = "Live" if n == "live" else "Active"
        return f"🟢 {label}"
    if n in {"inactive", "not-active", "not active"}:
        label = "Not-Active" if "not" in n else "Inactive"
        return f"🔴 {label}"
    return f"⚪ {s or 'Unknown'}"


def _prepared_inventory() -> pd.DataFrame | None:
    """Alias for ``_shared_inventory`` (prepare_inventory source of truth)."""
    return _shared_inventory()


@st.cache_data(show_spinner=False)
def india_location_suggestions(query: str) -> list[str]:
    """Fuzzy India city + state suggestions (not limited to tenant cities)."""
    from india_locations import suggest_india_locations

    return suggest_india_locations(query, limit=30)


def run_recommendation(city, bed_type, customer_gender):
    """Delegate to the existing engine — no logic here."""
    from recommendation_engine import recommend_rooms

    loader, inventory = get_engine()
    return recommend_rooms(
        loader,
        city=city,
        bed_type=bed_type or None,
        budget=None,
        gender=customer_gender,
        room_inventory=inventory,
        top_n=3,
    )


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _business_priority(hist_occ) -> str:
    """Display-only business priority from historical occupancy (does NOT affect
    ranking). <70% -> High, 70-85% -> Medium, >85% -> Low."""
    v = _to_float(hist_occ)
    if v is None:
        return "⚪ Unknown"
    if v < 70:
        return "🔴 High"
    if v <= 85:
        return "🟡 Medium"
    return "🟢 Low"


def _kpi_dict(summary: pd.DataFrame | None) -> dict:
    if summary is None or summary.empty:
        return {}
    return dict(zip(summary["metric"], summary["value"]))


def _money(value) -> str:
    f = _to_float(value)
    return "—" if f is None else f"₹{f:,.0f}"


@st.cache_resource(show_spinner=False)
def _beds_master_rate_lookup() -> dict:
    """apartment_code|bed_code → current_rate from live beds_master (display only).

    Uses the same DataLoader instance as get_live_analytics().
    """
    from preprocessing import resolve_beds_master

    try:
        loader, _, _, _ = get_live_analytics()
        raw = loader.beds_master()
    except Exception:  # noqa: BLE001
        return {}
    if raw is None or raw.empty:
        return {}
    resolved, _ = resolve_beds_master(raw, bed_map=loader.bed_map())
    if resolved is None or resolved.empty:
        return {}
    out: dict = {}
    for _, r in resolved.iterrows():
        apt = str(r.get("apartment_code") or "").strip().upper()
        bed = str(r.get("bed_code") or "").strip().upper()
        rate = r.get("current_rate")
        if not apt or not bed:
            continue
        try:
            if rate is None or (isinstance(rate, float) and pd.isna(rate)):
                continue
            out[f"{apt}|{bed}"] = float(rate)
        except (TypeError, ValueError):
            continue
    return out


def _vacant_bed_display_rent(apartment_code, bed_code, fallback=None):
    """Card-only: current_rate for the displayed vacant bed from beds_master."""
    apt = str(apartment_code or "").strip().upper()
    bed = str(bed_code or "").strip().upper()
    if apt and bed:
        rate = _beds_master_rate_lookup().get(f"{apt}|{bed}")
        if rate is not None:
            return rate
    return fallback


def _missing(what: str) -> None:
    st.warning(f"Live {what} is unavailable. Check the production data source.")


# --------------------------------------------------------------------------- #
# Page 0 — Inventory Overview (Active Vacant Beds vs Inactive Inventory)
# --------------------------------------------------------------------------- #
def page_inventory_overview():
    st.header("Inventory Overview")
    st.caption(
        "Active Vacant Beds are recommendable. Inactive Inventory is informational only. "
        "These two lists are never mixed."
    )

    from inventory_views import (
        active_vacant_beds,
        inactive_inventory_summary,
    )

    inv = _shared_inventory()
    if inv is None:
        _missing("inventory")
        return

    vacant = active_vacant_beds(inv)
    inactive_sum = inactive_inventory_summary(inv)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Active Vacant Beds", len(vacant))
    with c2:
        st.metric("Inactive Apartments", len(inactive_sum))

    with st.expander(f"Active Vacant Beds = {len(vacant)}", expanded=True):
        if vacant.empty:
            st.info("No active vacant beds right now.")
        else:
            show = vacant.copy()
            show["Occupancy"] = show["current_occupancy_pct"].map(
                lambda v: f"{v}%" if pd.notna(v) else "—"
            )
            show["Rent"] = show["current_rent"].map(_money)
            st.dataframe(
                show[
                    [
                        "apartment_code",
                        "room_code",
                        "bed_code",
                        "Rent",
                        "Occupancy",
                        "demand_score",
                    ]
                ].rename(
                    columns={
                        "apartment_code": "Apartment",
                        "room_code": "Room",
                        "bed_code": "Bed",
                        "demand_score": "Demand Score",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )

    st.subheader("Inactive Inventory")
    if inactive_sum.empty:
        st.caption("No inactive apartments.")
    else:
        show_i = inactive_sum.copy()
        show_i["Status"] = show_i["apartment_status"].map(_status_badge)
        show_i["Occupancy"] = show_i["current_occupancy_pct"].map(
            lambda v: f"{v}%" if pd.notna(v) else "—"
        )
        st.dataframe(
            show_i[
                ["apartment_code", "Status", "rooms", "beds", "available_beds", "Occupancy"]
            ].rename(
                columns={
                    "apartment_code": "Apartment",
                    "rooms": "Rooms",
                    "beds": "Beds",
                    "available_beds": "Available Beds",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


# --------------------------------------------------------------------------- #
# Page 1 — Customer Recommendation
# --------------------------------------------------------------------------- #
def page_recommendation():
    st.header("Customer Recommendation")
    st.caption(
        "Only ACTIVE vacant beds are considered. Rooms in inactive apartments "
        "are never recommended."
    )

    from inventory_views import (
        active_inventory,
        active_vacant_summary,
        has_same_state_vacant_rooms,
        print_active_vacant_inventory_summary,
        vacant_beds_by_roommate_state,
    )
    from roommate_matching import (
        _states_of,
        occupants_gender_label,
        occupants_state_summary,
    )

    inv = _shared_inventory()
    if inv is None:
        _missing("inventory")
        return

    # Console verification (ACTIVE inventory only).
    print_active_vacant_inventory_summary(inv)

    summary = active_vacant_summary(inv)
    state_counts = vacant_beds_by_roommate_state(inv)

    # --- ACTIVE VACANT BED SUMMARY ---
    st.markdown("## ACTIVE VACANT BED SUMMARY")
    st.caption("Generated only from Active recommendable inventory. Inactive apartments excluded.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Active Vacant Beds", summary["total_active_vacant_beds"])
    m2.metric("Total Active Vacant Rooms", summary["total_active_vacant_rooms"])
    m3.metric(
        "Total Active Apartments with Vacancies",
        summary["total_active_apartments_with_vacancies"],
    )

    st.subheader("Vacant Beds by Current Roommate State")
    st.caption(
        "Active recommendable inventory only. Empty rooms → Vacant / No Occupants."
    )
    if state_counts.empty:
        st.info("No active vacant beds right now.")
    else:
        st.dataframe(state_counts, use_container_width=True, hide_index=True)

    st.divider()

    bed_types = ["Any"]
    if "bed_type" in inv.columns:
        active = active_inventory(inv)
        bed_types += sorted(x for x in active["bed_type"].dropna().unique())

    typed = st.text_input(
        "Customer City or State",
        placeholder="Type city or state — e.g. Sivagangai, Chennai, Tamil Nadu",
        key="reco_city_typed",
    )
    # No preselection: the field starts empty and suggestions appear only once the
    # user starts typing (no default state, no starter list).
    options = [""]
    if typed.strip():
        options = [typed.strip()] + [
            s for s in india_location_suggestions(typed) if s != typed.strip()
        ]

    c1, c2, c3 = st.columns(3)
    with c1:
        picked = st.selectbox(
            "Suggestions (fuzzy)",
            options,
            help="Start typing a city or state to see suggestions.",
        )
    with c2:
        bed_type = st.selectbox("Preferred Bed Type", bed_types)
    with c3:
        customer_gender = st.selectbox(
            "Customer Gender *",
            ["Male", "Female", "Any"],
            help="Mandatory. Filters rooms by inventory Gender Allowed before ranking.",
        )

    if not st.button("Find Rooms", type="primary"):
        return

    city = (picked or typed or "").strip()
    if not city:
        st.error("Please type or select a city / state.")
        return
    if not customer_gender:
        st.error("Customer Gender is required.")
        return

    with st.spinner("Finding the best rooms…"):
        rec = run_recommendation(
            city,
            None if bed_type == "Any" else bed_type,
            customer_gender,
        )

    state = rec.attrs.get("detected_state")
    kind = rec.attrs.get("input_kind", "city")
    st.info(
        f"Matching State: **{state or 'Unknown'}**  ·  "
        f"Customer Gender: **{customer_gender}**  ·  "
        f"Input interpreted as: **{kind}**"
    )

    # Keep summary + state table visible after search (display only).
    with st.expander("ACTIVE VACANT BED SUMMARY (reference)", expanded=False):
        r1, r2, r3 = st.columns(3)
        r1.metric("Total Active Vacant Beds", summary["total_active_vacant_beds"])
        r2.metric("Total Active Vacant Rooms", summary["total_active_vacant_rooms"])
        r3.metric(
            "Total Active Apartments with Vacancies",
            summary["total_active_apartments_with_vacancies"],
        )
        if not state_counts.empty:
            st.dataframe(state_counts, use_container_width=True, hide_index=True)

    # Fallback explanation — one clear message for why these cards are shown.
    req_bt = rec.attrs.get("bed_type_requested")
    bt_fallback = rec.attrs.get("bed_type_fallback")
    bt_shown = rec.attrs.get("bed_type_shown")
    same_in_result = rec.attrs.get("same_state_in_result")
    if bt_fallback and req_bt:
        # Priority 3: requested bed type has no vacancy -> any available type.
        st.warning(
            f"No {req_bt} beds are currently vacant. "
            f"Showing the best available {bt_shown} beds."
        )
    elif req_bt and state and not same_in_result and not rec.empty:
        # Priority 2: requested type available, but no same-state roommate.
        st.warning(
            f"No {req_bt} beds with {state} roommates are available. "
            f"Showing the best available {req_bt} beds."
        )
    elif state and not has_same_state_vacant_rooms(inv, state):
        # Any-bed-type query with no same-state roommate anywhere.
        st.warning(
            f"No active vacant room currently has roommates from **{state}**."
        )
        st.caption("Active vacant beds by roommate state:")
        if not state_counts.empty:
            st.dataframe(state_counts, use_container_width=True, hide_index=True)

    if rec.empty:
        st.info(
            "No active vacant beds are available to recommend right now. "
            "See the Active Vacant Bed Summary above."
        )
        return

    # Inventory lookup for card display fields (does not change ranking).
    inv_by_room = {
        (str(r["apartment_code"]), str(r["room_code"])): r
        for _, r in inv.iterrows()
    }

    cols = st.columns(len(rec))
    for col, (_, row) in zip(cols, rec.iterrows()):
        key = (str(row["apartment_code"]), str(row["room_code"]))
        src = inv_by_room.get(key)
        occupied = int(src["occupied_beds"]) if src is not None else 0
        occ_states = (
            _states_of(src["occupant_states"]) if src is not None else []
        )
        occ_genders = src["occupant_genders"] if src is not None else []
        gender_allowed = (
            str(src.get("gender_allowed")).strip()
            if src is not None and pd.notna(src.get("gender_allowed"))
            else (row.get("gender_allowed") or "Unknown")
        )
        if not gender_allowed or gender_allowed.lower() in ("nan", "none"):
            gender_allowed = row.get("gender_allowed") or "Unknown"
        apt_status = (
            str(src.get("apartment_status", "Active"))
            if src is not None
            else "Active"
        )
        with col:
            with st.container(border=True):
                st.markdown(f"### {row['apartment_code']} · {row['room_code']}")
                st.markdown(f"**Apartment:** {row['apartment_code']}")
                st.markdown(f"**Room:** {row['room_code']}")
                st.markdown(
                    f"**Vacant Bed:** {row['bed_code']} · {row['bed_type']}"
                )
                st.markdown(
                    f"**Current Occupants:** {occupied if occupied > 0 else 'None'}"
                )
                st.markdown(
                    f"**Current Occupant States:** "
                    f"{occupants_state_summary(occ_states)}"
                )
                st.markdown(f"**Gender Allowed:** {gender_allowed}")
                if src is not None and pd.notna(src.get("toilet_type")):
                    st.markdown(f"**Toilet Type:** {src.get('toilet_type')}")
                st.markdown(
                    f"**Current Occupant Gender:** "
                    f"{occupants_gender_label(occ_genders)}"
                )
                st.markdown(
                    f"**Apartment Status:** {_status_badge(apt_status)}"
                )
                st.markdown(f"**Matching State:** {state or 'Unknown'}")
                same = str(row.get("same_state_roommate", "No")).strip() or "No"
                st.markdown(f"**Same-state roommate:** {same}")
                note = str(row.get("same_state_note") or "").strip()
                if same != "Yes" and note:
                    st.caption(note)
                st.markdown(
                    f"**Similarity:** {row.get('match_similarity', 'Unknown')}"
                )
                # Display only: bind Current Rent to the Vacant Bed shown above,
                # not room-level inventory/ranking monthly_rent (max rate).
                display_rent = _vacant_bed_display_rent(
                    row["apartment_code"],
                    row.get("bed_code"),
                    fallback=row.get("monthly_rent"),
                )
                st.metric("Current Rent", _money(display_rent))
                m1, m2 = st.columns(2)
                m1.metric("Occupancy", f"{row['current_occupancy_pct']}%")
                m2.metric("Demand", f"{row['demand_score']}")
                _hist_occ = src.get("historical_occupancy_pct") if src is not None else None
                st.markdown(
                    f"**Business Priority:** {_business_priority(_hist_occ)}"
                )
                st.markdown(f"**Why:** {row['reason']}")


# --------------------------------------------------------------------------- #
# Page 2 — Blocked Rooms
# --------------------------------------------------------------------------- #
def page_blocked_rooms():
    st.header("Blocked Rooms")
    st.caption(
        "Active blocked rooms use Critical / Delayed / Normal / Unknown. "
        "Inactive apartments are labeled Inactive with ₹0 leakage."
    )

    show = _shared_blocked()
    if show is None:
        _missing("blocked rooms")
        return

    # KPIs from revenue analytics (Inactive leakage already forced to 0 there).
    kpis_df, _ = _portfolio_from_blocked(show)
    kpis = _kpi_dict(kpis_df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Critical Rooms", kpis.get("total_critical_rooms", "—"))
    c2.metric("Delayed Rooms", kpis.get("total_delayed_rooms", "—"))
    c3.metric("Unknown Rooms", kpis.get("total_unknown_rooms", "—"))
    c4.metric("Total Est. Revenue Leakage", _money(kpis.get("total_estimated_revenue_leakage")))

    st.divider()

    show = show.copy()
    show["Apartment Status"] = show["apartment_status"].map(_status_badge)

    f1, f2, f3 = st.columns(3)
    with f1:
        apts = ["All"] + sorted(x for x in show["apartment_code"].dropna().unique())
        apt = st.selectbox("Filter by Apartment", apts)
    with f2:
        vac_statuses = ["All"] + sorted(
            x for x in show["vacancy_status"].dropna().unique()
        )
        vac_status = st.selectbox("Filter by Vacancy Status", vac_statuses)
    with f3:
        apt_status_choice = st.selectbox(
            "Apartment Status", ["All", "Active", "Inactive"], key="blk_apt_status"
        )

    view = show
    if apt != "All":
        view = view[view["apartment_code"] == apt]
    if vac_status != "All":
        view = view[view["vacancy_status"] == vac_status]
    if apt_status_choice != "All":
        view = view[view["apartment_status"] == apt_status_choice]

    display_cols = {
        "apartment_code": "Apartment",
        "room_code": "Room",
        "bed_code": "Bed",
        "Apartment Status": "Apartment Status",
        "current_vacant_days": "Current Vacant Days",
        "expected_fill_days": "Expected Fill Days",
        "vacancy_status": "Vacancy Status",
        "estimated_revenue_loss": "Estimated Revenue Leakage",
        "reason": "Reason",
    }
    cols = [c for c in display_cols if c in view.columns]
    render_df = view[cols].rename(columns=display_cols)
    st.dataframe(
        render_df,
        use_container_width=True,
        hide_index=True,
        key="blocked_rooms_table",
    )


# --------------------------------------------------------------------------- #
# Page 3 — Revenue Leakage Dashboard
# --------------------------------------------------------------------------- #
def page_revenue_leakage():
    st.header("Revenue Leakage Dashboard")
    st.caption(
        "Inactive apartments are excluded entirely (same prepare_inventory source "
        "as Customer Recommendation)."
    )

    blocked = _shared_blocked()
    if blocked is None:
        _missing("blocked rooms")
        return

    # Exclude Inactive apartments completely from leakage KPIs and tables.
    active_blocked = blocked[
        blocked["apartment_status"].astype(str) != "Inactive"
    ].copy()
    summary, by_apt = _portfolio_from_blocked(active_blocked)

    kpis = _kpi_dict(summary)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Estimated Leakage", _money(kpis.get("total_estimated_revenue_leakage")))
    c2.metric("Highest Leakage Apartment",
              f"{kpis.get('highest_loss_apartment', '—')}")
    c2.caption(_money(kpis.get("highest_loss_apartment_value")))
    c3.metric("Highest Leakage Room", f"{kpis.get('highest_loss_room', '—')}")
    c3.caption(_money(kpis.get("highest_loss_room_value")))

    c4, c5, c6 = st.columns(3)
    c4.metric("Average Vacancy Days", kpis.get("average_vacancy_days", "—"))
    c5.metric("Vacant Beds", kpis.get("total_vacant_beds", "—"))
    c6.metric("Critical Rooms", kpis.get("total_critical_rooms", "—"))

    st.divider()
    st.subheader("Revenue Leakage by Apartment (Active only)")
    view = by_apt.copy()
    if "lifecycle_status" in view.columns:
        view = view[view["lifecycle_status"].astype(str) != "Inactive"]
        view = view.copy()
        view["Status"] = view["lifecycle_status"].map(_status_badge)
    display = view.rename(columns={
        "apartment_code": "Apartment",
        "Status": "Status",
        "estimated_revenue_loss": "Estimated Revenue Leakage",
        "critical_rooms": "Critical Rooms",
        "delayed_rooms": "Delayed Rooms",
        "inactive_rooms": "Inactive Rooms",
        "vacant_beds": "Vacant Beds",
        "avg_vacancy_days": "Avg Vacancy Days",
    })
    hide = [c for c in display.columns if c == "lifecycle_status"]
    st.dataframe(
        display.drop(columns=hide, errors="ignore"),
        use_container_width=True,
        hide_index=True,
    )

    chart_src = by_apt
    if "lifecycle_status" in chart_src.columns:
        chart_src = chart_src[chart_src["lifecycle_status"].astype(str) != "Inactive"]
    g1, g2 = st.columns(2)
    if "estimated_revenue_loss" in chart_src.columns and not chart_src.empty:
        with g1:
            st.caption("Revenue Leakage by Apartment (Active only)")
            st.bar_chart(chart_src.set_index("apartment_code")["estimated_revenue_loss"])
    if "critical_rooms" in chart_src.columns and not chart_src.empty:
        with g2:
            st.caption("Critical Rooms by Apartment (Active only)")
            st.bar_chart(chart_src.set_index("apartment_code")["critical_rooms"])


# --------------------------------------------------------------------------- #
# Page 4 — Occupancy Analytics
# --------------------------------------------------------------------------- #
def page_occupancy():
    st.header("Occupancy Analytics")
    st.caption(
        "Active apartments by default. Enable Show Inactive to view inactive inventory."
    )

    from inventory_views import active_inventory, inactive_inventory_summary, inactive_inventory

    inv = _shared_inventory()
    if inv is None:
        _missing("inventory")
        return

    show_inactive = st.checkbox("Show Inactive", value=False, key="occ_show_inactive")

    active = active_inventory(inv)
    inactive_rooms = inactive_inventory(inv)
    inactive_sum = inactive_inventory_summary(inv)

    f1, f2, f3 = st.columns(3)
    with f1:
        apts = sorted(x for x in active["apartment_code"].dropna().unique()) if not active.empty else []
        sel_apts = st.multiselect("Apartment (Active)", apts)
    with f2:
        types = sorted(x for x in active["bed_type"].dropna().unique()) if "bed_type" in active else []
        sel_types = st.multiselect("Room Type", types)
    with f3:
        occ_range = st.slider("Occupancy Range (%)", 0, 100, (0, 100))

    view = active.copy()
    if sel_apts:
        view = view[view["apartment_code"].isin(sel_apts)]
    if sel_types:
        view = view[view["bed_type"].isin(sel_types)]
    if "current_occupancy_pct" in view.columns:
        view = view[view["current_occupancy_pct"].between(occ_range[0], occ_range[1])]

    st.subheader("ACTIVE APARTMENTS")
    st.caption(f"{len(view)} active rooms")
    show = view.copy()
    if "apartment_status" in show.columns:
        show["Apartment Status"] = show["apartment_status"].map(_status_badge)
    display_cols = {
        "apartment_code": "Apartment",
        "room_code": "Room",
        "Apartment Status": "Apartment Status",
        "bed_type": "Room Type",
        "gender_allowed": "Gender Allowed",
        "toilet_type": "Toilet Type",
        "bed_lifecycle_status": "Bed Lifecycle",
        "current_occupancy_pct": "Current Occupancy %",
        "recent_occupancy_pct": "Recent Occupancy %",
        "historical_occupancy_pct": "Historical Occupancy %",
        "demand_score": "Demand Score",
        "current_revenue": "Current Revenue",
    }
    cols = [c for c in display_cols if c in show.columns]
    st.dataframe(
        show[cols].rename(columns=display_cols),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("INACTIVE APARTMENTS")
    if not show_inactive:
        st.caption(
            f"{len(inactive_sum)} inactive apartment(s) hidden. "
            "Enable **Show Inactive** to view."
        )
        return
    if inactive_sum.empty:
        st.caption("No inactive apartments.")
    else:
        show_i = inactive_sum.copy()
        show_i["Status"] = show_i["apartment_status"].map(_status_badge)
        show_i["Occupancy"] = show_i["current_occupancy_pct"].map(
            lambda v: f"{v}%" if pd.notna(v) else "—"
        )
        st.dataframe(
            show_i[
                ["apartment_code", "Status", "Occupancy", "available_beds", "rooms", "beds"]
            ].rename(
                columns={
                    "apartment_code": "Apartment",
                    "available_beds": "Available Beds",
                    "rooms": "Rooms",
                    "beds": "Beds",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("Inactive room detail"):
            detail = inactive_rooms.copy()
            detail["Apartment Status"] = detail["apartment_status"].map(_status_badge)
            detail["Room Status"] = detail["room_status"].map(_status_badge)
            dcols = {
                "apartment_code": "Apartment",
                "room_code": "Room",
                "Apartment Status": "Apartment Status",
                "Room Status": "Room Status",
                "gender_allowed": "Gender Allowed",
                "toilet_type": "Toilet Type",
                "bed_status": "Bed Status",
                "current_occupancy_pct": "Occupancy %",
                "available_beds": "Available Beds",
            }
            dc = [c for c in dcols if c in detail.columns]
            st.dataframe(
                detail[dc].rename(columns=dcols),
                use_container_width=True,
                hide_index=True,
            )


# --------------------------------------------------------------------------- #
# Page 5 — Room Search
# --------------------------------------------------------------------------- #
def page_room_search():
    st.header("Room Search")
    st.caption(
        "Search Active and Inactive inventory. Inactive apartments are shown "
        "with Inactive status — recommendation is disabled for them."
    )

    from roommate_matching import occupants_gender_label, occupants_state_summary

    inv = _shared_inventory()
    if inv is None:
        _missing("inventory")
        return
    history = _shared_occupancy_history()

    f1, f2, f3 = st.columns(3)
    with f1:
        apts = ["All"] + sorted(x for x in inv["apartment_code"].dropna().unique())
        apt = st.selectbox("Apartment Code", apts)
    with f2:
        room_q = st.text_input("Room Code contains")
    with f3:
        bed_q = st.text_input("Bed Code contains")

    rooms = inv.copy()
    if "occupant_genders" in rooms.columns:
        rooms["Current Occupants Gender"] = rooms["occupant_genders"].map(
            occupants_gender_label
        )
    else:
        rooms["Current Occupants Gender"] = "None"
    if "occupant_states" in rooms.columns:
        rooms["Current Occupants State"] = rooms["occupant_states"].map(
            occupants_state_summary
        )
    else:
        rooms["Current Occupants State"] = "None"

    if apt != "All":
        rooms = rooms[rooms["apartment_code"] == apt]
    if room_q.strip():
        rooms = rooms[
            rooms["room_code"].astype(str).str.contains(room_q.strip(), case=False, na=False)
        ]

    # Flag when the search hits inactive inventory.
    if not rooms.empty and (rooms["apartment_status"] == "Inactive").all():
        st.warning(
            "This apartment is **Inactive**. Recommendation is disabled. "
            "Informational view only."
        )
    elif not rooms.empty and (rooms["apartment_status"] == "Inactive").any():
        st.warning("Some results are Inactive apartments — recommendation is disabled for those.")

    st.subheader("Room summary")
    show = rooms.copy()
    show["Apartment Status"] = show["apartment_status"].map(_status_badge)
    show["Room Status"] = show["room_status"].map(_status_badge)
    show["Bed Status"] = show["bed_status"].map(_status_badge)
    show["Recommendable"] = show["recommendable"].map(
        lambda v: "Yes" if bool(v) else "No — Inactive"
    )
    # Display map: use badge columns for status; do not also rename raw bed_status
    # (that produced duplicate "Bed Status" after rename).
    room_cols = {
        "apartment_code": "Apartment",
        "room_code": "Room",
        "Apartment Status": "Apartment Status",
        "Room Status": "Room Status",
        "Bed Status": "Bed Status",
        "Recommendable": "Recommendable",
        "gender_allowed": "Gender Allowed",
        "toilet_type": "Toilet Type",
        "bed_lifecycle_status": "Bed Lifecycle",
        "Current Occupants Gender": "Current Occupants Gender",
        "Current Occupants State": "Current Occupants State",
        "bed_type": "Room Type",
        "current_occupancy_pct": "Occupancy %",
        "available_beds": "Available Beds",
        "demand_score": "Demand Score",
        "current_rent": "Current Rent",
    }
    rcols = [c for c in room_cols if c in show.columns]
    rcols = list(dict.fromkeys(rcols))
    display = show.loc[:, ~show.columns.duplicated()][rcols].rename(columns=room_cols)
    display = display.loc[:, ~display.columns.duplicated()]
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
    )

    room_codes = set(rooms["room_code"].astype(str))

    st.subheader("Current tenants")
    if history is not None and "is_active" in history.columns:
        is_active = history["is_active"].astype(str).str.lower().isin(["true", "1"])
        active = history[is_active].copy()
        active = active[active["room_code"].astype(str).isin(room_codes)]
        if bed_q.strip():
            active = active[
                active["bed_code"].astype(str).str.contains(bed_q.strip(), case=False, na=False)
            ]
        tenant_cols = {
            "apartment_code": "Apartment",
            "room_code": "Room",
            "bed_code": "Bed",
            "full_name": "Current Tenant",
            "onboarding_date": "Since",
        }
        tcols = [c for c in tenant_cols if c in active.columns]
        st.dataframe(
            active[tcols].rename(columns=tenant_cols),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Live occupancy history unavailable.")

    st.subheader("Vacant beds")
    # ALL currently vacant beds from the shared Bed-Map inventory — NOT the
    # Blocked dataframe. Includes fully-vacant rooms (e.g. B33, C31). Blocked
    # Rooms is occupancy-blockage only and must never drive this general list.
    from inventory_views import active_vacant_beds

    vb = active_vacant_beds(inv)
    if vb is not None and not vb.empty:
        vb = vb[vb["room_code"].astype(str).isin(room_codes)].copy()
        if bed_q.strip():
            vb = vb[
                vb["bed_code"].astype(str).str.contains(bed_q.strip(), case=False, na=False)
            ]
    if vb is None or vb.empty:
        st.caption("No vacant beds for this selection.")
    else:
        vb_show = vb.copy()
        vb_show["Rent"] = vb_show["current_rent"].map(_money)
        vb_show["Occupancy"] = vb_show["current_occupancy_pct"].map(
            lambda v: f"{v}%" if pd.notna(v) else "—"
        )
        if "bed_status" in vb_show.columns:
            vb_show["Bed Status"] = vb_show["bed_status"].map(_status_badge)
        vac_cols = {
            "apartment_code": "Apartment",
            "room_code": "Room",
            "bed_code": "Bed",
            "bed_type": "Room Type",
            "Rent": "Rent",
            "Occupancy": "Occupancy",
            "Bed Status": "Bed Status",
            "demand_score": "Demand Score",
        }
        vcols = [c for c in vac_cols if c in vb_show.columns]
        st.dataframe(
            vb_show[vcols].rename(columns=vac_cols),
            use_container_width=True,
            hide_index=True,
            key="room_search_vacant_beds",
        )


# --------------------------------------------------------------------------- #
# Page 7 — Maintenance Intelligence (Phase 2: descriptive analytics only)
# Additive: reuses the integrated maintenance datasets. Does NOT touch any of
# the six availability pages or their data.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Building maintenance model…")
def _maintenance_base():
    """Build the (expensive) maintenance model + filter options ONCE (cached)."""
    from data_loader import DataLoader
    import maintenance_analytics as MA

    loader = DataLoader()
    return loader, MA.build_base(loader)


@st.cache_resource(show_spinner="Scoring assets…")
def _maintenance_risk():
    """Rule-based risk scores + dashboard aggregates, built ONCE (cached)."""
    from data_loader import DataLoader
    import maintenance_risk as MR

    scores = MR.build_risk_scores(DataLoader())
    return scores, MR.risk_summary(scores)


def _dl(df: pd.DataFrame, label: str, key: str):
    """CSV export button for one analytics table."""
    if df is None or df.empty:
        return
    st.download_button(
        "⬇ CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name=f"maintenance_{key}.csv",
        mime="text/csv",
        key=f"dl_{key}",
    )


def _bar(df: pd.DataFrame, index_col: str, value_col: str, caption: str, key: str = None):
    """Labelled bar chart + CSV export, or a placeholder when empty."""
    st.caption(caption)
    if df is None or df.empty or index_col not in df.columns or value_col not in df.columns:
        st.caption("_No data available._")
        return
    st.bar_chart(df.set_index(index_col)[value_col])
    if key:
        _dl(df, caption, key)


def _table(df: pd.DataFrame, caption: str, key: str, empty_msg: str = "_No data._"):
    """DataFrame + CSV export, or a placeholder when empty."""
    st.caption(caption)
    if df is None or df.empty:
        st.caption(empty_msg)
        return
    st.dataframe(df, use_container_width=True, hide_index=True)
    _dl(df, caption, key)


def page_maintenance_intelligence():
    import maintenance_analytics as MA

    st.header("🛠 Maintenance Intelligence")
    st.caption(
        "Descriptive analytics from the integrated maintenance export — tickets, "
        "assets, vendors, purchases and costs. Read-only; independent of the "
        "availability pages."
    )
    try:
        loader, base = _maintenance_base()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to build maintenance analytics: {exc}")
        return
    opts = base["options"]

    # ---- Filters ---------------------------------------------------------- #
    with st.expander("🔎 Filters", expanded=True):
        f1 = st.columns(3)
        prop = f1[0].selectbox("Property", opts["property"])
        apt = f1[1].selectbox("Apartment", opts["apartment"])
        atype = f1[2].selectbox("Asset Type", opts["asset_type"])
        f2 = st.columns(3)
        itype = f2[0].selectbox("Issue Type", opts["issue_type"])
        vend = f2[1].selectbox("Vendor", opts["vendor"])
        dmin, dmax = opts.get("date_min"), opts.get("date_max")
        date_range = None
        if dmin is not None and dmax is not None:
            date_range = f2[2].date_input(
                "Date Range",
                value=(dmin.date(), dmax.date()),
                min_value=dmin.date(),
                max_value=dmax.date(),
            )
    filters = {"property": prop, "apartment": apt, "asset_type": atype,
               "issue_type": itype, "vendor": vend}
    if isinstance(date_range, (tuple, list)) and len(date_range) == 2:
        filters["date_from"] = str(date_range[0])
        filters["date_to"] = str(date_range[1])

    A = MA.compute_all(loader, filters=filters, base=base)
    k, dq = A["kpis"], A["dq"]

    # ---- 1. KPI cards ----------------------------------------------------- #
    st.subheader("1 · Key Metrics")
    r1 = st.columns(4)
    r1[0].metric("Total Tickets", f"{k['total_tickets']:,}")
    r1[1].metric("Open Tickets", f"{k['open_tickets']:,}")
    r1[2].metric("Resolved", f"{k['resolved_tickets']:,}")
    r1[3].metric("Resolution Rate", f"{k['resolution_rate_pct']}%")
    r2 = st.columns(4)
    r2[0].metric("Avg Resolution (hrs)", k["avg_resolution_hours"] if k["avg_resolution_hours"] is not None else "—")
    r2[1].metric("Total Maintenance Cost", _money(k["total_maintenance_cost"]))
    r2[2].metric("Total Assets", f"{k['total_assets']:,}")
    r2[3].metric("Assets In Service", f"{k['assets_in_service']:,}")
    r3 = st.columns(4)
    r3[0].metric("Active Vendors", f"{k['active_vendors']:,}")
    r3[1].metric("Purchase Value", _money(k["total_purchase_value"]))
    r3[2].metric("Low-Stock Items", f"{k['low_stock_items']:,}")
    st.divider()

    # ---- Data-quality panel (surfaced, not hidden) ----------------------- #
    st.subheader("⚠ Data Quality")
    st.caption("Source data gaps are shown here — not silently dropped. These "
               "cap how far predictive analytics can go (Phase 3).")
    d1 = st.columns(4)
    d1[0].metric("Assets w/o Purchase Date", f"{dq['assets_without_purchase_date']:,}")
    d1[1].metric("Assets w/o Warranty", f"{dq['assets_without_warranty']:,}")
    d1[2].metric("Tickets w/o Apartment/Asset", f"{dq['tickets_without_apartment']:,}")
    d1[3].metric("Tickets Unknown Type", f"{dq['tickets_unknown_type']:,}")
    d2 = st.columns(4)
    d2[0].metric("Invalid Vendor Names", f"{dq['invalid_vendor_count']:,}")
    d2[1].metric("Outlier Purchase Costs", f"{dq['outlier_purchase_count']:,}")
    if dq["invalid_vendor_names"]:
        d2[2].caption("Invalid vendors: " + ", ".join(dq["invalid_vendor_names"][:8]))
    with st.expander("Inspect data-quality issues"):
        _table(dq["tables"]["assets_without_purchase_date"], "Assets missing purchase date", "dq_no_purchase_date")
        _table(dq["tables"]["assets_without_warranty"], "Assets missing warranty", "dq_no_warranty")
        _table(dq["tables"]["tickets_unknown_type"], "Tickets with unknown maintenance type", "dq_unknown_type")
        _table(dq["tables"]["outlier_purchases"], "Outlier purchase costs (IQR fence)", "dq_outlier_purchases")
    st.divider()

    # ---- 2. Ticket analytics --------------------------------------------- #
    st.subheader("2 · Ticket Analytics")
    t = A["tickets"]
    g = st.columns(2)
    with g[0]:
        _bar(t["by_status"], "status", "count", "Tickets by Current Status", "ticket_status")
    with g[1]:
        _bar(t["by_type"], "maintenance_type", "count", "Tickets by Maintenance Type", "ticket_type")
    _bar(t["over_time"], "month", "tickets", "Tickets Opened per Month", "ticket_over_time")
    g2 = st.columns(2)
    with g2[0]:
        _bar(t["top_apartments"], "apartment_code", "count", "Top Apartments by Ticket Count", "ticket_top_apts")
    with g2[1]:
        res = t["resolution"]
        st.caption("Resolution Time")
        rc = st.columns(3)
        rc[0].metric("Avg (hrs)", res.get("avg_hours") if res.get("avg_hours") is not None else "—")
        rc[1].metric("Median (hrs)", res.get("median_hours") if res.get("median_hours") is not None else "—")
        rc[2].metric("Max (hrs)", res.get("max_hours") if res.get("max_hours") is not None else "—")
        _table(t["resolution_by_type"], "Avg Resolution by Type", "ticket_res_by_type", "_No resolved tickets._")
    st.divider()

    # ---- 3. Asset analytics ---------------------------------------------- #
    st.subheader("3 · Asset Analytics")
    a = A["assets"]
    g3 = st.columns(2)
    with g3[0]:
        _bar(a["by_type"], "asset_type", "count", "Assets by Type", "asset_type")
    with g3[1]:
        _bar(a["by_condition"], "condition", "count", "Assets by Condition", "asset_condition")
    g4 = st.columns(2)
    with g4[0]:
        _bar(a["age_buckets"], "age", "count", "Asset Age Distribution", "asset_age")
    with g4[1]:
        _bar(a["warranty"], "warranty", "count", "Warranty Status", "asset_warranty")
    _bar(a["by_apartment"], "apartment_code", "count", "Assets by Apartment", "asset_by_apt")
    _table(a["service_due"], "Assets Due for Service (next 30 days / overdue)", "asset_service_due",
           "_No assets due within 30 days (limited by available purchase dates)._")
    st.divider()

    # ---- 4. Vendor analytics --------------------------------------------- #
    st.subheader("4 · Vendor Analytics")
    v = A["vendors"]
    g5 = st.columns(2)
    with g5[0]:
        _bar(v["spend"].head(15), "vendor_name", "total_spend", "Spend by Vendor", "vendor_spend")
    with g5[1]:
        _table(v["ratings"], "Vendor Ratings (master)", "vendor_ratings", "_No vendor ratings._")
    _bar(v["tickets"].head(15), "vendor_name", "cost_lines", "Maintenance Cost Lines by Vendor", "vendor_cost_lines")
    st.divider()

    # ---- 5. Purchase & inventory analytics ------------------------------- #
    st.subheader("5 · Purchase & Inventory Analytics")
    p = A["purchases"]
    g6 = st.columns(2)
    with g6[0]:
        _bar(p["by_item"], "item", "total_cost", "Purchase Spend by Item", "purchase_by_item")
    with g6[1]:
        _bar(p["by_vendor"].head(15), "vendor", "purchase_cost", "Purchase Spend by Vendor", "purchase_by_vendor")
    _bar(p["over_time"], "month", "purchase_cost", "Purchase Spend per Month", "purchase_over_time")
    _table(p["stock"], "Stock vs Minimum Level", "stock_levels", "_No stock data._")
    _table(p["low_stock"], "Low-Stock Items (available < minimum)", "low_stock", "_No items below minimum stock._")
    st.divider()

    # ---- 6. Maintenance cost analytics ----------------------------------- #
    st.subheader("6 · Maintenance Cost Analytics")
    c = A["costs"]
    st.metric("Total Maintenance Cost (filtered tickets)", _money(c["total"]))
    g7 = st.columns(2)
    with g7[0]:
        _bar(c["by_type"], "t", "total_cost", "Cost by Maintenance Type", "cost_by_type")
    with g7[1]:
        _bar(c["parts_vs_labour"], "component", "cost", "Parts vs Labour Cost", "cost_parts_labour")
    _bar(c["by_apartment"], "a", "total_cost", "Cost by Apartment", "cost_by_apt")
    _bar(c["over_time"], "month", "cost", "Maintenance Cost per Month", "cost_over_time")
    _table(c["per_ticket"], "Top Tickets by Cost", "cost_per_ticket", "_No ticket costs._")
    st.caption(
        "Note: values reflect the source export as-is; the Data Quality panel above "
        "surfaces the known source gaps (missing dates/warranty, unknown types, "
        "invalid vendors, outlier costs)."
    )
    st.divider()

    # ---- 7. Predictive Maintenance (rule-based, NOT ML) ------------------ #
    st.subheader("7 · Predictive Maintenance")
    st.info(
        "ℹ️ **Rule-based** predictive engine — asset risk is scored by a "
        "transparent weighted-rule model (see `maintenance_risk.py`), **not** a "
        "machine-learning model. Per-asset **confidence** reflects how much source "
        "data was available; sparse assets score with lower confidence."
    )
    try:
        _scores, rs = _maintenance_risk()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to build risk scores: {exc}")
        rs = None
    if rs:
        rk = rs["kpis"]
        _H = "Rule-based risk engine — not machine learning."
        rr1 = st.columns(3)
        rr1[0].metric("Critical Assets", f"{rk['critical_assets']:,}", help=_H)
        rr1[1].metric("High-Risk Assets", f"{rk['high_risk_assets']:,}", help=_H)
        rr1[2].metric("Due for Service", f"{rk['due_for_service']:,}",
                      help="Assets past their maintenance-cycle service date (rule-based).")
        rr2 = st.columns(3)
        rr2[0].metric("Replacement Candidates", f"{rk['replacement_candidates']:,}",
                      help="Recommended action = Replace (rule-based; Critical + end-of-life).")
        rr2[1].metric("Average Risk Score", rk["avg_risk_score"], help=_H)
        rr2[2].metric("Average Confidence", f"{rk['avg_confidence']}%",
                      help="Mean per-asset confidence; low = sparse source data, not low risk.")

        rcw = st.columns(2)
        with rcw[0]:
            _bar(rs["distribution"], "risk_level", "count", "Risk Distribution", "risk_distribution")
        with rcw[1]:
            _bar(rs["confidence_buckets"], "confidence", "count", "Confidence Distribution", "risk_confidence")
        rcw2 = st.columns(2)
        with rcw2[0]:
            _bar(rs["by_type"], "asset_type", "avg_risk_score", "Avg Risk Score by Asset Type", "risk_by_type")
        with rcw2[1]:
            _bar(rs["by_apartment"], "apartment_code", "avg_risk_score", "Avg Risk Score by Apartment", "risk_by_apartment")

        _table(rs["high_risk"], "High-Risk Assets (High + Critical)", "risk_high_assets",
               "_No high-risk assets._")
        _table(rs["due_for_service"], "Assets Due for Service", "risk_due_service",
               "_No assets currently due for service._")
        _table(rs["replacement"], "Replacement Candidates", "risk_replacement",
               "_No replacement candidates (no asset reached Critical + end-of-life)._")
        st.caption(
            "Rule-based predictive engine (Phase 3B) — deterministic weighted rules, "
            "not machine learning. Scoring formula and factor weights: `maintenance_risk.py`."
        )


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Page 8 — Asset Predictive Analytics (ticket-centric engine, from scratch)
# Standalone; reads only; no DB writes. Uses src/asset_engine.py exclusively.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Running ticket-centric asset engine…")
def _asset_engine():
    from data_loader import DataLoader
    import asset_engine as AE

    loader = DataLoader()
    built = AE.build(loader)
    room_intel = AE.room_intelligence(loader, built)
    type_intel = AE.asset_type_intelligence(loader, built)
    bv = AE.business_views(loader, built)
    planner = AE.replacement_planner(loader, built)
    report_card = AE.apartment_report_card(built, bv)
    roi = AE.asset_roi(loader, built, bv)
    return built, {
        "room_risk": AE.room_risk(built),
        "type_risk": AE.type_risk(built),
        "charts": AE.charts(built),
        "exports": AE.exports(built),
        "coverage": AE.coverage(loader, built),
        "room_intel": room_intel,
        "type_intel": type_intel,
        "executive": AE.executive(loader, built, room_intel, type_intel),
        "business": bv,
        # extension 3 — management/business planning
        "budget": AE.budget_forecast(loader, built, bv),
        "planner": planner,
        "workload": AE.workload_forecast(built),
        "roi": roi,
        "report_card": report_card,
        "performance": AE.asset_performance(built, bv),
        "recommendations": AE.business_recommendations(built, bv, planner, report_card, roi),
        # extension 4 — SLA, brand, purchase
        "sla": AE.sla_dashboard(loader, built),
        "brand": AE.brand_analysis(built, bv),
        "purchase": AE.purchase_recommendation(loader, built, bv),
    }


def _risk_badge(level) -> str:
    return {"Critical": "🔴 Critical", "High": "🟠 High",
            "Medium": "🟡 Medium", "Low": "🟢 Low"}.get(str(level), str(level))


def _management_report_html(ex, bf, pl, recs) -> str:
    """One-click monthly management summary (printable HTML)."""
    bvv = ex.get("business") or {}
    ek = bvv.get("exec_kpis", {})
    cal = bvv.get("calendar", {})

    def tbl(df, n=15):
        return df.head(n).to_html(index=False, border=0) if (df is not None and not df.empty) else "<p><i>None.</i></p>"

    kpi_rows = "".join(f"<tr><td>{k.replace('_', ' ').title()}</td><td>{v}</td></tr>" for k, v in ek.items())
    rec_html = ""
    if recs is not None and not recs.empty:
        for _, r in recs.iterrows():
            rec_html += (f"<li><b>[{r['priority']}] {r['recommendation']}</b><br>"
                         f"WHY: {r['why']}<br>EXPECTED IMPACT: {r['expected_impact']}</li>")
    cal_rows = "".join(f"<tr><td>{lbl}</td><td>{len(cal.get(key, []))}</td></tr>"
                       for lbl, key in [("Today/Overdue", "today"), ("This Week", "this_week"),
                                        ("This Month", "this_month"), ("Next Month", "next_month")])
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Maintenance Management Report</title>
<style>body{{font-family:Arial,sans-serif;margin:32px;color:#222}}
h1{{border-bottom:2px solid #444}}h2{{margin-top:28px;color:#333}}
table{{border-collapse:collapse;margin:8px 0}}td,th{{border:1px solid #ccc;padding:4px 10px;text-align:left}}
li{{margin:8px 0}}@media print{{a,button{{display:none}}}}</style></head><body>
<h1>🔧 Maintenance Management Report</h1>
<p>Ticket-centric predictive maintenance summary. Statistical (not ML). Read-only.</p>
<h2>Key Metrics</h2><table>{kpi_rows}</table>
<h2>Maintenance Budget Forecast</h2>{tbl(bf.get('windows'))}
<h2>Replacement Plan</h2>{tbl(pl.get('summary'))}
<h2>Maintenance Calendar</h2><table><tr><th>Window</th><th>Assets Due</th></tr>{cal_rows}</table>
<h2>Top Risky Apartments</h2>{tbl(bvv.get('apartment_health'), 10)}
<h2>Business Recommendations</h2><ul>{rec_html or '<li>None.</li>'}</ul>
</body></html>"""


def page_asset_predictive():
    st.header("🔧 Asset Predictive Analytics")
    st.info(
        "**Ticket-centric engine.** Every maintenance ticket is attributed to a physical "
        "asset (direct `asset_id`, else Apartment/Room → Asset Allocation → Issue Type → "
        "Asset Type), joined to allocations + assets, and aged by purchase date (else "
        "earliest allocation date). Health, risk, maintenance-due, replacement, failure "
        "trend, room risk and asset-type risk are computed from ticket frequency, recent "
        "failures, age, expected life and maintenance cycle. Statistical (not ML). "
        "Read-only — no database records are modified."
    )
    try:
        built, ex = _asset_engine()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to run asset engine: {exc}")
        return
    A, k = built["assets"], built["kpis"]
    if A.empty:
        st.warning("No maintenance tickets available to analyze.")
        return
    ch = ex["charts"]
    bv = ex.get("business") or {}
    if bv.get("assets") is not None and not bv["assets"].empty:
        A = bv["assets"]   # superset: adds reliability_score/level + lifecycle_stage columns

    # ---- Coverage & mapping ---- #
    st.subheader("Engine Coverage & Ticket Mapping")
    cov = ex["coverage"]
    e1 = st.columns(4)
    e1[0].metric("Total Maintenance Tickets", f"{cov['total_tickets']:,}")
    e1[1].metric("Tickets Successfully Processed", f"{cov['tickets_processed']:,}",
                 help="Participate in prediction via room and/or asset-type intelligence.")
    e1[2].metric("Tickets Waiting for Mapping", f"{cov['tickets_waiting_mapping']:,}")
    e1[3].metric("Coverage %", f"{cov['coverage_pct']}%",
                 help="Processed ÷ total. Asset-pinned coverage separately below.")
    e2 = st.columns(4)
    e2[0].metric("Total Assets", f"{cov['total_assets']:,}")
    e2[1].metric("Allocated Assets", f"{cov['allocated_assets']:,}")
    e2[2].metric("Assets w/ Purchase Date", f"{cov['assets_with_purchase_date']:,}")
    e2[3].metric("Assets using Allocation Date", f"{cov['assets_using_allocation_date']:,}")
    c = k["confidence"]
    st.caption(
        f"Whole maintenance history participates: **{cov['tickets_processed']} / {cov['total_tickets']} "
        f"tickets ({cov['coverage_pct']}%)** via room + asset-type intelligence. "
        f"Asset-pinned (specific asset): {cov['asset_pinned_reliable']} reliable "
        f"({cov['asset_coverage_pct']}%), {cov['asset_pinned_incl_medium']} incl. medium. "
        f"Confidence — Verified {c['Verified']} · High {c['High']} · Medium {c['Medium']} · Low {c['Low']}."
    )
    st.divider()

    # ---- Health / risk cards ---- #
    st.subheader("Asset Health & Risk")
    r1 = st.columns(4)
    r1[0].metric("🟢 Healthy", f"{k['healthy']:,}")
    r1[1].metric("🟡 Medium Risk", f"{k['medium_risk']:,}")
    r1[2].metric("🟠 High Risk", f"{k['high_risk']:,}")
    r1[3].metric("🔴 Critical", f"{k['critical']:,}")
    r2 = st.columns(4)
    r2[0].metric("Due for Maintenance", f"{k['due_maintenance']:,}")
    r2[1].metric("Due for Replacement", f"{k['due_replacement']:,}")
    r2[2].metric("Repeat-Failure Assets", f"{k['repeat_failures']:,}")
    r2[3].metric("Avg Health Score", round(A["health_score"].mean(), 1))
    st.divider()

    # ---- Room risk + asset-type risk ---- #
    st.subheader("Room Risk & Asset-Type Risk")
    rt = st.columns(2)
    with rt[0]:
        rr = ex["room_risk"]
        _bar(rr.head(15) if rr is not None else None, "apartment_code", "room_risk_score",
             "Room Risk Score (top apartments)", "ae_room")
        _table(rr.head(20) if rr is not None else None, "Room Risk Detail", "ae_room_tbl", "_No data._")
    with rt[1]:
        tr = ex["type_risk"]
        _bar(tr.head(15) if tr is not None else None, "asset_type", "type_risk_score",
             "Asset-Type Risk Score", "ae_type")
        _table(tr if tr is not None else None, "Asset-Type Risk Detail", "ae_type_tbl", "_No data._")
    st.divider()

    # ---- Room Intelligence (ALL tickets) ---- #
    st.subheader("🏢 Room Intelligence (all tickets)")
    ri = ex["room_intel"]
    if ri is None or ri.empty:
        st.caption("_No room data._")
    else:
        ic = st.columns(4)
        ic[0].metric("Rooms w/ Increasing Demand", int((ri["demand_trend"] == "Increasing").sum()))
        ic[1].metric("Rooms w/ Repeated Failures", int((ri["repeated_failures"] >= 1).sum()))
        ic[2].metric("Rooms w/ Multiple Asset Failures", int(ri["multiple_asset_failures"].sum()))
        ic[3].metric("Need Preventive Inspection", int(ri["preventive_inspection"].sum()))
        _bar(ri.head(15), "apartment_code", "total_tickets", "Maintenance Demand by Apartment", "ri_demand")
        _table(ri.head(25)[["apartment_code", "total_tickets", "recent_30d", "demand_trend",
                            "repeated_failures", "assets_failing", "est_maintenance_cost",
                            "room_risk_level", "preventive_inspection"]],
               "Room Intelligence Detail", "ri_tbl")
    st.divider()

    # ---- Asset-Type Intelligence (ALL tickets) ---- #
    st.subheader("🧩 Asset-Type Intelligence (all tickets)")
    ti = ex["type_intel"]
    if ti is None or ti.empty:
        st.caption("_No asset-type data._")
    else:
        tc = st.columns(3)
        tc[0].metric("High-Risk Asset Types", int(ti["high_risk"].sum()))
        tc[1].metric("Types Approaching End-of-Life", int(ti["approaching_end_of_life"].sum()))
        tc[2].metric("Asset Types Tracked", int(len(ti)))
        _bar(ti.head(15), "asset_type", "failure_frequency", "Failure Frequency by Asset Type", "ti_freq")
        _table(ti[["asset_type", "assets", "failure_frequency", "avg_age_months",
                   "avg_tickets_per_asset", "avg_maintenance_interval_days", "avg_health",
                   "approaching_end_of_life", "high_risk"]],
               "Asset-Type Intelligence Detail", "ti_tbl")
    st.divider()

    # ---- Executive Dashboard ---- #
    st.subheader("📊 Executive Dashboard")
    exe = ex["executive"]
    for line in exe.get("insights", []):
        st.markdown(f"- {line}")
    wc = st.columns(3)
    wc[0].metric("Workload Forecast (next 30d)", f"{exe['workload_forecast_30d']} tickets",
                 help=f"Last 30d = {exe['workload_last_30d']} tickets (damped trend).")
    wc[1].metric("Likely to Fail (30d)", len(exe["likely_fail_30d"]))
    wc[2].metric("Replacement Candidates", len(exe["replacement"]))
    pv = exe.get("purchase_vs_allocation", {})
    if pv:
        pvdf = pd.DataFrame([{"age_source": kk, "assets": vv["assets"], "avg_age_months": vv["avg_age_months"]}
                             for kk, vv in pv.items()])
        _table(pvdf, "Purchase vs Allocation Age Comparison", "exe_pv")
    eg = st.columns(2)
    with eg[0]:
        _table(exe["top_risky_apartments"][["apartment_code", "total_tickets", "demand_trend", "room_risk_level"]]
               if not exe["top_risky_apartments"].empty else None, "Top Risky Apartments", "exe_rooms", "_None._")
        _table(exe["likely_fail_30d"], "Assets Likely to Fail Next Month", "exe_likely", "_None._")
    with eg[1]:
        _table(exe["top_risky_asset_types"][["asset_type", "failure_frequency", "high_risk", "approaching_end_of_life"]]
               if not exe["top_risky_asset_types"].empty else None, "Top Risky Asset Categories", "exe_types", "_None._")
        _table(exe["replacement"], "Assets Needing Replacement", "exe_replace", "_None._")
    _table(exe["preventive"], "Assets Needing Preventive Maintenance", "exe_prev", "_None._")
    st.divider()

    # ---- Visualizations ---- #
    st.subheader("Predictive Visualizations")
    g = st.columns(2)
    with g[0]:
        _bar(ch.get("health_distribution"), "health", "assets", "Asset Health Distribution", "ae_health")
    with g[1]:
        _bar(ch.get("monthly_trend"), "month", "tickets", "Monthly Failure Trend", "ae_monthly")
    g2 = st.columns(2)
    with g2[0]:
        _bar(ch.get("failure_by_type"), "asset_type", "tickets", "Failures by Asset Type", "ae_ftype")
    with g2[1]:
        _bar(ch.get("maintenance_timeline"), "window", "assets", "Maintenance Due Timeline", "ae_mtl")
    _bar(ch.get("replacement_by_type"), "asset_type", "assets", "Replacement Forecast by Type", "ae_repl")
    _table(ch.get("top_failing_assets"), "Top Frequently-Failing Assets", "ae_topfail", "_No data._")
    st.divider()

    # ---- Recommendations + alerts ---- #
    st.subheader("Predictive Recommendations")
    rd = A["recommendation"].value_counts()
    rc = st.columns(len(rd) if len(rd) else 1)
    for i, (nm, cnt) in enumerate(rd.items()):
        rc[i].metric(nm, int(cnt))
    _table(A[A["recommendation"] != "No Action Needed"][
        ["asset_code", "asset_type", "apartment_code", "risk_level", "recommendation",
         "failure_prob_30d", "age_months", "reason"]].sort_values("failure_prob_30d", ascending=False),
        "Assets needing action", "ae_recs", "_No actions recommended._")
    alerts = A[A["alerts"].astype(str).str.len() > 0][["asset_code", "asset_type", "apartment_code", "risk_level", "alerts"]]
    _table(alerts, f"⚠ Active Alerts ({len(alerts)})", "ae_alerts", "_No active alerts._")
    st.divider()

    # ========================================================================= #
    # Business Maintenance Intelligence (items 1-10) — additive
    # ========================================================================= #
    if bv:
        ek = bv["exec_kpis"]
        # ---- 5. Executive KPI strip ---- #
        st.subheader("📈 Executive KPIs")
        x1 = st.columns(4)
        x1[0].metric("Total Assets", f"{ek['total_assets']:,}")
        x1[1].metric("Allocated Assets", f"{ek['allocated_assets']:,}")
        x1[2].metric("Total Tickets", f"{ek['total_tickets']:,}")
        x1[3].metric("Coverage %", f"{ek['coverage_pct']}%")
        x2 = st.columns(4)
        x2[0].metric("Assets w/ Purchase Date", f"{ek['assets_with_purchase_date']:,}")
        x2[1].metric("Assets using Allocation Date", f"{ek['assets_using_allocation_date']:,}")
        x2[2].metric("Maintenance Due", f"{ek['maintenance_due']:,}")
        x2[3].metric("Replacement Due", f"{ek['replacement_due']:,}")
        x3 = st.columns(4)
        x3[0].metric("🟢 Healthy", f"{ek['healthy']:,}")
        x3[1].metric("🟡 Medium", f"{ek['medium_risk']:,}")
        x3[2].metric("🟠 High", f"{ek['high_risk']:,}")
        x3[3].metric("🔴 Critical", f"{ek['critical_risk']:,}")
        x4 = st.columns(2)
        x4[0].metric("Top Problem Apartment", ek["top_problem_apartment"])
        x4[1].metric("Top Problem Asset Type", ek["top_problem_asset_type"])
        st.divider()

        # ---- 1. Asset Reliability ---- #
        st.subheader("🔩 Asset Reliability (0–100)")
        rl = A["reliability_level"].value_counts().reindex(["High", "Medium", "Low", "Poor"]).fillna(0).astype(int)
        rc = st.columns(4)
        for i, lvl in enumerate(["High", "Medium", "Low", "Poor"]):
            rc[i].metric(f"{lvl} Reliability", int(rl[lvl]))
        _table(A.sort_values("reliability_score")[
            ["asset_code", "asset_type", "apartment_code", "reliability_score", "reliability_level",
             "ticket_count", "repeat_count", "age_months"]].head(30),
            "Reliability (lowest first)", "bv_reliability")
        st.divider()

        # ---- 10. Asset Lifecycle ---- #
        st.subheader("♻ Asset Lifecycle")
        _bar(bv["lifecycle_counts"], "stage", "assets", "Assets by Lifecycle Stage", "bv_lifecycle")
        st.divider()

        # ---- 2. Apartment Health ---- #
        st.subheader("🏢 Apartment Health Score")
        _bar(bv["apartment_health"].head(15), "apartment_code", "health_score",
             "Apartment Health (lowest = worst; top 15)", "bv_apt_health_chart")
        _table(bv["apartment_health"], "Apartment Health Detail", "bv_apt_health")
        st.divider()

        # ---- 3. Asset-Type Health ---- #
        st.subheader("🧩 Asset-Type Health")
        _table(bv["asset_type_health"], "Asset-Type Health Detail", "bv_type_health")
        st.divider()

        # ---- 4. Maintenance Calendar ---- #
        st.subheader("📅 Maintenance Calendar")
        cal = bv["calendar"]
        cc = st.columns(4)
        cc[0].metric("Today / Overdue", len(cal["today"]))
        cc[1].metric("This Week", len(cal["this_week"]))
        cc[2].metric("This Month", len(cal["this_month"]))
        cc[3].metric("Next Month", len(cal["next_month"]))
        for label, key in [("Today / Overdue", "today"), ("This Week", "this_week"),
                           ("This Month", "this_month"), ("Next Month", "next_month")]:
            if not cal[key].empty:
                _table(cal[key], label, f"bv_cal_{key}")
        if all(cal[k].empty for k in cal):
            st.caption("_No assets due within 60 days — asset maintenance cycles (6–24 mo) exceed the "
                       "current ticket-history window; schedule populates as history lengthens._")
        st.divider()

        # ---- 6. Failure Hotspots ---- #
        st.subheader("🔥 Failure Hotspots")
        hs = bv["hotspots"]
        h1 = st.columns(2)
        with h1[0]:
            _bar(hs["apartment_density"], "apartment_code", "tickets", "Apartment-wise Ticket Density", "bv_hs_apt")
        with h1[1]:
            _bar(hs["monthly_trend"], "month", "tickets", "Monthly Ticket Trend", "bv_hs_month")
        h2 = st.columns(2)
        with h2[0]:
            _bar(hs["type_failures"], "asset_type", "total_failures", "Asset-Type-wise Failures", "bv_hs_type")
        with h2[1]:
            _bar(hs["floor_density"] if not hs["floor_density"].empty else None, "floor", "tickets",
                 "Floor-wise Ticket Density", "bv_hs_floor")
        st.divider()

        # ---- 7. Asset Rankings ---- #
        st.subheader("🏆 Asset Rankings")
        rk = bv["rankings"]
        kg = st.columns(2)
        with kg[0]:
            _table(rk["least_reliable"], "Least Reliable", "bv_rk_least")
            _table(rk["most_repaired"], "Most Frequently Repaired", "bv_rk_repaired")
            _table(rk["highest_risk"], "Highest Risk", "bv_rk_risk")
        with kg[1]:
            _table(rk["most_reliable"], "Most Reliable", "bv_rk_most")
            _table(rk["near_end_of_life"], "Near End-of-Life", "bv_rk_eol")
        st.divider()

        # ---- 8. Repeat Failure Analysis ---- #
        st.subheader("🔁 Repeat Failure Analysis")
        _table(bv["repeat_analysis"], "Assets with repeated failures", "bv_repeat",
               "_No repeat failures._")
        st.divider()

        # ---- 9. Preventive Maintenance Queue ---- #
        st.subheader("🧰 Preventive Maintenance Queue")
        pq = bv["preventive_queue"]
        pcnt = pq["priority"].value_counts().reindex(["Critical", "High", "Medium"]).fillna(0).astype(int) if not pq.empty else None
        if pcnt is not None:
            pc = st.columns(3)
            pc[0].metric("🔴 Critical", int(pcnt["Critical"]))
            pc[1].metric("🟠 High", int(pcnt["High"]))
            pc[2].metric("🟡 Medium", int(pcnt["Medium"]))
        _table(pq, "Prioritized work list (with reason)", "bv_queue", "_Queue empty._")
        st.divider()

    # ========================================================================= #
    # Management & Business Planning (extension 3) — additive
    # ========================================================================= #
    # ---- 1. Maintenance Budget Forecast ---- #
    st.subheader("💰 Maintenance Budget Forecast")
    bf = ex["budget"]
    st.caption(f"Based on recent ticket rate (~{bf['monthly_rate']}/month) × avg ticket cost "
               f"{_money(bf['avg_ticket_cost'])} (cost is an estimate — source closure_cost is null).")
    bw = bf["windows"].copy()
    bcols = st.columns(4)
    for i, (_, w) in enumerate(bw.iterrows()):
        bcols[i].metric(w["window"], f"{int(w['expected_jobs'])} jobs")
        bcols[i].caption(_money(w["estimated_budget"]))
    _table(bw, "Budget Forecast", "mp_budget")
    _table(bf["top_contributing_types"], "Top Contributing Asset Types", "mp_budget_types")
    st.divider()

    # ---- 2. Asset Replacement Planner ---- #
    st.subheader("🗓 Asset Replacement Planner")
    pl = ex["planner"]
    _table(pl["summary"], "Replacement Plan Summary (count · est. cost · priority)", "mp_plan_sum", "_No assets in replacement window._")
    _table(pl["detail"], "Replacement Plan Detail", "mp_plan_detail", "_None._")
    st.divider()

    # ---- 3. Maintenance Workload Forecast ---- #
    st.subheader("📈 Maintenance Workload Forecast")
    wf = ex["workload"]
    ww = wf["windows"]
    wc = st.columns(4)
    for i, (_, w) in enumerate(ww.iterrows()):
        wc[i].metric(w["window"], f"{int(w['expected_tickets'])} tickets")
    wg = st.columns(2)
    with wg[0]:
        _bar(wf["by_apartment"], "apartment_code", "expected_next_month", "Expected Next Month by Apartment", "mp_wl_apt")
    with wg[1]:
        _bar(wf["by_type"], "issue_type", "expected_next_month", "Expected Next Month by Issue/Asset Type", "mp_wl_type")
    st.divider()

    # ---- 4. Asset ROI ---- #
    st.subheader("💹 Asset ROI (value for money)")
    roi = ex["roi"]
    if not roi.empty:
        st.metric("Poor-Value Asset Types", int(roi["poor_value"].sum()))
    _table(roi, "Asset ROI by Type", "mp_roi", "_No ROI data._")
    st.divider()

    # ---- 6. Apartment Report Card ---- #
    st.subheader("🏅 Apartment Maintenance Report Card (Best → Worst)")
    _table(ex["report_card"], "Apartment Report Card", "mp_report_card", "_No data._")
    st.divider()

    # ---- 7. Asset Performance Comparison ---- #
    st.subheader("⚖ Asset Performance Comparison")
    _table(ex["performance"], "Asset-Type Performance (reliability, failures, age, freq, high-risk %)", "mp_perf", "_No data._")
    st.divider()

    # ---- 8. Business Recommendations ---- #
    st.subheader("💡 Business Recommendations")
    recs = ex["recommendations"]
    if recs is None or recs.empty:
        st.caption("_No recommendations._")
    else:
        for _, r in recs.iterrows():
            badge = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(r["priority"], "⚪")
            st.markdown(f"**{badge} {r['priority']} — {r['recommendation']}**  \n"
                        f"WHY: {r['why']}  \nEXPECTED IMPACT: {r['expected_impact']}")
        _dl(recs, "recommendations", "mp_recs")
    st.divider()

    # ---- 9. Printable Management Report ---- #
    st.subheader("🖨 Printable Management Report")
    st.caption("One-click monthly summary — download the HTML and print / save as PDF from the browser.")
    html = _management_report_html(ex, bf, pl, recs)
    st.download_button("⬇ Download Management Report (HTML)", html.encode("utf-8"),
                       file_name="maintenance_management_report.html", mime="text/html", key="mp_report_html")
    st.divider()

    # ========================================================================= #
    # Extension 4 — SLA / Brand / Purchase (additive; auto-hide when data absent)
    # ========================================================================= #
    # ---- 1. Maintenance SLA Dashboard ---- #
    sla = ex.get("sla") or {}
    if sla:
        st.subheader("⏱ Maintenance SLA Dashboard")
        o = sla["overall"]
        s1 = st.columns(4)
        s1[0].metric("Total Tickets", f"{o['total_tickets']:,}")
        s1[1].metric("Open", f"{o['open_tickets']:,}")
        s1[2].metric("Closed", f"{o['closed_tickets']:,}")
        s1[3].metric("Avg Resolution (hrs)", o["avg_resolution_hours"] if o["avg_resolution_hours"] is not None else "—")
        s2 = st.columns(4)
        s2[0].metric("Median Resolution (hrs)", o["median_resolution_hours"] if o["median_resolution_hours"] is not None else "—")
        s2[1].metric("Fastest (hrs)", o["fastest_resolution_hours"] if o["fastest_resolution_hours"] is not None else "—")
        s2[2].metric("Slowest (hrs)", o["slowest_resolution_hours"] if o["slowest_resolution_hours"] is not None else "—")
        s2[3].metric("SLA Met %", f"{o['sla_met_pct']}%" if o["sla_met_pct"] is not None else "—")
        st.caption(f"SLA Violated {o['sla_violated_pct']}% · resolution measured on {o['resolution_measured_on']} "
                   f"tickets, SLA on {o['sla_measured_on']} (tickets with both resolved + deadline).")
        _table(sla["by_apartment"], "SLA by Apartment", "sla_apt")
        _table(sla["by_issue_type"], "SLA by Issue Type", "sla_issue")
        _table(sla["by_asset_type"], "SLA by Asset Type", "sla_type")
        if sla.get("has_technician"):
            _table(sla["technician"], "Technician-wise (assigned_to id)", "sla_tech")
        st.divider()

    # ---- 2. Vendor / Brand Analysis ---- #
    brand = ex.get("brand") or {}
    if brand:
        st.subheader("🏭 Vendor / Brand Analysis")
        _table(brand["by_brand"], "Brand Reliability (all brands)", "brand_all")
        bcol = st.columns(2)
        with bcol[0]:
            _table(brand["most_reliable"][["brand", "total_assets", "failure_rate", "avg_reliability"]],
                   "Most Reliable Brands", "brand_best")
        with bcol[1]:
            _table(brand["worst_performing"][["brand", "total_assets", "failure_rate", "avg_reliability"]],
                   "Worst Performing Brands", "brand_worst")
        if brand.get("recommendations"):
            st.markdown("**Brand recommendations:**")
            for line in brand["recommendations"]:
                st.markdown(f"- {line}")
        st.caption("Brand metrics cover assets with a populated brand field only.")
        st.divider()

    # ---- 3. Purchase Recommendation ---- #
    pur = ex.get("purchase")
    if pur is not None and not pur.empty:
        st.subheader("🛒 Purchase Recommendation")
        prc = pur["recommendation"].value_counts()
        pcol = st.columns(len(prc) if len(prc) else 1)
        for i, (nm, cnt) in enumerate(prc.items()):
            pcol[i].metric(nm, int(cnt))
        _table(pur, "Purchase Guidance by Asset Type", "purchase_rec")
        st.divider()

    # ---- Search + detail ---- #
    st.subheader("Asset Search & Profile")
    f = st.columns(4)
    q = f[0].text_input("Asset Code / Brand / Model contains")
    apt_opts = ["All"] + sorted([x for x in A["apartment_code"].dropna().unique() if str(x).strip()])
    type_opts = ["All"] + sorted([x for x in A["asset_type"].dropna().unique() if str(x).strip()])
    a_apt = f[1].selectbox("Apartment", apt_opts)
    a_type = f[2].selectbox("Asset Type", type_opts)
    a_risk = f[3].selectbox("Risk Level", ["All", "Critical", "High", "Medium", "Low"])
    v = A.copy()
    if q.strip():
        ql = q.strip().lower()
        v = v[v["asset_code"].astype(str).str.lower().str.contains(ql)
              | v["brand"].astype(str).str.lower().str.contains(ql)
              | v["model"].astype(str).str.lower().str.contains(ql)]
    if a_apt != "All":
        v = v[v["apartment_code"] == a_apt]
    if a_type != "All":
        v = v[v["asset_type"] == a_type]
    if a_risk != "All":
        v = v[v["risk_level"] == a_risk]
    cols = ["asset_code", "asset_type", "apartment_code", "bed_code", "age_months", "age_source",
            "health_score", "risk_level", "failure_prob_30d", "ticket_count", "recommendation"]
    _table(v[[c for c in cols if c in v.columns]], f"{len(v)} assets", "ae_search", "_No assets match._")

    if not v.empty:
        pick = st.selectbox("Open asset profile", ["—"] + v["asset_code"].tolist())
        if pick and pick != "—":
            r = v[v["asset_code"] == pick].iloc[0]
            st.markdown(f"### {pick} — {r['asset_type']}  ·  {_risk_badge(r['risk_level'])}")
            p1 = st.columns(4)
            p1[0].metric("Health Score", r["health_score"])
            p1[1].metric("30d Failure Prob", f"{r['failure_prob_30d']}%")
            p1[2].metric("Failure Trend", r["failure_trend"])
            p1[3].metric("Total Tickets", int(r["ticket_count"]))
            p2 = st.columns(4)
            p2[0].metric("Age (months)", "—" if pd.isna(r["age_months"]) else r["age_months"])
            p2[1].metric("Age Source", r["age_source"])
            p2[2].metric("Expected Life (mo)", "—" if pd.isna(r["expected_life_months"]) else r["expected_life_months"])
            p2[3].metric("Maint. Cycle (mo)", "—" if pd.isna(r["maintenance_cycle_months"]) else r["maintenance_cycle_months"])
            p3 = st.columns(4)
            p3[0].metric("Maintenance Due (days)", "—" if pd.isna(r["maintenance_due_days"]) else int(r["maintenance_due_days"]))
            p3[1].metric("Purchase Date", "—" if pd.isna(r["purchase_date"]) else str(r["purchase_date"])[:10])
            p3[2].metric("Recent (30d)", int(r["recent_30d"]))
            p3[3].metric("Exp. Cost 1y", _money(r["expected_cost_1y"]))
            st.success(f"**Recommendation: {r['recommendation']}** — {r['reason']}"
                       + (f"  ·  Alerts: {r['alerts']}" if str(r['alerts']).strip() else ""))
            # ---- Maintenance History Timeline (item 5) ----
            st.markdown("**Maintenance History Timeline**")
            alloc = r.get("age_estimated_from") if r.get("age_source") == "Allocation Date" else "—"
            due = pd.to_numeric(pd.Series([r.get("maintenance_due_days")]), errors="coerce").iloc[0]
            nxt = (pd.Timestamp.today().normalize() + pd.Timedelta(days=float(due))).date() if pd.notna(due) else None
            tlm = st.columns(4)
            tlm[0].metric("Allocation Date", alloc or "—")
            tlm[1].metric("Purchase Date", "—" if pd.isna(r["purchase_date"]) else str(r["purchase_date"])[:10])
            tlm[2].metric("Last Maintenance", "—" if pd.isna(r["last_ticket"]) else str(r["last_ticket"])[:10])
            tlm[3].metric("Next Predicted", str(nxt) if nxt is not None else "—")
            tl = built["mapped"]
            tl = tl[tl["asset_id"] == r["asset_id"]][["created_at", "issue_type", "confidence"]].sort_values("created_at")
            _table(tl, "Every Maintenance Ticket", "ae_timeline", "_No tickets._")
    st.divider()

    # ---- Export ---- #
    st.subheader("⬇ Export Reports")
    exp = ex["exports"]
    ec = st.columns(3)
    for i, (key, label) in enumerate([("asset_health_report", "Full Asset Health Report"),
                                      ("maintenance_schedule", "Maintenance Schedule"),
                                      ("replacement_plan", "Replacement Plan")]):
        df = exp.get(key)
        with ec[i]:
            st.caption(label)
            if df is not None and not df.empty:
                st.download_button("⬇ CSV", df.to_csv(index=False).encode("utf-8"),
                                   file_name=f"asset_{key}.csv", mime="text/csv", key=f"aexp_{key}")
                st.caption(f"{len(df)} rows")
            else:
                st.caption("_empty_")



# --------------------------------------------------------------------------- #
# Shell
# --------------------------------------------------------------------------- #
PAGES = {
    "Inventory Overview": page_inventory_overview,
    "Customer Recommendation": page_recommendation,
    "Blocked Rooms": page_blocked_rooms,
    "Revenue Leakage": page_revenue_leakage,
    "Occupancy Analytics": page_occupancy,
    "Room Search": page_room_search,
    "Maintenance Intelligence": page_maintenance_intelligence,
    "Asset Predictive Analytics": page_asset_predictive,
}


def main():
    st.sidebar.title("Availability AI")
    choice = st.sidebar.radio("Navigate", list(PAGES.keys()))
    st.sidebar.divider()
    st.sidebar.caption(
        "Live production data. "
        "Active Vacant Beds = recommendable. Inactive = informational only."
    )

    PAGES[choice]()


if __name__ == "__main__":
    main()