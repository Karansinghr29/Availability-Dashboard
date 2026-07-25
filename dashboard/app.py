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
# Shell
# --------------------------------------------------------------------------- #
PAGES = {
    "Inventory Overview": page_inventory_overview,
    "Customer Recommendation": page_recommendation,
    "Blocked Rooms": page_blocked_rooms,
    "Revenue Leakage": page_revenue_leakage,
    "Occupancy Analytics": page_occupancy,
    "Room Search": page_room_search,
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