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

    # apartments.start_date drives the activation-anchored occupancy rule (new
    # apartments measured only from start_date onward) — shared by every
    # availability page so occupancy is consistent across all dashboards.
    apartments = None
    try:
        apartments = loader.apartment_master()
    except Exception:  # noqa: BLE001
        apartments = None

    inventory = prepare_inventory(
        build_room_inventory(
            bookings, tenants=tenants, beds_master=beds_master, bed_map=bed_map,
            apartments=apartments,
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
    # Bed-level view keyed by the physical DB record (bed_id) — NOT deduplicated
    # by (apartment_code, bed_code). Two vacant records that share a bed_code but
    # differ by bed_id are BOTH shown. This is display-only and does not change
    # the recommendation eligibility rule / active_vacant_beds. Inactive-apartment
    # beds stay excluded (same rule as everywhere else).
    from bed_inventory import vacant_beds_by_bed_id

    loader, _ = get_engine()
    vb = vacant_beds_by_bed_id(loader)
    if vb is not None and not vb.empty:
        vb = vb[vb["room_code"].astype(str).isin(room_codes)].copy()
        if bed_q.strip():
            vb = vb[
                vb["bed_code"].astype(str).str.contains(bed_q.strip(), case=False, na=False)
            ]
    if vb is None or vb.empty:
        st.caption("No vacant beds for this selection.")
    else:
        n_dup = int(vb["duplicate_bed_code"].sum())
        st.caption(
            "Each vacant **physical bed** is listed by its unique **Bed ID**; beds "
            "are never merged by bed code. When two rows share a bed code but have "
            "different Bed IDs they are two separate database records (a real "
            "second bed or a data-entry duplicate — both kept, none deleted)."
            + (f" ⚠ {n_dup} bed(s) here share a bed code — see the Bed ID column."
               if n_dup else "")
        )
        vb_show = vb.copy()
        vb_show["Rent"] = vb_show["monthly_rate"].map(_money)
        # Bed ID is small secondary metadata (short form); the ⚠ marks a shared code.
        vb_show["Bed ID"] = vb_show["bed_id"].astype(str).str.slice(0, 8) + "…"
        vb_show["Bed"] = [
            f"{code} ⚠" if dup else code
            for code, dup in zip(vb_show["bed_code"].astype(str), vb_show["duplicate_bed_code"])
        ]
        vac_cols = {
            "apartment_code": "Apartment",
            "room_code": "Room",
            "Bed": "Bed",
            "bed_type": "Bed Type",
            "toilet_type": "Toilet Type",
            "Rent": "Rent",
            "occupancy_status": "Status",
            "occupancy_history": "History",
            "Bed ID": "Bed ID",
        }
        vcols = [c for c in vac_cols if c in vb_show.columns]
        st.dataframe(
            vb_show[vcols].rename(columns=vac_cols),
            use_container_width=True,
            hide_index=True,
            key="room_search_vacant_beds",
        )


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


def _xtable(df: pd.DataFrame, caption: str, key: str, empty_msg: str = "_No data._", expanded: bool = False):
    """Detail table inside a collapsible expander (charts/KPIs stay in the default view)."""
    n = 0 if df is None else len(df)
    with st.expander(f"📋 {caption}" + (f"  ({n})" if n else ""), expanded=expanded):
        if df is None or df.empty:
            st.caption(empty_msg)
        else:
            st.dataframe(df, use_container_width=True, hide_index=True)
            _dl(df, caption, key)


def _insight_card(col, emoji: str, title: str, value, color: str):
    """Colored insight card with a left accent bar."""
    col.markdown(
        f"<div style='border-left:6px solid {color};background:rgba(127,127,127,0.08);"
        f"padding:12px 16px;border-radius:8px;height:100%'>"
        f"<div style='font-size:13px;font-weight:600;opacity:0.85'>{emoji} {title}</div>"
        f"<div style='font-size:26px;font-weight:800;margin-top:4px;line-height:1.15'>{value}</div>"
        f"</div>",
        unsafe_allow_html=True)


def _gap(px: int = 12):
    """Vertical spacer between sections."""
    st.markdown(f"<div style='height:{px}px'></div>", unsafe_allow_html=True)


_RISK_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}


def _room_asset_index(A):
    """Presentation-only per-apartment asset rollup from the engine's existing per-asset
    scores. No new prediction — only reads columns already produced by asset_engine."""
    idx = {}
    if A is None or A.empty:
        return idx
    for apt, g in A.groupby("apartment_code"):
        if not str(apt).strip():
            continue
        gg = g.copy()
        gg["_r"] = gg["risk_level"].map(_RISK_ORDER).fillna(9)
        top_risk = gg.sort_values(["_r", "health_score", "failure_prob_30d"],
                                  ascending=[True, True, False]).iloc[0]
        predict = gg.sort_values(["failure_prob_30d", "_r"],
                                 ascending=[False, True]).iloc[0]
        idx[apt] = {"assets": gg, "top_risk": top_risk, "predict": predict}
    return idx


def _room_label(apt, bed=None):
    """'Apartment A23' or 'Apartment A23 / Bed B2' from existing allocation fields."""
    apt = str(apt).strip() or "—"
    bed = str(bed).strip() if bed is not None else ""
    return f"Apartment {apt}" + (f" / Bed {bed}" if bed and bed.lower() != "nan" else "")


def _why_bullets(r):
    """WHY the asset is predicted to fail next — built only from existing engine columns."""
    bits = []
    tc = int(r.get("ticket_count", 0) or 0)
    rec = int(r.get("recent_30d", 0) or 0)
    at = str(r.get("asset_type", "asset")) or "asset"
    if tc:
        bits.append(f"{tc} historical {at} ticket(s)" + (f", {rec} in last 30d" if rec else ""))
    rp = int(r.get("repeat_count", 0) or 0)
    if rp:
        bits.append(f"{rp} repeat failure(s)")
    am, el, src, ar = r.get("age_months"), r.get("expected_life_months"), r.get("age_source"), r.get("age_ratio")
    if pd.notna(am):
        life_txt = f" vs expected {int(el)} mo" if pd.notna(el) else ""
        cmp = ""
        if pd.notna(ar):
            cmp = " — exceeds expected lifecycle" if ar >= 1 else f" ({int(100 * ar)}% of lifecycle)"
        bits.append(f"asset age {am} mo ({src}){life_txt}{cmp}")
    tr = r.get("failure_trend")
    if tr and tr != "Stable":
        bits.append(f"maintenance trend: {tr}")
    fp = r.get("failure_prob_30d")
    if pd.notna(fp):
        bits.append(f"{fp}% modeled 30-day failure probability")
    return bits or ["limited history — flagged by current risk level"]


def _loc(apt, bed=None):
    """Consistent physical-location label: 'A23 / Bed C1', or 'A23' when no bed."""
    apt = str(apt).strip()
    if not apt or apt.lower() == "nan":
        apt = "—"
    bed = str(bed).strip() if bed is not None else ""
    return f"{apt} / Bed {bed}" if bed and bed.lower() != "nan" else apt


def _add_location(df, A=None):
    """Presentation-only: replace apartment_code (+bed_code) with a single 'location'
    column ('A23 / Bed C1'). If the row has no bed_code, look it up by asset_code from A;
    falls back to apartment code alone when no bed is available. No engine change."""
    if df is None or df.empty:
        return df
    d = df.copy()
    n = len(d)
    if "bed_code" in d.columns:
        beds = d["bed_code"].astype(str)
    elif A is not None and "asset_code" in d.columns:
        bmap = dict(zip(A["asset_code"].astype(str), A["bed_code"].astype(str)))
        beds = d["asset_code"].astype(str).map(bmap)
    else:
        beds = pd.Series([""] * n, index=d.index)
    if "apartment_code" in d.columns:
        apts = d["apartment_code"].astype(str)
    elif A is not None and "asset_code" in d.columns:
        amap = dict(zip(A["asset_code"].astype(str), A["apartment_code"].astype(str)))
        apts = d["asset_code"].astype(str).map(amap)
    else:
        apts = pd.Series([""] * n, index=d.index)
    loc = [_loc(a, b) for a, b in zip(apts, beds)]
    insert_at = d.columns.get_loc("apartment_code") if "apartment_code" in d.columns else 0
    d.insert(insert_at, "location", loc)
    return d.drop(columns=[c for c in ["apartment_code", "bed_code"] if c in d.columns], errors="ignore")


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Page 8 — Asset Predictive Analytics (ticket-centric engine, from scratch)
# Standalone; reads only; no DB writes. Uses src/asset_engine.py exclusively.
# --------------------------------------------------------------------------- #
def _room_reason(n, r30, repeat, prob, age_ratio=None, overdue=False, trend=""):
    """Qualitative reason bullets — derived only from existing engine outputs + ticket counts."""
    q = []
    if n >= 5:
        q.append("High ticket frequency")
    elif n >= 2:
        q.append("Multiple tickets from this room")
    if repeat >= 1:
        q.append("Repeat failures")
    if r30 >= 1:
        q.append("Recent activity in last 30 days")
    if prob is not None and prob >= 80:
        q.append("Similar assets fail frequently")
    if age_ratio is not None and pd.notna(age_ratio) and age_ratio >= 1:
        q.append("Asset age exceeds expected lifecycle")
    if overdue:
        q.append("Maintenance cycle exceeded")
    if trend in ("Degrading", "Rapidly Degrading"):
        q.append("Failure trend worsening")
    return q or ["Flagged by current risk level"]


def _join_and(items):
    items = [str(x) for x in items if str(x).strip()]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _asset_reason(r):
    """Short per-asset reason bullets — from existing engine columns only."""
    b = []
    tc = int(r.get("ticket_count", 0) or 0)
    rec = int(r.get("recent_30d", 0) or 0)
    rp = int(r.get("repeat_count", 0) or 0)
    if tc >= 4:
        b.append("High historical failure frequency")
    elif tc >= 2:
        b.append("Multiple tickets")
    if rp >= 1:
        b.append("Repeat failures")
    if rec >= 1:
        b.append("Recent activity (30d)")
    ar = r.get("age_ratio")
    if pd.notna(ar) and ar >= 1:
        b.append("Age exceeds expected lifecycle")
    if bool(r.get("maintenance_overdue")):
        b.append("Near/over maintenance cycle")
    if r.get("failure_trend") in ("Degrading", "Rapidly Degrading"):
        b.append("Failure trend worsening")
    return b or ["Flagged by current risk level"]


def _room_reco(asset_type, risk, overdue, replace, prob):
    """Preventive-maintenance recommendation — phrased from existing engine flags only."""
    a = asset_type or "the asset"
    if replace:
        return f"Replace {a} — end of expected life."
    if risk == "Critical" or overdue:
        return f"Inspect {a} within 7 days."
    if risk == "High" or (prob is not None and prob >= 80):
        return f"Inspect {a} within 30 days."
    if risk == "Medium":
        return f"Schedule preventive maintenance for {a}."
    return f"Monitor {a}; no immediate action."


def _room_risk(total, repeat, prob, health, r30):
    """Presentation-only room prioritisation score (does NOT alter engine scoring).
    Combines ticket volume, repeat issues, predicted failure probability, asset health
    and recent activity into a rank/label for the maintenance team."""
    s = min(total, 20) * 1.5 + min(repeat, 5) * 6.0 + (prob or 0) * 0.25
    if health is not None and pd.notna(health):
        s += max(0.0, 100.0 - float(health)) * 0.15
    s += min(r30, 5) * 2.0
    lvl = "Critical" if s >= 70 else "High" if s >= 50 else "Medium" if s >= 30 else "Low"
    return round(s, 1), lvl


def _ticket_room_view(loader, built, assets=None):
    """PRESENTATION LAYER ONLY — authoritative room comes from the ticket itself
    (maintenance_tickets.bed_id → beds_master.bed_code → apartment_code), NOT from
    the asset's current allocation. Each room is then matched to the engine's existing
    per-asset predictions to surface the asset most likely to fail next in that room.
    The prediction engine (asset_engine.py) is not modified."""
    import asset_engine as AE
    S = AE._s
    mt = loader.maintenance_tickets().copy()
    if mt.empty or "bed_id" not in mt.columns:
        return {"table": pd.DataFrame(), "rooms": {}}
    beds, aptm, iss = loader.beds_master_uuid(), loader.apartment_master(), loader.issue_types()
    bcode = dict(zip(beds["id"].map(S), beds["bed_code"].map(S))) if not beds.empty else {}
    bapt = dict(zip(beds["id"].map(S), beds["apartment_id"].map(S))) if (not beds.empty and "apartment_id" in beds.columns) else {}
    aptcode = dict(zip(aptm["id"].map(S), aptm["apartment_code"].map(S))) if not aptm.empty else {}
    iname = dict(zip(iss["id"].map(S), iss["name"].map(S))) if not iss.empty else {}

    bed_id = mt["bed_id"].map(S)
    room = bed_id.map(lambda b: bcode.get(b, ""))
    apt_via_bed = bed_id.map(lambda b: aptcode.get(bapt.get(b, ""), ""))
    apt_direct = mt["apartment_id"].map(lambda x: aptcode.get(S(x), S(x)))
    apt = [a if a else d for a, d in zip(apt_via_bed, apt_direct)]
    T = pd.DataFrame({
        "apartment_code": apt, "room": list(room),
        "issue_type": [iname.get(S(x), "") for x in mt["issue_type_id"]],
        "created_at": pd.to_datetime(mt.get("created_at"), errors="coerce", utc=True).dt.tz_localize(None),
    })
    T = T[T["room"].astype(str).str.strip().replace({"nan": ""}) != ""]
    if T.empty:
        return {"table": pd.DataFrame(), "rooms": {}}

    A = assets if (assets is not None and not assets.empty) else built["assets"]
    now = pd.Timestamp.today().normalize()
    rows, rooms = [], {}
    for (apt_c, rm), g in T.groupby(["apartment_code", "room"]):
        n = len(g)
        r30 = int((g["created_at"] >= now - pd.Timedelta(days=30)).sum())
        r90 = int((g["created_at"] >= now - pd.Timedelta(days=90)).sum())
        _iss = g["issue_type"][g["issue_type"].astype(bool)]
        top_issue = _iss.value_counts().idxmax() if not _iss.empty else ""
        conf = "High"
        cand = A[(A["apartment_code"] == apt_c) & (A["bed_code"].astype(str) == str(rm))] if not A.empty else A
        if (cand is None or cand.empty) and not A.empty:
            cand = A[A["apartment_code"] == apt_c]
            conf = "Medium"
        distinct_issues = int(_iss.nunique()) if not _iss.empty else 0
        issue_breakdown = [f"{kk}: {vv}" for kk, vv in _iss.value_counts().head(6).items()] if not _iss.empty else []
        top_issue_count = int(_iss.value_counts().iloc[0]) if not _iss.empty else 0
        repeat, last_maint, next_maint, prob, asset_tickets = 0, "", "—", None, n
        brand, health = "", None
        pred_age, pred_trend, est_window = None, "", "—"
        pred_segment, pred_anom_score, pred_anom_flag, room_max_anom = "—", None, False, None
        inspection, high_risk_assets, effort = [], 0, "Low"
        needs_today = needs_week = multi_high_risk = likely_30d = False
        if cand is None or cand.empty:
            p_type, p_code = (top_issue or "—"), ""
            conf = "Low"
            reason = _room_reason(n, r30, 0, None)
            reco = "Inspect room; no scored asset currently mapped to this bed."
            combined_reco = "This room has recurring tickets but no scored asset is currently mapped to the bed; inspect on the next visit."
        else:
            _cs = cand.sort_values("failure_prob_30d", ascending=False)
            pr = _cs.iloc[0]
            p_type, p_code, prob = pr["asset_type"], pr["asset_code"], pr["failure_prob_30d"]
            brand = str(pr.get("brand", "") or "").strip()
            health = pr.get("health_score")
            repeat = int(pr.get("repeat_count", 0) or 0)
            asset_tickets = int(pr.get("ticket_count", 0) or 0)
            lt = pr.get("last_ticket")
            last_maint = str(lt)[:10] if pd.notna(lt) else ""
            due = pr.get("maintenance_due_days")
            if pd.notna(due):
                due = int(due)
                next_maint = "Overdue" if due <= 0 else f"Within {due} days"
            ar = pr.get("age_ratio")
            pred_age = pr.get("age_months")
            pred_trend = pr.get("failure_trend", "")
            aiv = pr.get("avg_interval_days")
            if pd.notna(aiv) and pd.notna(lt):
                _est = lt.normalize() + pd.Timedelta(days=float(aiv))
                est_window = (f"~{_est.date()} (overdue)" if _est < now else f"~{_est.date()}")
            reason = _room_reason(n, r30, repeat, prob, ar,
                                  bool(pr.get("maintenance_overdue")), pr.get("failure_trend", ""))
            reco = _room_reco(p_type, pr.get("risk_level", ""), bool(pr.get("maintenance_overdue")),
                              bool(pr.get("replacement_recommended")), prob)
            # ML enrichment (additive; only if the ML columns were merged onto the asset frame)
            pred_segment = str(pr.get("ml_risk_segment", "—") or "—")
            _pas = pr.get("ml_anomaly_score")
            pred_anom_score = float(_pas) if pd.notna(_pas) else None
            pred_anom_flag = bool(pr.get("ml_anomaly_flag", False))
            if "ml_anomaly_score" in _cs.columns and _cs["ml_anomaly_score"].notna().any():
                room_max_anom = round(float(_cs["ml_anomaly_score"].max()), 1)
            for _, arw in _cs.iterrows():
                _b = str(arw.get("bed_code", "")).strip()
                inspection.append({
                    "asset_type": arw["asset_type"], "asset_code": arw["asset_code"],
                    "bed": (_b if _b and _b.lower() != "nan" else ""),
                    "prob": arw["failure_prob_30d"], "risk": arw["risk_level"],
                    "health": arw["health_score"], "reasons": _asset_reason(arw),
                })
            high_risk_assets = int(_cs["risk_level"].isin(["High", "Critical"]).sum())
            effort = "High" if high_risk_assets >= 3 else "Medium" if high_risk_assets == 2 else "Low"
            _others = [a["asset_type"] for a in inspection[1:]
                       if (a["prob"] is not None and a["prob"] >= 50) or a["risk"] in ("Critical", "High")][:3]
            if _others:
                combined_reco = (f"This room has a high probability of future {inspection[0]['asset_type']} "
                                 f"failure. During the next maintenance visit, also inspect {_join_and(_others)} "
                                 f"to reduce repeat visits.")
            else:
                combined_reco = (f"This room has an elevated probability of {inspection[0]['asset_type']} "
                                 f"failure. Prioritise it during the next preventive maintenance visit.")
            _due = pr.get("maintenance_due_days")
            needs_today = bool(pr.get("risk_level") == "Critical" or bool(pr.get("maintenance_overdue")))
            needs_week = bool(((pd.notna(_due) and 0 <= int(_due) <= 7) or pr.get("risk_level") == "High")
                              and not needs_today)
            multi_high_risk = high_risk_assets >= 2
            likely_30d = bool(prob is not None and prob >= 60)
        risk_score, risk_level = _room_risk(n, repeat, prob, health, r30)
        visit_priority = "High" if risk_level in ("Critical", "High") else risk_level
        preventive_recommended = "Yes" if (risk_level != "Low" or needs_today or needs_week) else "No"
        likely_repeat = "Yes" if likely_30d else "No"
        rec = {
            "room_label": f"{apt_c}-{rm}", "apartment_code": apt_c, "room": rm,
            "issue_type": top_issue, "distinct_issues": distinct_issues,
            "issue_breakdown": issue_breakdown, "similar_issue_tickets": top_issue_count,
            "predicted_asset": p_type, "predicted_asset_code": p_code, "brand": (brand or "—"),
            "failure_prob": (f"{prob}%" if prob is not None else "—"),
            "_p": (float(prob) if prob is not None else -1.0),
            "confidence": conf, "reason": "; ".join(reason), "recommendation": reco,
            "combined_reco": combined_reco, "high_risk_assets": high_risk_assets, "effort": effort,
            "needs_today": needs_today, "needs_week": needs_week,
            "multi_high_risk": multi_high_risk, "likely_30d": likely_30d,
            "risk_score": risk_score, "risk_level": risk_level,
            "visit_priority": visit_priority, "preventive_recommended": preventive_recommended,
            "likely_repeat": likely_repeat,
            "pred_age": pred_age, "pred_trend": (pred_trend or "—"), "est_failure_window": est_window,
            "ml_segment": pred_segment, "ml_anomaly_score": pred_anom_score,
            "ml_anomaly_flag": pred_anom_flag, "room_max_anomaly": room_max_anom,
            "total_tickets": n, "recent_30d": r30, "recent_90d": r90,
            "repeat_failures": repeat, "asset_ticket_count": asset_tickets,
            "last_maintenance": (last_maint or "—"), "expected_next_maintenance": next_maint,
        }
        rows.append(rec)
        rooms[(apt_c, rm)] = {**rec, "assets": (cand.copy() if cand is not None else pd.DataFrame()),
                              "inspection_priority": inspection}
    tbl = pd.DataFrame(rows).sort_values(["_p", "total_tickets"], ascending=[False, False]).reset_index(drop=True)
    return {"table": tbl, "rooms": rooms}


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

    # ---- ADDITIVE unsupervised ML layer (src/ml): GMM segmentation + Isolation Forest ----
    # Does not touch the rule engine above. Merges ML columns onto the (superset) asset frame
    # so downstream presentation can show rule + Poisson + ML side by side. Fails soft.
    ml_df = pd.DataFrame()
    ml_meta = {}
    try:
        from ml import run_asset_ml
        _base_assets = bv.get("assets") if (bv.get("assets") is not None and not bv["assets"].empty) else built["assets"]
        ml_df = run_asset_ml(_base_assets, built["mapped"])
        ml_meta = dict(ml_df.attrs.get("meta", {}))
        assets_ml = _base_assets.merge(ml_df, on="asset_id", how="left") if not ml_df.empty else _base_assets
    except Exception as exc:  # noqa: BLE001 — ML is optional; never break the rule dashboard
        assets_ml = bv.get("assets") if bv else built["assets"]
        ml_meta = {"error": str(exc)}

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
        # presentation-layer: authoritative room from ticket bed_id (not allocation)
        "room_view": _ticket_room_view(loader, built, assets_ml),
        # additive unsupervised ML (GMM + Isolation Forest); assets_ml = superset + ML cols
        "ml": ml_df,
        "ml_meta": ml_meta,
        "assets_ml": assets_ml,
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
    cov = ex["coverage"]
    bv = ex.get("business") or {}
    if bv.get("assets") is not None and not bv["assets"].empty:
        A = bv["assets"]   # superset: adds reliability_score/level + lifecycle_stage columns
    if ex.get("assets_ml") is not None and not ex["assets_ml"].empty:
        A = ex["assets_ml"]  # superset + additive ML columns (GMM segment, anomaly)
    _HAS_ML = "ml_risk_segment" in A.columns and A["ml_risk_segment"].notna().any()
    _ml_seg = dict(zip(A["asset_code"].astype(str), A["ml_risk_segment"])) if _HAS_ML else {}
    _ml_anom = dict(zip(A["asset_code"].astype(str), A["ml_anomaly_score"])) if _HAS_ML else {}
    _ml_flag = dict(zip(A["asset_code"].astype(str), A["ml_anomaly_flag"])) if _HAS_ML else {}

    # Apartment-summary enrichment sourced from the AUTHORITATIVE ticket room view
    # (maintenance_tickets.bed_id), NOT from asset allocation. Per apartment we surface
    # its highest-probability room prediction.
    _rv = ex.get("room_view") or {"table": pd.DataFrame(), "rooms": {}}
    _rvt = _rv["table"]
    _apt_best = {}
    if not _rvt.empty:
        for _apt_c, _g in _rvt.groupby("apartment_code"):
            _apt_best[_apt_c] = _g.sort_values(["_p", "total_tickets"], ascending=[False, False]).iloc[0]
    _pred_asset_map = {a: r["predicted_asset"] for a, r in _apt_best.items()}
    _pred_loc_map = {a: _loc(a, r["room"]) for a, r in _apt_best.items()}
    _pred_prob_map = {a: r["failure_prob"] for a, r in _apt_best.items()}
    _pred_reason_map = {a: r["reason"] for a, r in _apt_best.items()}
    _top_asset_map = {a: (f"{r['predicted_asset']} ({r['predicted_asset_code']})"
                          if r["predicted_asset_code"] else r["predicted_asset"])
                      for a, r in _apt_best.items()}

    def _apt_pred(df):
        """Attach predicted-asset / location / probability / reason to an apartment-summary
        table (presentation only — reads the engine's existing per-asset scores)."""
        if df is None or df.empty or "apartment_code" not in df.columns:
            return df
        d = df.copy()
        at = d.columns.get_loc("apartment_code") + 1
        d.insert(at, "predicted_asset", d["apartment_code"].map(_pred_asset_map).fillna("—"))
        d.insert(at + 1, "predicted_location", d["apartment_code"].map(_pred_loc_map).fillna("—"))
        d.insert(at + 2, "predicted_failure_prob", d["apartment_code"].map(_pred_prob_map).fillna("—"))
        d.insert(at + 3, "reason", d["apartment_code"].map(_pred_reason_map).fillna("—"))
        return d

    # ---- HERO: full-width Maintenance Health Score ---- #
    _hs = A["health_score"]
    score = int(round(float(_hs.mean()))) if _hs.notna().any() else 0
    if score >= 75:
        _emoji, _status, _color = "🟢", "Healthy", "#2e9e5b"
    elif score >= 60:
        _emoji, _status, _color = "🟡", "Attention Needed", "#d9a406"
    else:
        _emoji, _status, _color = "🔴", "Critical", "#d64545"

    def _hstat(label, value):
        return (f"<div style='flex:1 1 130px;text-align:center;min-width:110px'>"
                f"<div style='font-size:13px;opacity:0.75'>{label}</div>"
                f"<div style='font-size:30px;font-weight:800;line-height:1.1'>{value}</div></div>")

    _tickets = format(cov["total_tickets"], ",")
    _assets = format(cov["total_assets"], ",")
    _covpct = f"{cov['coverage_pct']}%"
    _stats_html = (_hstat("Total Tickets", _tickets)
                   + _hstat("Total Assets", _assets)
                   + _hstat("Coverage %", _covpct))
    st.markdown(
        f"<div style='border:1px solid rgba(127,127,127,0.25);border-radius:14px;"
        f"padding:22px 26px;background:linear-gradient(90deg,{_color}22,rgba(127,127,127,0.05));'>"
        f"<div style='display:flex;flex-wrap:wrap;gap:26px;align-items:center;'>"
        f"<div style='flex:0 0 auto'>"
        f"<div style='font-size:12px;font-weight:700;letter-spacing:0.6px;opacity:0.75'>MAINTENANCE HEALTH SCORE</div>"
        f"<div style='font-size:58px;font-weight:900;line-height:1'>{score}"
        f"<span style='font-size:22px;font-weight:700'> / 100</span></div>"
        f"<div style='font-size:20px;font-weight:800;color:{_color}'>{_emoji} {_status}</div></div>"
        f"<div style='flex:1 1 auto;display:flex;flex-wrap:wrap;gap:20px;justify-content:space-around'>"
        f"{_stats_html}"
        f"</div></div></div>",
        unsafe_allow_html=True)
    st.caption(
        f"ℹ️ Overall score = average of existing per-asset health scores (statistical, read-only; no new model). "
        f"Asset age uses purchase date where available (**{cov['assets_with_purchase_date']:,}** assets); "
        f"the remaining **{cov['assets_using_allocation_date']:,}** are aged from earliest allocation date. "
        f"{cov['tickets_processed']:,} / {cov['total_tickets']:,} tickets processed ({cov['coverage_pct']}%).")
    with st.expander("ℹ️ How this engine works"):
        st.markdown(
            "**Ticket-centric engine.** Every maintenance ticket is attributed to a physical "
            "asset (direct `asset_id`, else Apartment/Room → Asset Allocation → Issue Type → "
            "Asset Type), joined to allocations + assets, and aged by purchase date (else "
            "earliest allocation date). Health, risk, maintenance-due, replacement, failure "
            "trend, room risk and asset-type risk are computed from ticket frequency, recent "
            "failures, age, expected life and maintenance cycle. Statistical (not ML). "
            "Read-only — no database records are modified.")
    st.divider()

    # ---- Today's Priority Rooms (very top — the day's action list) ---- #
    if not _rvt.empty:
        _gap(4)
        st.subheader("🚨 Today's Priority Rooms")
        st.caption("**Ranked by:** Composite Room Risk (ticket volume + repeat failures + predicted failure "
                   "probability + asset health + recent activity). The Top 10 rooms to inspect today. "
                   "_The same room may also appear in later sections under a different ranking metric "
                   "(e.g. Average Apartment Health, Recent Ticket Volume, SLA) — every section states its own metric._")
        _tp = _rvt.sort_values(["risk_score", "_p"], ascending=[False, False]).head(10).reset_index(drop=True).copy()
        _tp.insert(0, "Rank", range(1, len(_tp) + 1))
        _tp_show = _tp.rename(columns={
            "room_label": "Room", "risk_level": "Priority", "predicted_asset": "Predicted Asset",
            "failure_prob": "Failure Probability", "recommendation": "Recommended Action"})
        _table(_tp_show[["Rank", "Room", "Priority", "Predicted Asset", "Failure Probability", "Recommended Action"]],
               "Today's Priority Rooms (Top 10 — inspect today)", "rv_today_top")
        st.divider()

    # ---- Asset Health & Risk (grouped) ---- #
    _gap(4)
    st.subheader("Asset Health & Risk")
    st.caption("**Ranked by:** portfolio counts — assets grouped by their engine risk band (health-score driven).")
    hr = st.columns([3, 1])
    hr[0].markdown(
        f"<div style='border-left:6px solid #3b82c4;background:rgba(127,127,127,0.08);"
        f"padding:12px 16px;border-radius:8px'>"
        f"<div style='font-size:13px;font-weight:600;opacity:0.85'>Asset Risk Distribution</div>"
        f"<div style='font-size:22px;font-weight:800;margin-top:6px;line-height:1.5'>"
        f"🟢 {k['healthy']:,} Healthy &nbsp;&nbsp; 🟡 {k['medium_risk']:,} Medium &nbsp;&nbsp; "
        f"🟠 {k['high_risk']:,} High &nbsp;&nbsp; 🔴 {k['critical']:,} Critical</div></div>",
        unsafe_allow_html=True)
    _insight_card(hr[1], "🔧", "Replacement Due", f"{k['due_replacement']:,}", "#d9a406")
    # ---- ADDITIVE ML view (GMM risk segments + Isolation Forest anomalies) ---- #
    if "ml_risk_segment" in A.columns and A["ml_risk_segment"].notna().any():
        _gap(6)
        _mlmeta = ex.get("ml_meta") or {}
        _agree = _mlmeta.get("if_lof_agreement")
        st.caption("**ML view (unsupervised, additive — does not replace the rule engine):** "
                   "GMM clusters assets into risk segments; Isolation Forest flags anomalous assets. "
                   "Rule risk bands and Poisson probability above are unchanged."
                   + (f" IF/LOF agreement {_agree}%." if _agree is not None else ""))
        _seg = A["ml_risk_segment"].value_counts()
        mc = st.columns(5)
        _insight_card(mc[0], "🟢", "ML Low", int(_seg.get("Low", 0)), "#2e9e5b")
        _insight_card(mc[1], "🟡", "ML Medium", int(_seg.get("Medium", 0)), "#d9a406")
        _insight_card(mc[2], "🟠", "ML High", int(_seg.get("High", 0)), "#e07b39")
        _insight_card(mc[3], "🔴", "ML Critical", int(_seg.get("Critical", 0)), "#d64545")
        _insight_card(mc[4], "⚠️", "Anomalies (IF)", int(A["ml_anomaly_flag"].sum()), "#8b5cf6")
    st.divider()

    # ---- Room Intelligence (per room, from the ticket's own bed selection) ---- #
    st.subheader("🏢 Room Intelligence (room-first — Apartment-Bed)")
    st.caption("Grouped by **Apartment + Bed** (the Vishful room, e.g. `B44-B1`), taken from the ticket's "
               "own `maintenance_tickets.bed_id → bed_code` — the room the technician selected, **not** the "
               "asset's current allocation. Each room is matched to the engine's existing per-asset predictions.")
    if _rvt.empty:
        st.caption("_No ticket room data (bed_id) available._")
    else:
        oc = st.columns(4)
        _insight_card(oc[0], "🔴", "Inspect Today", int(_rvt["needs_today"].sum()), "#d64545")
        _insight_card(oc[1], "🟠", "Preventive This Week", int(_rvt["needs_week"].sum()), "#e07b39")
        _insight_card(oc[2], "🟣", "Rooms w/ Multiple High-Risk Assets", int(_rvt["multi_high_risk"].sum()), "#8b5cf6")
        _insight_card(oc[3], "🟡", "Likely Another Ticket (30d)", int(_rvt["likely_30d"].sum()), "#d9a406")
        _gap(8)
        ic = st.columns(3)
        ic[0].metric("Rooms (Apartment-Bed) with Tickets", f"{len(_rvt):,}")
        ic[1].metric("Rooms Active (30d)", int((_rvt["recent_30d"] > 0).sum()))
        ic[2].metric("High-Confidence Predictions", int((_rvt["confidence"] == "High").sum()))
        _bar(_rvt.head(15)[["room_label", "total_tickets"]], "room_label", "total_tickets",
             "Ticket Volume by Room (Apartment-Bed, top 15)", "rv_chart")
        st.caption("**Ranked by:** predicted 30-day failure probability (highest first). "
                   "The chart above is ranked by Recent Ticket Volume — so the #1 room can differ between the two.")
        _rv_show = _rvt.head(30).rename(columns={
            "room_label": "Room (Apt-Bed)", "apartment_code": "Apartment", "room": "Bed",
            "total_tickets": "Total Tickets", "distinct_issues": "Distinct Issues",
            "issue_type": "Most Frequent Issue", "predicted_asset": "Predicted Asset",
            "predicted_asset_code": "Asset Code", "brand": "Brand",
            "failure_prob": "Failure Probability", "asset_ticket_count": "Historical Asset Tickets",
            "last_maintenance": "Last Maintenance", "expected_next_maintenance": "Expected Next Maintenance",
            "confidence": "Confidence", "recommendation": "Recommendation"})
        _table(_rv_show[["Room (Apt-Bed)", "Apartment", "Bed", "Total Tickets", "Distinct Issues",
                         "Most Frequent Issue", "Predicted Asset", "Asset Code", "Brand",
                         "Failure Probability", "Historical Asset Tickets", "Last Maintenance",
                         "Expected Next Maintenance", "Confidence", "Recommendation"]],
               "Room Intelligence — grouped by Apartment + Bed", "rv_tbl")

        # ---- Room drill-down: full maintenance history + prediction per room ---- #
        _gap(6)
        st.markdown("**🛏 Room & Asset Action Detail** — full maintenance history behind each room prediction "
                    "(expand to view every asset in that room)")
        for _, rr in _rvt.head(12).iterrows():
            info = _rv["rooms"].get((rr["apartment_code"], rr["room"]), {})
            with st.expander(f"🏢 {rr['room_label']} · Issue: {rr['issue_type'] or '—'} "
                             f"· {rr['total_tickets']} ticket(s)"):
                cA, cB = st.columns(2)
                with cA:
                    st.markdown(f"**Room {rr['room_label']}**  \n"
                                f"Apartment: {rr['apartment_code']}  \nBed: {rr['room']}")
                    st.markdown("**Historical Tickets**")
                    st.markdown(f"- Total tickets: **{rr['total_tickets']}**\n"
                                f"- Issues raised (distinct types): **{rr['distinct_issues']}**\n"
                                f"- Most frequent issue: **{rr['issue_type'] or '—'}**\n"
                                f"- Last 30 days: **{rr['recent_30d']}**\n"
                                f"- Last 90 days: **{rr['recent_90d']}**\n"
                                f"- Repeat failures: **{rr['repeat_failures']}**")
                    if info.get("issue_breakdown"):
                        st.markdown("**Issues raised:**")
                        for ib in info["issue_breakdown"]:
                            st.markdown(f"- {ib}")
                with cB:
                    st.markdown("**Prediction**")
                    if rr["predicted_asset_code"]:
                        st.markdown(f"⚠ **{rr['predicted_asset']}** ({rr['predicted_asset_code']})"
                                    + (f" · {rr['brand']}" if rr["brand"] and rr["brand"] != "—" else ""))
                    else:
                        st.markdown(f"{rr['predicted_asset']} — from ticket history "
                                    f"(no scored asset mapped to this room)")
                    _age = "—" if not pd.notna(rr["pred_age"]) else f"{rr['pred_age']} months"
                    st.markdown("**Prediction Evidence**")
                    st.markdown(f"- Historical tickets: **{rr['total_tickets']}**\n"
                                f"- Similar issue tickets: **{rr['similar_issue_tickets']}**\n"
                                f"- Repeat failures: **{rr['repeat_failures']}**\n"
                                f"- Recent tickets (30 days): **{rr['recent_30d']}**\n"
                                f"- Asset age: **{_age}**\n"
                                f"- Failure trend: **{rr['pred_trend']}**\n"
                                f"- Failure probability: **{rr['failure_prob']}**\n"
                                f"- Confidence: **{rr['confidence']}**")
                    st.markdown("**Prediction Timeline**")
                    st.markdown(f"- Last maintenance date: **{rr['last_maintenance']}**\n"
                                f"- Expected next maintenance window: **{rr['expected_next_maintenance']}**\n"
                                f"- Estimated failure window: **{rr['est_failure_window']}**")
                    if rr.get("ml_segment", "—") not in ("—", None):
                        _af = " · ⚠️ anomaly" if rr.get("ml_anomaly_flag") else ""
                        _asc = rr.get("ml_anomaly_score")
                        st.markdown("**ML signal (additive)**")
                        st.markdown(f"- Risk segment (GMM): **{rr['ml_segment']}**\n"
                                    f"- Anomaly score (Isolation Forest): "
                                    f"**{_asc if _asc is not None else '—'}**{_af}")
                _insp = info.get("inspection_priority") or []
                if _insp:
                    st.markdown("**Inspection Priority**")
                    for i, a in enumerate(_insp, 1):
                        _bt = f" · Bed {a['bed']}" if a["bed"] else ""
                        st.markdown(f"**{i}. {_risk_badge(a['risk'])} {a['asset_type']}** "
                                    f"({a['asset_code']}){_bt} — Probability **{a['prob']}%** · health {a['health']}")
                        st.markdown("&nbsp;&nbsp;&nbsp;&nbsp;" + " · ".join(a["reasons"]))
                st.success(f"**Recommendation:** {rr['combined_reco']}")
                st.markdown("**Business Impact**")
                st.markdown(f"- Likely repeat ticket: **{rr['likely_repeat']}**\n"
                            f"- Estimated maintenance effort: **{rr['effort']}**\n"
                            f"- Visit priority: **{rr['visit_priority']}**\n"
                            f"- Preventive inspection recommended: **{rr['preventive_recommended']}**")
    st.divider()

    # ---- Room Maintenance Risk Ranking (executive; presentation prioritisation) ---- #
    _gap(4)
    st.subheader("🏆 Room Maintenance Risk Ranking")
    st.caption("**Ranked by:** Composite Room Risk (ticket volume + repeat failures + predicted failure "
               "probability + asset health + recent activity). Full supervisor worklist. "
               "Prioritisation only — the per-asset prediction scores are unchanged.")
    if _rvt.empty:
        st.caption("_No room data._")
    else:
        _rank = _rvt.sort_values(["risk_score", "_p"], ascending=[False, False]).reset_index(drop=True).copy()
        rc = st.columns(3)
        _insight_card(rc[0], "🔴", "Critical Rooms", int((_rank["risk_level"] == "Critical").sum()), "#d64545")
        _insight_card(rc[1], "🟠", "High-Risk Rooms", int((_rank["risk_level"] == "High").sum()), "#e07b39")
        _insight_card(rc[2], "🟡", "Medium-Risk Rooms", int((_rank["risk_level"] == "Medium").sum()), "#d9a406")
        _gap(8)
        _rank.insert(0, "Rank", range(1, len(_rank) + 1))
        _rank["ml_anomaly"] = _rank["ml_anomaly_flag"].map(lambda v: "⚠️ Yes" if bool(v) else "No") \
            if "ml_anomaly_flag" in _rank.columns else "—"
        _rank_show = _rank.rename(columns={
            "room_label": "Room", "risk_level": "Risk (rule)", "predicted_asset": "Predicted Asset",
            "failure_prob": "Poisson Probability", "total_tickets": "Total Tickets",
            "high_risk_assets": "High-Risk Assets in Room", "effort": "Estimated Maintenance Effort",
            "ml_segment": "ML Segment", "room_max_anomaly": "Max Anomaly (ML)", "ml_anomaly": "Anomaly Flag",
            "recommendation": "Recommendation"})
        _cols = ["Rank", "Room", "Risk (rule)", "ML Segment", "Predicted Asset", "Poisson Probability",
                 "Max Anomaly (ML)", "Anomaly Flag", "Total Tickets", "High-Risk Assets in Room",
                 "Estimated Maintenance Effort", "Recommendation"]
        _table(_rank_show[[c for c in _cols if c in _rank_show.columns]].head(40),
               "Room Maintenance Risk Ranking — rule + Poisson + ML (highest first)", "rv_rank")
    st.divider()

    # ---- Asset-Type Intelligence (ALL tickets) ---- #
    st.subheader("🧩 Asset-Type Intelligence (all tickets)")
    st.caption("**Ranked by:** Failure Frequency (count of issue-tickets mapped to each asset type via the "
               "issue→asset-type bridge).")
    ti = ex["type_intel"]
    if ti is None or ti.empty:
        st.caption("_No asset-type data._")
    else:
        tc = st.columns(1)
        tc[0].metric("High-Risk Asset Types", int(ti["high_risk"].sum()))
        _bar(ti.head(15), "asset_type", "failure_frequency", "Failure Frequency by Asset Type", "ti_freq")
        _xtable(ti[["asset_type", "assets", "failure_frequency", "avg_age_months",
                    "avg_tickets_per_asset", "avg_maintenance_interval_days", "avg_health",
                    "approaching_end_of_life", "high_risk"]],
                "Asset-Type Intelligence Detail", "ti_tbl")
    st.divider()

    # ---- Owner Action List (formerly Executive Dashboard) ---- #
    _gap(4)
    st.subheader("🧭 Owner Action List")
    st.caption("**Ranked by:** Composite Maintenance Priority — the same room-risk composite used by the Risk "
               "Ranking (ticket volume + repeat failures + predicted failure probability + asset health + "
               "recent activity). One prioritised list of what the owner should action first.")
    exe = ex["executive"]
    _ta, _tt = exe["top_risky_apartments"], exe["top_risky_asset_types"]
    _top_apt = _ta.iloc[0]["apartment_code"] if not _ta.empty else "—"
    _top_type = _tt.iloc[0]["asset_type"] if not _tt.empty else "—"
    _top_apt_focus = _top_asset_map.get(_top_apt, "")
    _top_apt_val = (f"{_top_apt}<div style='font-size:13px;font-weight:600;opacity:0.8'>"
                    f"Focus asset: {_top_apt_focus}</div>") if _top_apt_focus else _top_apt
    ec = st.columns(3)
    _insight_card(ec[0], "🔴", "Top Risk Apartment", _top_apt_val, "#d64545")
    _insight_card(ec[1], "🟠", "Top Failing Asset Type", _top_type, "#e07b39")
    _insight_card(ec[2], "🟡", "Assets Likely to Fail (30 Days)", f"{len(exe['likely_fail_30d']):,}", "#d9a406")
    _gap(8)
    if not _rvt.empty:
        _oal = _rvt.sort_values(["risk_score", "_p"], ascending=[False, False]).head(15).reset_index(drop=True).copy()
        _oal.insert(0, "Priority #", range(1, len(_oal) + 1))
        _oal_show = _oal.rename(columns={
            "room_label": "Room", "risk_level": "Priority", "predicted_asset": "Predicted Asset",
            "failure_prob": "Failure Probability", "effort": "Estimated Effort",
            "combined_reco": "Recommended Action"})
        _table(_oal_show[["Priority #", "Room", "Priority", "Predicted Asset", "Failure Probability",
                          "Estimated Effort", "Recommended Action"]],
               "Owner Action List — Top 15 by Composite Maintenance Priority", "exe_oal")
    _gap(6)
    st.caption("Supporting detail — each list below is ranked by its **own** metric:")
    _ta_x = _apt_pred(_ta[["apartment_code", "total_tickets", "demand_trend", "room_risk_level"]].copy()) \
        if not _ta.empty else None
    _xtable(_ta_x, "Top Risk Apartments — ranked by Composite Room Risk", "exe_rooms", "_None._")
    _xtable(_tt[["asset_type", "failure_frequency", "high_risk", "approaching_end_of_life"]]
            if not _tt.empty else None, "Top Failing Asset Types — ranked by Failure Frequency", "exe_types", "_None._")
    _xtable(_add_location(exe["likely_fail_30d"], A),
            "Assets Likely to Fail Next 30 Days — ranked by 30-day Failure Probability", "exe_likely", "_None._")
    st.divider()

    # ---- Visualizations ---- #
    _gap(4)
    st.subheader("Predictive Visualizations")
    st.caption("Descriptive distributions (not ranked): asset-health spread, monthly ticket trend, and "
               "tickets-summed-per-asset-type. The table below is ranked by ticket count.")
    g = st.columns(2)
    with g[0]:
        _bar(ch.get("health_distribution"), "health", "assets", "Asset Health Distribution", "ae_health")
    with g[1]:
        _bar(ch.get("monthly_trend"), "month", "tickets", "Monthly Failure Trend", "ae_monthly")
    _bar(ch.get("failure_by_type"), "asset_type", "tickets", "Failures by Asset Type", "ae_ftype")
    _xtable(_add_location(ch.get("top_failing_assets"), A), "Top Frequently-Failing Assets", "ae_topfail", "_No data._")
    st.divider()

    # ---- Recommendations + alerts ---- #
    _gap(4)
    st.subheader("Predictive Recommendations")
    st.caption("**Ranked by:** 30-day Failure Probability (highest first). Per-asset engine actions "
               "(Replace / Repair Immediately / Schedule Maintenance / Monitor).")
    rd = A["recommendation"].value_counts()
    rc = st.columns(len(rd) if len(rd) else 1)
    for i, (nm, cnt) in enumerate(rd.items()):
        rc[i].metric(nm, int(cnt))
    _gap(6)
    _xtable(_add_location(A[A["recommendation"] != "No Action Needed"][
        ["asset_code", "asset_type", "apartment_code", "bed_code", "risk_level", "recommendation",
         "failure_prob_30d", "age_months", "reason"]].sort_values("failure_prob_30d", ascending=False)),
        "Assets needing action", "ae_recs", "_No actions recommended._")
    alerts = A[A["alerts"].astype(str).str.len() > 0][["asset_code", "asset_type", "apartment_code", "bed_code", "risk_level", "alerts"]]
    _xtable(_add_location(alerts), f"⚠ Active Alerts ({len(alerts)})", "ae_alerts", "_No active alerts._")
    st.divider()

    # ========================================================================= #
    # Business Maintenance Intelligence (items 1-10) — additive
    # ========================================================================= #
    if bv:
        # ---- 2. Apartment Health ---- #
        _gap(4)
        st.subheader("🏢 Apartment Health Score")
        st.caption("**Ranked by:** Average Apartment Health Score (lowest = worst, shown first). "
                   "A whole-building rollup — distinct from the room-level failure-probability ranking above.")
        _bar(bv["apartment_health"].head(15), "apartment_code", "health_score",
             "Apartment Health (lowest = worst; top 15)", "bv_apt_health_chart")
        _ah = _apt_pred(bv["apartment_health"].copy())
        if _HAS_ML and _ah is not None and not _ah.empty and "apartment_code" in _ah.columns:
            _ml_apt = A.groupby("apartment_code").agg(
                ml_anomalies=("ml_anomaly_flag", lambda s: int(s.fillna(False).sum())),
                ml_critical_assets=("ml_risk_segment", lambda s: int((s == "Critical").sum()))).reset_index()
            _ah = _ah.merge(_ml_apt, on="apartment_code", how="left")
            _ah["ml_anomalies"] = _ah["ml_anomalies"].fillna(0).astype(int)
            _ah["ml_critical_assets"] = _ah["ml_critical_assets"].fillna(0).astype(int)
        _xtable(_ah, "Apartment Health Detail (+ ML anomalies / critical-segment assets)", "bv_apt_health")
        st.divider()

        # ---- 6. Failure Hotspots ---- #
        _gap(4)
        st.subheader("🔥 Failure Hotspots")
        st.caption("**Ranked by:** Recent Ticket Volume (density) — apartments, months and floors with the "
                   "most tickets. A volume view, not a health or probability view.")
        hs = bv["hotspots"]
        h1 = st.columns(2)
        with h1[0]:
            _bar(hs["apartment_density"], "apartment_code", "tickets", "Apartment-wise Ticket Density", "bv_hs_apt")
        with h1[1]:
            _bar(hs["monthly_trend"], "month", "tickets", "Monthly Ticket Trend", "bv_hs_month")
        _xtable(_apt_pred(hs["apartment_density"].copy()),
                "Hotspot Apartments — predicted asset to action", "bv_hs_apt_tbl")
        h2 = st.columns(2)
        with h2[0]:
            _bar(hs["type_failures"], "asset_type", "total_failures", "Asset-Type-wise Failures", "bv_hs_type")
        with h2[1]:
            _bar(hs["floor_density"] if not hs["floor_density"].empty else None, "floor", "tickets",
                 "Floor-wise Ticket Density", "bv_hs_floor")
        st.divider()

        # ---- 9. Preventive Maintenance Queue ---- #
        _gap(4)
        st.subheader("🧰 Preventive Maintenance Queue")
        st.caption("**Ranked by:** Asset Health (worst first) among all non-low-risk assets, with priority band.")
        pq = bv["preventive_queue"]
        pcnt = pq["priority"].value_counts().reindex(["Critical", "High", "Medium"]).fillna(0).astype(int) if not pq.empty else None
        if pcnt is not None:
            pc = st.columns(3)
            _insight_card(pc[0], "🔴", "Critical", f"{int(pcnt['Critical']):,}", "#d64545")
            _insight_card(pc[1], "🟠", "High", f"{int(pcnt['High']):,}", "#e07b39")
            _insight_card(pc[2], "🟡", "Medium", f"{int(pcnt['Medium']):,}", "#d9a406")
            _gap(6)
        _pq = _add_location(pq, A)
        if _HAS_ML and _pq is not None and not _pq.empty and "asset_code" in _pq.columns:
            _pq = _pq.copy()
            _pq["ml_segment"] = _pq["asset_code"].astype(str).map(_ml_seg).fillna("—")
            _pq["ml_anomaly_score"] = _pq["asset_code"].astype(str).map(_ml_anom)
            _pq["ml_anomaly"] = _pq["asset_code"].astype(str).map(_ml_flag).map(
                lambda v: "⚠️ Yes" if bool(v) else "No")
        _xtable(_pq, "Prioritized work list (with reason + ML segment/anomaly)", "bv_queue", "_Queue empty._")
        st.divider()

    # ========================================================================= #
    # Business Recommendations (extension 3) — additive
    # ========================================================================= #
    # ---- Business Recommendations (sorted Critical → High → Medium → Low) ---- #
    _gap(4)
    st.subheader("💡 Business Recommendations")
    st.caption("**Ranked by:** Priority (Critical → High → Medium → Low).")
    recs = ex["recommendations"]
    if recs is None or recs.empty:
        st.caption("_No recommendations._")
    else:
        _prio_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
        recs_sorted = recs.sort_values(
            "priority", key=lambda s: s.map(_prio_order).fillna(9), kind="stable")
        for _, r in recs_sorted.iterrows():
            badge = {"Critical": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}.get(r["priority"], "⚪")
            st.markdown(f"**{badge} {r['priority']} — {r['recommendation']}**  \n"
                        f"WHY: {r['why']}  \nEXPECTED IMPACT: {r['expected_impact']}")
        _dl(recs_sorted, "recommendations", "mp_recs")
    st.divider()

    # ========================================================================= #
    # Extension 4 — SLA / Brand / Purchase (additive; auto-hide when data absent)
    # ========================================================================= #
    # ---- 1. Maintenance SLA Dashboard ---- #
    sla = ex.get("sla") or {}
    if sla:
        _gap(4)
        st.subheader("⏱ Maintenance SLA Dashboard")
        st.caption("**Ranked by:** Ticket Volume within each breakdown; SLA-met % measured only on tickets that "
                   "have both a resolved time and an SLA deadline.")
        o = sla["overall"]
        _met = o["sla_met_pct"]
        _viol = o["sla_violated_pct"]
        sg = st.columns(2)
        _insight_card(sg[0], "🟢", "SLA Met",
                      f"{_met}%" if _met is not None else "—", "#2e9e5b")
        _insight_card(sg[1], "🔴", "SLA Violated",
                      f"{_viol}%" if _viol is not None else "—", "#d64545")
        _gap(8)
        s1 = st.columns(3)
        s1[0].metric("Total Tickets", f"{o['total_tickets']:,}")
        s1[1].metric("Open", f"{o['open_tickets']:,}")
        s1[2].metric("Closed", f"{o['closed_tickets']:,}")
        _fmt = lambda x: x if x is not None else "—"
        st.caption(
            f"Resolution (hrs) — avg {_fmt(o['avg_resolution_hours'])} · median {_fmt(o['median_resolution_hours'])} · "
            f"fastest {_fmt(o['fastest_resolution_hours'])} · slowest {_fmt(o['slowest_resolution_hours'])}. "
            f"Resolution measured on {o['resolution_measured_on']} tickets, SLA on {o['sla_measured_on']} "
            f"(tickets with both resolved + deadline).")
        _xtable(sla["by_apartment"], "SLA by Apartment", "sla_apt")
        _xtable(sla["by_issue_type"], "SLA by Issue Type", "sla_issue")
        _xtable(sla["by_asset_type"], "SLA by Asset Type", "sla_type")
        if sla.get("has_technician"):
            _xtable(sla["technician"], "Technician-wise (assigned_to id)", "sla_tech")
        st.divider()

    # ---- 2. Vendor / Brand Analysis ---- #
    brand = ex.get("brand") or {}
    if brand:
        _gap(4)
        st.subheader("🏭 Vendor / Brand Analysis")
        st.caption("**Ranked by:** Average Brand Reliability (best first) and Failure Rate (worst first). "
                   "Covers assets with a populated brand only.")
        _mb = brand["most_reliable"]
        _wb = brand["worst_performing"]
        bc = st.columns(2)
        if not _mb.empty:
            _b = _mb.iloc[0]
            _insight_card(bc[0], "🏆", "Best Performing Brand",
                          f"{_b['brand']}<div style='font-size:13px;font-weight:600;opacity:0.8'>"
                          f"reliability {_b['avg_reliability']} · failure rate {_b['failure_rate']}</div>",
                          "#2e9e5b")
        if not _wb.empty:
            _b = _wb.iloc[0]
            _insight_card(bc[1], "⚠️", "Worst Performing Brand",
                          f"{_b['brand']}<div style='font-size:13px;font-weight:600;opacity:0.8'>"
                          f"reliability {_b['avg_reliability']} · failure rate {_b['failure_rate']}</div>",
                          "#d64545")
        _gap(8)
        _xtable(brand["by_brand"], "Brand Reliability (all brands)", "brand_all")
        _xtable(_mb[["brand", "total_assets", "failure_rate", "avg_reliability"]],
                "Most Reliable Brands", "brand_best")
        _xtable(_wb[["brand", "total_assets", "failure_rate", "avg_reliability"]],
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
        _gap(4)
        st.subheader("🛒 Purchase Recommendation")
        st.caption("**Ranked by:** Purchase Priority (Avoid → Reduce → Monitor → Continue), driven by each "
                   "asset type's average reliability.")
        _rec = pur["recommendation"].astype(str)

        def _pcount(sub):
            return int(_rec.str.contains(sub, case=False, na=False).sum())
        pcol = st.columns(4)
        _insight_card(pcol[0], "🟢", "Continue", f"{_pcount('Continue'):,}", "#2e9e5b")
        _insight_card(pcol[1], "🟡", "Monitor", f"{_pcount('Monitor'):,}", "#d9a406")
        _insight_card(pcol[2], "🟠", "Reduce", f"{_pcount('Reduce'):,}", "#e07b39")
        _insight_card(pcol[3], "🔴", "Avoid", f"{_pcount('Avoid'):,}", "#d64545")
        _gap(8)
        _xtable(pur, "Purchase Guidance by Asset Type", "purchase_rec")
        st.divider()

    # ---- Export ---- #
    _gap(4)
    st.subheader("⬇ Export Reports")
    st.caption("Not ranked — downloadable CSVs. Maintenance Schedule is ordered soonest-due first; "
               "the others are full exports of scored assets.")
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
    st.divider()

    # ---- Asset Search & Profile (final section) ---- #
    _gap(4)
    st.subheader("Asset Search & Profile")
    st.caption("Not ranked — a lookup/filter tool. Results reflect your search and filters; open any asset "
               "for its full profile and ticket timeline.")
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
            "health_score", "risk_level", "failure_prob_30d", "ticket_count", "recommendation",
            "ml_risk_segment", "ml_anomaly_score", "ml_anomaly_flag"]
    _table(_add_location(v[[c for c in cols if c in v.columns]]), f"{len(v)} assets", "ae_search", "_No assets match._")

    if not v.empty:
        pick = st.selectbox("Open asset profile", ["—"] + v["asset_code"].tolist())
        if pick and pick != "—":
            r = v[v["asset_code"] == pick].iloc[0]
            st.markdown(f"### {pick} — {r['asset_type']}  ·  {_risk_badge(r['risk_level'])}")
            st.caption(f"📍 {_loc(r.get('apartment_code'), r.get('bed_code'))}")
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
            _xtable(tl, "Every Maintenance Ticket", "ae_timeline", "_No tickets._", expanded=True)



# --------------------------------------------------------------------------- #
# Page 9 — Maintenance Forecast (ADDITIVE time-series analytics layer)
# Uses src/forecasting.py on the unified 18-month event timeline. Does NOT
# replace the rule engine, Poisson scores, or the ML segmentation/anomaly layer.
# --------------------------------------------------------------------------- #
# Bump this string whenever the forecasting output schema changes so the cached
# result is invalidated (avoids a stale dict that predates new keys like
# avg_ticket_cost / portfolio_backtest / explain).
_FC_VERSION = "v2-intervals-backtest-budget-explain"


@st.cache_resource(show_spinner="Fitting maintenance forecasting models…")
def _forecasts(_version: str = _FC_VERSION):
    from data_loader import DataLoader
    import forecasting as FC
    return FC.run_forecasts(DataLoader(), top_apartments=15)


def _fc_table(items):
    rows = []
    for r in items:
        m = r.get("metrics", {})
        rows.append({
            "Entity": r["label"], "Months": r["n_months"], "Best Model": r["best_model"],
            "Next-Month Forecast": r["forecast_next"],
            "Lower": r.get("forecast_lower", "—"), "Upper": r.get("forecast_upper", "—"),
            "Confidence": r["confidence"],
            "Best MAPE %": (m.get(r["best_model"], {}) or {}).get("MAPE", "—") if r["best_model"] in m else "—",
            "Best MAE": (m.get(r["best_model"], {}) or {}).get("MAE", "—") if r["best_model"] in m else "—",
        })
    return pd.DataFrame(rows)


def _fc_band_chart(p):
    """History line + next-month forecast point with a shaded confidence band.
    Uses Altair for the band; falls back to a multi-line chart if Altair is unavailable."""
    hist = p["history"].rename("value").rename_axis("month").reset_index()
    hist["kind"] = "history"
    fpt = pd.DataFrame({"month": [p["next_period"]], "value": [p["forecast_next"]],
                        "kind": ["forecast"],
                        "lower": [p.get("forecast_lower")], "upper": [p.get("forecast_upper")]})
    try:
        import altair as alt
        base = alt.Chart(hist).mark_line().encode(x="month:T", y="value:Q")
        band = alt.Chart(fpt).mark_area(opacity=0.25, color="#e07b39").encode(
            x="month:T", y="lower:Q", y2="upper:Q")
        pt = alt.Chart(fpt).mark_point(size=90, color="#d64545", filled=True).encode(x="month:T", y="value:Q")
        st.altair_chart(band + base + pt, use_container_width=True)
    except Exception:
        _h = p["history"].rename("history").to_frame()
        _f = pd.DataFrame({"forecast": [p["forecast_next"]], "lower": [p.get("forecast_lower")],
                           "upper": [p.get("forecast_upper")]}, index=[p["next_period"]])
        st.line_chart(pd.concat([_h, _f], axis=1))


def page_forecast():
    st.header("📈 Maintenance Forecast")
    st.info(
        "**Additive time-series layer.** Forecasts next month's maintenance ticket **workload** "
        "at Portfolio, Asset-Type and Apartment levels from the **unified 18-month event timeline** "
        "(`created_at` else `resolved_at` else `closed_at`). Compares **Seasonal Naive, Holt-Winters (ETS), "
        "SARIMA and Prophet** on a chronological hold-out and picks the lowest MAPE (MAE tie-break). "
        "It forecasts volume — not which specific asset fails. The rule engine, Poisson scores and "
        "ML segmentation are unchanged.  \n_Prophet is compared on the Portfolio series; per-entity "
        "(asset-type / apartment) uses the three fast models (Seasonal Naive / ETS / SARIMA) for "
        "responsiveness — Prophet's fit is ~15s/series._"
    )
    try:
        fc = _forecasts()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Forecasting failed: {exc}")
        return

    p = fc.get("portfolio")
    if not p:
        st.warning("No maintenance history available to forecast.")
        return
    st.caption(f"History through **{fc.get('as_of')}** · next forecast month "
               f"**{p['next_period'].date() if p.get('next_period') is not None else '—'}**.")

    # ---- Portfolio (with prediction interval + shaded band) ---- #
    _gap(4)
    st.subheader("Portfolio — Next-Month Workload")
    pc = st.columns(5)
    pc[0].metric("Forecast", f"{p['forecast_next']:.0f}")
    pc[1].metric("Lower Bound", f"{p.get('forecast_lower', 0):.0f}")
    pc[2].metric("Upper Bound", f"{p.get('forecast_upper', 0):.0f}")
    pc[3].metric("Interval", f"{p.get('confidence_pct') or '—'}%")
    pc[4].metric("Best Model", p["best_model"])
    _fc_band_chart(p)
    if p.get("metrics"):
        _mt = pd.DataFrame(p["metrics"]).T.rename_axis("Model").reset_index()
        _table(_mt, "Model comparison on hold-out (lower = better)", "fc_port_metrics")
    st.divider()

    # ---- Apartment ---- #
    _gap(4)
    st.subheader("Apartment — Next-Month Workload (top 15 by volume)")
    st.caption("**Ranked by:** forecast volume. Sparse apartments fall back to naive and are flagged Low confidence.")
    _at = _fc_table(fc.get("by_apartment", []))
    if not _at.empty:
        _bar(_at.sort_values("Next-Month Forecast", ascending=False).head(15),
             "Entity", "Next-Month Forecast", "Forecast next-month tickets by apartment", "fc_apt_bar")
        _table(_at.sort_values("Next-Month Forecast", ascending=False), "Apartment Forecasts", "fc_apt_tbl")
    else:
        st.caption("_No apartment series._")
    st.caption("Caveat: 18–19 monthly points support Seasonal-Naive/ETS well and are marginal for "
               "SARIMA/Prophet yearly seasonality (which prefer ≥24 months). Short per-entity series "
               "(<6 months) use a naive forecast and are marked Low confidence.")


# --------------------------------------------------------------------------- #
# Shell
# --------------------------------------------------------------------------- #
def page_maintenance_investigation():
    """Read-only investigation page (Phase 4); logic lives in its own module."""
    from maintenance_investigation import render
    render()


def page_sla_performance():
    """SLA Performance Analytics — standalone operations page (own module)."""
    from sla_analytics import render
    render()


PAGES = {
    "Inventory Overview": page_inventory_overview,
    "Customer Recommendation": page_recommendation,
    "Blocked Rooms": page_blocked_rooms,
    "Revenue Leakage": page_revenue_leakage,
    "Occupancy Analytics": page_occupancy,
    "Room Search": page_room_search,
    "Asset Predictive Analytics": page_asset_predictive,
    "Maintenance Forecast": page_forecast,
    "Maintenance Investigation": page_maintenance_investigation,
    "SLA Performance": page_sla_performance,
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