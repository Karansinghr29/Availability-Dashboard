"""
recommendation_engine.py  —  PHASE 2 STEP 4
===========================================

End-to-end CUSTOMER room recommendation. Reads data ONLY through data_loader,
builds the Room Inventory (single source of truth), resolves the customer's
state, and ranks available ROOMS via the P1-P5 ladder.

Output is always ROOM-level: a room + a specific available bed. Apartment
occupancy/revenue are internal ranking signals only.

Public API
----------
recommend_rooms(loader, *, city, bed_type=None, budget=None, top_n=3) -> DataFrame
    Columns == ranking.OUTPUT_COLUMNS:
      apartment_code, room_code, bed_code, bed_type, monthly_rent,
      current_occupancy_pct, available_beds, occupant_states, demand_score, reason
    The result also carries `.attrs["detected_state"]`.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from preprocessing import build_room_inventory
from ranking import rank_recommendations
from india_locations import is_india_state
from inventory_views import (
    active_inventory,
    active_vacant_rooms,
    prepare_inventory,
)
from roommate_matching import (
    STATIC_CITY_STATE,
    _states_of,
    assert_candidates_match_customer_gender,
    customer_match_label,
    ensure_gender_allowed_column,
    filter_rooms_for_customer_gender,
    occupants_state_summary,
    resolve_state,
)
from utils import (
    build_city_state_map,
    print_candidate_pool_after_inactive_filter,
    print_inventory_lifecycle_summary,
)


def _apartment_metrics(room_inventory: pd.DataFrame) -> pd.DataFrame:
    """Internal ranking signal: occupancy % aggregated per apartment.

    Revenue is intentionally excluded — it is not used in recommendation ranking.
    """
    grp = room_inventory.groupby("apartment_code")
    metrics = grp.agg(
        _cap=("capacity", "sum"),
        _occ=("occupied_beds", "sum"),
    ).reset_index()
    metrics["apartment_occupancy_pct"] = (
        100 * metrics["_occ"] / metrics["_cap"].where(metrics["_cap"] > 0)
    ).round(2)
    return metrics[["apartment_code", "apartment_occupancy_pct"]]


def _enrich_display_metadata(
    result: pd.DataFrame, room_inventory: pd.DataFrame, customer_state: Optional[str]
) -> pd.DataFrame:
    """Fix card metadata using room occupants, falling back to apartment occupants.

    Display only — does not change ranking. Never overwrites inventory
    ``gender_allowed`` (that value is what the Customer Gender filter used).
    """
    if result is None or result.empty:
        return result
    out = result.copy()
    inv = room_inventory
    inv_by_room = {}
    if inv is not None and not inv.empty:
        for _, row in inv.iterrows():
            inv_by_room[(str(row["apartment_code"]), str(row["room_code"]))] = row

    states_list, similarity_list, gender_allowed_list = [], [], []
    for _, rec in out.iterrows():
        key = (str(rec["apartment_code"]), str(rec["room_code"]))
        room = inv_by_room.get(key)
        # ROOM-level occupants ONLY — the same dataset used by same_state_roommate
        # and the card's "Current Occupant States". No apartment-aggregate fallback:
        # same-state matching is a room concept, so Occupant States, Same-state
        # roommate and Similarity must all reflect the same room (never claim a
        # same-state occupant the room does not actually have).
        room_states = _states_of(room["occupant_states"]) if room is not None else []

        states_list.append(occupants_state_summary(room_states))
        similarity_list.append(customer_match_label(room_states, customer_state))

        # Prefer inventory Gender Allowed (filter source of truth).
        inv_ga = None
        if room is not None and "gender_allowed" in room.index:
            inv_ga = room.get("gender_allowed")
        if inv_ga is None or (isinstance(inv_ga, float) and pd.isna(inv_ga)) or str(inv_ga).strip() in ("", "nan"):
            inv_ga = rec.get("gender_allowed")
        if inv_ga is None or (isinstance(inv_ga, float) and pd.isna(inv_ga)) or str(inv_ga).strip() in ("", "nan"):
            inv_ga = "Unknown"
        gender_allowed_list.append(str(inv_ga).strip())

    out["occupants_state_summary"] = states_list
    out["match_similarity"] = similarity_list
    out["gender_allowed"] = gender_allowed_list
    if "occupants_gender_label" in out.columns:
        out = out.drop(columns=["occupants_gender_label"])
    return out


def recommend_rooms(
    loader,
    *,
    city: str,
    bed_type: Optional[str] = None,
    budget: Optional[float] = None,
    gender: Optional[str] = None,
    occupation: Optional[str] = None,
    age: Optional[float] = None,
    use_compatibility_in_ranking: bool = False,
    top_n: int = 3,
    room_inventory: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return the Top-N ROOM recommendations for a customer enquiry.

    Candidate pipeline (gender filter BEFORE ranking):
      Active Inventory → drop Inactive → available beds
      → Customer Gender filter → (bed type + location inside ranking) → Top-N
    """
    tenants = None
    try:
        tenants = loader.tenants()
    except Exception:  # noqa: BLE001
        pass

    if room_inventory is None:
        beds_master = None
        try:
            beds_master = loader.beds_master()
        except Exception:  # noqa: BLE001
            beds_master = None
        bed_map = None
        try:
            bed_map = loader.bed_map()
        except Exception:  # noqa: BLE001
            bed_map = None
        room_inventory = build_room_inventory(
            loader.bookings(), tenants=tenants, beds_master=beds_master, bed_map=bed_map
        )

    # Shared lifecycle views — Active vacant only is recommendable.
    room_inventory = prepare_inventory(room_inventory)

    # Step 1 — active vacant inventory summary (print only; no calc changes).
    print_inventory_lifecycle_summary(room_inventory)

    learned_map = build_city_state_map(tenants) if tenants is not None else {}
    state = resolve_state(city, learned_map, STATIC_CITY_STATE)

    # Step 2 — ACTIVE inventory only (inactive apartments never enter the pool).
    eligible = active_inventory(room_inventory)
    print_candidate_pool_after_inactive_filter(room_inventory, eligible)

    # Step 3 — available beds only.
    candidates = active_vacant_rooms(room_inventory)
    candidates = ensure_gender_allowed_column(candidates)
    before_gender = len(candidates)

    # Step 4 — Customer Gender filter BEFORE ranking.
    print(f"\nCustomer Gender : {gender}")
    print(f"Candidate rooms BEFORE gender filter : {before_gender}")
    candidates = filter_rooms_for_customer_gender(candidates, gender)
    print(f"Candidate rooms AFTER gender filter  : {len(candidates)}")
    if candidates is not None and not candidates.empty:
        print("\nRoom | Gender Allowed")
        for _, r in candidates.iterrows():
            print(f"{r['room_code']} | {r.get('gender_allowed', 'Unknown')}")
    else:
        print("(no rooms remaining after gender filter)")

    assert_candidates_match_customer_gender(candidates, gender)

    # Step 5 — ranking (bed-type filter -> same-state tier -> historical need).
    # Ranking/scoring is unchanged; only the fallback AFTER filtering changes below.
    apt_metrics = _apartment_metrics(eligible)

    def _rank(bt):
        return rank_recommendations(
            candidates,
            state=state,
            apartment_metrics=apt_metrics,
            bed_type=bt,
            budget=None,  # Budget removed from Customer Recommendation UI.
            customer_gender=gender,
            customer_occupation=occupation,
            customer_age=age,
            use_compatibility_in_ranking=use_compatibility_in_ranking,
            top_n=top_n,
        )

    # Priority 1 (same-state + requested type) and Priority 2 (requested type only,
    # historical order) both come from ranking the requested-bed-type pool.
    result = _rank(bed_type)

    # Priority 3 (fallback after filtering): the requested bed type has NO vacancy
    # at all -> relax the bed-type filter and show ANY available type (same-state
    # tier + historical order preserved). No scoring change; only the filter relaxed.
    bed_type_fallback = False
    bed_type_shown = None
    if bed_type and (result is None or result.empty):
        result = _rank(None)
        if result is not None and not result.empty:
            bed_type_fallback = True
            bed_type_shown = ", ".join(
                sorted(str(x) for x in result["bed_type"].dropna().unique())
            )

    result = _enrich_display_metadata(result, room_inventory, state)

    # Final safety: cards must never show conflicting Gender Allowed.
    if result is not None and not result.empty and gender:
        assert_candidates_match_customer_gender(result, gender)

    same_state_in_result = bool(
        result is not None
        and not result.empty
        and (result["same_state_roommate"].astype(str) == "Yes").any()
    )

    result.attrs["detected_state"] = state
    result.attrs["query"] = {
        "city": city,
        "bed_type": bed_type,
        "customer_gender": gender,
    }
    result.attrs["input_kind"] = "state" if is_india_state(city) else "city"
    result.attrs["candidate_rooms_after_inactive_filter"] = len(eligible)
    result.attrs["active_vacant_rooms"] = before_gender
    result.attrs["candidate_rooms_after_gender_filter"] = len(candidates)
    # Fallback signalling for the dashboard message (display only).
    result.attrs["bed_type_requested"] = bed_type
    result.attrs["bed_type_fallback"] = bed_type_fallback
    result.attrs["bed_type_shown"] = bed_type_shown
    result.attrs["same_state_in_result"] = same_state_in_result
    return result


# --------------------------------------------------------------------------- #
# Demo: 5 real customer enquiries.
# --------------------------------------------------------------------------- #

def _demo() -> None:
    import logging

    from data_loader import DataLoader

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    loader = DataLoader()
    tenants = loader.tenants()
    beds_master = None
    try:
        beds_master = loader.beds_master()
    except Exception:  # noqa: BLE001
        beds_master = None
    bed_map = None
    try:
        bed_map = loader.bed_map()
    except Exception:  # noqa: BLE001
        bed_map = None
    inventory = build_room_inventory(
        loader.bookings(), tenants=tenants, beds_master=beds_master, bed_map=bed_map
    )

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)

    examples = [
        {"city": "Coimbatore", "bed_type": None, "budget": None},
        {"city": "Kochi", "bed_type": "Triple", "budget": 14000},
        {"city": "Hyderabad", "bed_type": "Double", "budget": 15000},
        {"city": "Mumbai", "bed_type": None, "budget": 20000},
        {"city": "Gurgaon", "bed_type": "Single", "budget": None},
    ]

    from pathlib import Path

    saved = []
    for ex in examples:
        rec = recommend_rooms(loader, room_inventory=inventory, **ex, top_n=3)
        print("\n" + "=" * 78)
        print(f"Customer City: {ex['city']}  |  Bed Type: {ex['bed_type']}  |  Budget: {ex['budget']}")
        print(f"Detected State: {rec.attrs.get('detected_state')}")
        if rec.empty:
            print("No available rooms match.")
            continue
        for i, row in rec.iterrows():
            print(f"\n  #{i + 1}  {row['apartment_code']} / {row['room_code']} / bed {row['bed_code']} "
                  f"({row['bed_type']})")
            print(f"      Rent: {row['monthly_rent']} | Occupancy: {row['current_occupancy_pct']}% | "
                  f"Available beds: {row['available_beds']} | "
                  f"Compatibility: {row['compatibility_score']} | Demand: {row['demand_score']}")
            print(f"      Roommates' states: {row['occupant_states'] or '-'}")
            print(f"      Reason: {row['reason']}")
        tagged = rec.copy()
        tagged.insert(0, "query_city", ex["city"])
        tagged.insert(1, "detected_state", rec.attrs.get("detected_state"))
        tagged.insert(2, "rank", range(1, len(tagged) + 1))
        saved.append(tagged)

    if saved:
        out_dir = Path(__file__).resolve().parents[1] / "outputs"
        out_dir.mkdir(exist_ok=True)
        pd.concat(saved, ignore_index=True).to_csv(
            out_dir / "recommendations_demo.csv", index=False
        )
        print(f"\nSaved demo recommendations -> {out_dir / 'recommendations_demo.csv'}")


if __name__ == "__main__":
    _demo()
