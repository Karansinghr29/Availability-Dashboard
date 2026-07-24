"""TEMPORARY debug harness — Sivagangai recommendation pipeline trace.

Does NOT change ranking business logic. Prints every stage the user requested.
Run: python _debug_recommendation_pipeline.py
"""
from __future__ import annotations

import pandas as pd

from data_loader import DataLoader
from india_locations import location_key
from preprocessing import build_room_inventory
from ranking import (
    _apply_bed_type_filter,
    rank_recommendations,
)
from recommendation_engine import (
    _apartment_metrics,
    _exclude_inactive_for_recommendation,
)
from roommate_matching import (
    STATIC_CITY_STATE,
    _states_of,
    resolve_state,
    same_state_rooms,
)
from utils import annotate_lifecycle_status, build_city_state_map, normalize_text

CUSTOMER_LOCATION = "Sivagangai"


def _sep(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> None:
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.max_colwidth", 80)

    loader = DataLoader()
    tenants = loader.tenants()
    beds_master = None
    try:
        beds_master = loader.beds_master()
    except Exception:  # noqa: BLE001
        beds_master = None
    inv = build_room_inventory(
        loader.bookings(), tenants=tenants, beds_master=beds_master
    )
    inv = annotate_lifecycle_status(inv)

    learned_map = build_city_state_map(tenants) if tenants is not None else {}

    # ------------------------------------------------------------------ STEP 1
    _sep("STEP 1 — resolve_state()")
    original = CUSTOMER_LOCATION
    normalized = location_key(original)
    detected = resolve_state(original, learned_map, STATIC_CITY_STATE)
    in_learned = normalized in {location_key(k) for k in learned_map}
    in_static = normalized in STATIC_CITY_STATE
    print(f"Original input:    {original!r}")
    print(f"Normalized input:  {normalized!r}")
    print(f"Detected state:    {detected!r}")
    print(f"In learned map:    {in_learned}")
    print(f"In static map:     {in_static}")
    if in_static:
        print(f"Static map value:  {STATIC_CITY_STATE[normalized]!r}")
    if in_learned:
        print(f"Learned map value: {learned_map.get(original) or learned_map.get(normalized)!r}")

    # ------------------------------------------------------------------ STEP 2
    _sep("STEP 2 — Occupied rooms with occupant_states + available_beds")
    print(f"Room inventory rows: {len(inv)}")
    print(f"Has occupant_states column: {'occupant_states' in inv.columns}")
    if "occupant_states" in inv.columns:
        nonempty = inv["occupant_states"].apply(lambda c: len(_states_of(c)) > 0)
        print(f"Rooms with non-empty occupant_states: {int(nonempty.sum())} / {len(inv)}")
        print(f"Rooms with empty occupant_states:     {int((~nonempty).sum())}")

    occupied = inv[inv["occupied_beds"] > 0].copy()
    print(f"\nOccupied rooms (occupied_beds > 0): {len(occupied)}")
    rows = []
    for _, r in occupied.iterrows():
        states = _states_of(r["occupant_states"])
        rows.append(
            {
                "room_code": r["room_code"],
                "apartment_code": r["apartment_code"],
                "occupant_states": ", ".join(states) if states else "(EMPTY)",
                "available_beds": int(r["available_beds"]),
                "occupied_beds": int(r["occupied_beds"]),
            }
        )
    occ_df = pd.DataFrame(rows)
    print(occ_df.to_string(index=False))

    # State frequency across inventory
    all_states = []
    for cell in inv["occupant_states"]:
        all_states.extend(_states_of(cell))
    from collections import Counter

    print("\nOccupant state frequency (all rooms):")
    for st, n in Counter(all_states).most_common():
        print(f"  {st}: {n}")
    print(f"  (empty rooms / no state): {int((~nonempty).sum())}")

    # ------------------------------------------------------------------ STEP 3
    _sep("STEP 3 — same_state_rooms() output")
    eligible = _exclude_inactive_for_recommendation(inv)
    candidates = eligible[eligible["available_beds"] >= 1].copy()
    print(f"Eligible after inactive exclusion: {len(eligible)}")
    print(f"Candidates with available_beds>=1: {len(candidates)}")
    from config import get_temporarily_inactive_apartments
    inactive_codes = get_temporarily_inactive_apartments()
    still = eligible["apartment_code"].astype(str).str.strip().str.upper().isin(inactive_codes).any()
    print(f"Config-inactive apartments still in eligible? {still} (codes={sorted(inactive_codes)})")

    matched = same_state_rooms(candidates, detected) if detected else candidates.iloc[0:0]
    print(f"\nsame_state_rooms(state={detected!r}) count: {len(matched)}")
    if matched.empty:
        print("*** NO SAME-STATE ROOMS MATCHED ***")
        # Also try full inventory (including full rooms) for diagnosis
        all_match = same_state_rooms(
            eligible.assign(available_beds=eligible["available_beds"].clip(lower=1)),
            detected,
        ) if detected else eligible.iloc[0:0]
        # better: check occupancy match ignoring availability
        target = normalize_text(detected) if detected else None
        any_occ = eligible[
            eligible["occupant_states"].apply(
                lambda c: bool(target)
                and target in {normalize_text(s) for s in _states_of(c)}
            )
        ]
        print(f"Rooms with Tamil Nadu occupant (any availability): {len(any_occ)}")
        if not any_occ.empty:
            print(
                any_occ[
                    ["room_code", "occupant_states", "available_beds", "occupied_beds"]
                ].to_string(index=False)
            )
    else:
        show = matched.copy()
        show["matching_state"] = detected
        show["occupant_states_str"] = show["occupant_states"].apply(
            lambda c: ", ".join(_states_of(c))
        )
        print(
            show[["room_code", "occupant_states_str", "matching_state", "available_beds"]].to_string(
                index=False
            )
        )

    # ------------------------------------------------------------------ STEP 4
    _sep("STEP 4 — Immediately after same-state flag / tier assignment (inside ranking)")
    # Mirror ranking.py steps exactly (do not change logic)
    pool = candidates.copy()
    pool, bed_mode = _apply_bed_type_filter(pool, None)
    print(f"After bed-type filter (mode={bed_mode}): {len(pool)} rooms")

    target = normalize_text(detected) if detected else None
    pool["_same_state"] = pool["occupant_states"].apply(
        lambda c: bool(target)
        and target in {normalize_text(s) for s in _states_of(c)}
    )
    apt = _apartment_metrics(eligible)
    pool = pool.merge(apt, on="apartment_code", how="left")
    pool["_tier"] = pool["_same_state"].map({True: 1, False: 2})
    pool["_biz_occ"] = pool["apartment_occupancy_pct"].where(
        pool["_tier"] == 1, pool["current_occupancy_pct"]
    )
    pool["_budget_dist"] = 0.0

    print(f"tier=1 count: {int((pool['_tier'] == 1).sum())}")
    print(f"tier=2 count: {int((pool['_tier'] == 2).sum())}")
    print(f"_same_state True count: {int(pool['_same_state'].sum())}")

    step4 = pool[
        [
            "room_code",
            "_tier",
            "current_occupancy_pct",
            "apartment_occupancy_pct",
            "demand_score",
            "_budget_dist",
            "occupant_states",
            "_same_state",
        ]
    ].copy()
    step4 = step4.rename(
        columns={
            "_tier": "tier",
            "current_occupancy_pct": "occupancy",
            "_budget_dist": "budget_distance",
        }
    )
    step4["occupant_states"] = step4["occupant_states"].apply(
        lambda c: ", ".join(_states_of(c)) if _states_of(c) else "(EMPTY)"
    )
    # Show tier-1 first, then a sample of tier-2
    t1 = step4[step4["tier"] == 1]
    t2 = step4[step4["tier"] == 2].nsmallest(15, "occupancy")
    print("\n--- Tier 1 (same-state) ---")
    print(t1.to_string(index=False) if not t1.empty else "(none)")
    print("\n--- Tier 2 sample (lowest room occupancy) ---")
    print(t2.to_string(index=False))

    # ------------------------------------------------------------------ STEP 5
    _sep("STEP 5 — Immediately BEFORE sorting (full sort keys)")
    before = pool[
        [
            "room_code",
            "apartment_code",
            "_tier",
            "_biz_occ",
            "demand_score",
            "_budget_dist",
            "_same_state",
            "occupant_states",
            "available_beds",
            "current_occupancy_pct",
            "apartment_occupancy_pct",
        ]
    ].copy()
    before["occupant_states"] = before["occupant_states"].apply(
        lambda c: ", ".join(_states_of(c)) if _states_of(c) else "(EMPTY)"
    )
    print(f"Rows before sort: {len(before)}")
    print("Unique apartments in pool:", sorted(before["apartment_code"].astype(str).unique()))
    # Show lowest occupancy candidates (what wins if all tier 2)
    print("\nLowest 10 by (_tier, _biz_occ, -demand):")
    preview = before.sort_values(
        by=["_tier", "_biz_occ", "demand_score"],
        ascending=[True, True, False],
    ).head(10)
    print(preview.to_string(index=False))

    # ------------------------------------------------------------------ STEP 6
    _sep("STEP 6 — Immediately AFTER sorting")
    pool["_demand"] = pool["demand_score"].fillna(-1)
    pool["_recent_occ"] = pool["recent_occupancy_pct"].fillna(-1)
    pool["_recent_fill"] = pool["recent_fill_rate"].fillna(-1)
    pool["_vac"] = pool["average_vacancy_days_recent"].fillna(float("inf"))
    sort_by = ["_tier", "_biz_occ", "_demand", "_budget_dist", "_recent_occ", "_recent_fill", "_vac"]
    sort_asc = [True, True, False, True, False, False, True]
    sorted_pool = pool.sort_values(by=sort_by, ascending=sort_asc).reset_index(drop=True)
    after = sorted_pool.head(10)[
        [
            "room_code",
            "apartment_code",
            "_tier",
            "_biz_occ",
            "demand_score",
            "_budget_dist",
            "_same_state",
            "occupant_states",
            "current_occupancy_pct",
            "apartment_occupancy_pct",
        ]
    ].copy()
    after["occupant_states"] = after["occupant_states"].apply(
        lambda c: ", ".join(_states_of(c)) if _states_of(c) else "(EMPTY)"
    )
    print(after.to_string(index=False))

    rank1 = sorted_pool.iloc[0]
    print("\n--- Why Rank 1 ---")
    print(f"room_code:              {rank1['room_code']}")
    print(f"apartment_code:         {rank1['apartment_code']}")
    print(f"_tier:                  {rank1['_tier']}  (1=same-state, 2=other)")
    print(f"_same_state:            {rank1['_same_state']}")
    print(f"occupant_states:        {_states_of(rank1['occupant_states'])}")
    print(f"_biz_occ:               {rank1['_biz_occ']}")
    print(f"current_occupancy_pct:  {rank1['current_occupancy_pct']}")
    print(f"apartment_occupancy_pct:{rank1['apartment_occupancy_pct']}")
    print(f"demand_score:           {rank1['demand_score']}")
    print(f"available_beds:         {rank1['available_beds']}")
    if int(rank1["_tier"]) == 2 and int((pool["_tier"] == 1).sum()) == 0:
        print(
            "\nROOT CAUSE CANDIDATE: ZERO tier-1 (same-state) rooms in the candidate pool.\n"
            "With no same-state matches, EVERY customer location collapses to the SAME\n"
            "tier-2 sort: lowest room occupancy → highest demand → ...\n"
            "Customer state is computed but never changes the ordered result."
        )
    elif int(rank1["_tier"]) == 1:
        print(
            "\nRank 1 is a genuine same-state (tier 1) room. Location IS influencing ranking."
        )
    else:
        print(
            f"\nRank 1 is tier 2 even though {int((pool['_tier'] == 1).sum())} tier-1 rooms exist — "
            "investigate sort / merge overwrite."
        )

    # Official engine output for comparison
    _sep("OFFICIAL recommend_rooms() Top-3")
    from recommendation_engine import recommend_rooms

    rec = recommend_rooms(
        loader, city=CUSTOMER_LOCATION, room_inventory=inv, top_n=3
    )
    print(f"detected_state attrs: {rec.attrs.get('detected_state')}")
    if rec.empty:
        print("EMPTY RESULT")
    else:
        print(
            rec[
                [
                    "apartment_code",
                    "room_code",
                    "occupant_states",
                    "match_similarity",
                    "demand_score",
                    "current_occupancy_pct",
                    "reason",
                ]
            ].to_string(index=False)
        )

    # Cross-city comparison: does location change Top-3?
    _sep("CROSS-CHECK — Top-3 room_codes for several locations")
    for loc in [
        "Sivagangai",
        "Coimbatore",
        "Kochi",
        "Hyderabad",
        "Mumbai",
        "Tamil Nadu",
        "Kerala",
        "Andhra Pradesh",
        "Uttar Pradesh",
    ]:
        r = recommend_rooms(loader, city=loc, room_inventory=inv, top_n=3)
        codes = list(r["room_code"]) if not r.empty else []
        st = r.attrs.get("detected_state")
        print(f"{loc:20s} → state={st!s:20s} rooms={codes}")


if __name__ == "__main__":
    main()
