"""
ranking.py  —  PHASE 2 STEP 4 (part)
====================================

Ranks candidate ROOMS into the final Top-N room recommendations using the
business priority ladder, bed-type preference, budget signal and demand score.

The recommendation unit is always a ROOM + a specific available bed. Apartment
occupancy/revenue are internal ranking signals only. `room_code` is opaque
(never derived here); everything comes from the Room Inventory.

Ranking precedence (matches the approved business requirement):
  Priority 1  available bed + roommate from same state (same ROOM only).
              Ordered by apartment occupancy (asc), then demand, then budget.
  Priority 2  If no Priority-1 room exists, lowest-occupancy available rooms.
              Messaging must say clearly that no same-state roommate is
              available — never pretend this is a same-state match.
  Priority 3  When a same-state roommate bed opens later, Priority 1 wins
              again automatically (no code change required — tier recompute).

  Bed type: filter to the requested type; if none available, fall back to the
            NEAREST available type (by sharing size).
  Budget:   closer rent-to-budget ranks higher — the FINAL tie-breaker only,
            never a filter.

IMPORTANT: revenue is NOT used anywhere in recommendation ranking. The engine
optimises roommate compatibility, occupancy and fill strategy. Revenue stays in
the analytics modules (pricing, leakage, management) only.

Final ranking (approved / ACTIVE):
    Priority 1  lowest-occupancy apartment + same-state room
    Priority 2  highest demand score
    Priority 3  budget match

compatibility_score is computed and returned as a DISPLAY-ONLY column. It does
NOT affect ranking unless `use_compatibility_in_ranking=True` (a future business
toggle). When enabled, compatibility slots in right after occupancy.

Within a tier the ACTIVE key order is:
    occupancy (asc) -> demand_score (desc) -> budget distance (asc)
    -> recent occupancy (desc) -> recent fill (desc) -> recent vacancy days (asc).

Output (one row per recommended ROOM):
    apartment_code, room_code, bed_code, bed_type, monthly_rent,
    current_occupancy_pct, available_beds, occupant_states, demand_score, reason
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from roommate_matching import (
    _states_of,
    compatibility_scores,
    customer_match_label,
    occupants_gender_label,
    occupants_state_summary,
)
from utils import normalize_text

OUTPUT_COLUMNS = (
    "apartment_code", "room_code", "bed_code", "bed_type", "gender_allowed",
    "lifecycle_status", "monthly_rent", "current_occupancy_pct", "available_beds",
    "occupant_states", "occupants_gender_label", "occupants_state_summary",
    "same_state_roommate", "same_state_note",
    "match_similarity", "compatibility_score", "demand_score", "reason",
)

# Ordinal sizes used ONLY to pick the "nearest" bed type when the requested one
# is unavailable. This is bed-type ordering metadata, not room capacity (which
# stays dynamically derived in the Room Inventory).
_BED_TYPE_ORDINAL = {"single": 1, "double": 2, "triple": 3, "quad": 4, "quadruple": 4}


def _first_available_bed(cell) -> Optional[str]:
    if isinstance(cell, list):
        return cell[0] if cell else None
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return None
    parts = [s.strip() for s in str(cell).split(";") if s.strip()]
    return parts[0] if parts else None


def _apply_bed_type_filter(pool: pd.DataFrame, requested: Optional[str]):
    """Return (filtered_pool, mode) where mode in {'any','exact','nearest'}."""
    if not requested:
        return pool, "any"
    req = normalize_text(requested)
    work = pool.copy()
    work["_bt"] = work["bed_type"].map(normalize_text)

    if (work["_bt"] == req).any():
        return work[work["_bt"] == req], "exact"

    req_ord = _BED_TYPE_ORDINAL.get(req)
    avail_types = [t for t in work["_bt"].dropna().unique()]
    if req_ord is None or not avail_types:
        return work, "any"
    # Nearest by |size difference|; tie -> smaller (step down, usually cheaper).
    nearest = sorted(
        avail_types, key=lambda t: (abs(_BED_TYPE_ORDINAL.get(t, 999) - req_ord),
                                    _BED_TYPE_ORDINAL.get(t, 999))
    )[0]
    return work[work["_bt"] == nearest], "nearest"


def _same_state_note(same_state: bool, customer_state: Optional[str]) -> str:
    """Display note when Same-state roommate is No (never pretend Priority 1)."""
    if same_state:
        return ""
    if customer_state:
        return f"No roommate from {customer_state} is currently available."
    return "No same-state roommate is currently available."


def _build_reason(row, *, medians, budget, bed_mode, requested_bed_type,
                  customer_state=None, compatibility_in_ranking=False) -> str:
    parts = []
    states = [s for s in _states_of(row.get("occupant_states"))]

    if row["_same_state"]:
        # Priority 1 — available bed + roommate from same state (same room).
        parts.append(f"Same-state roommates ({', '.join(states)})" if states
                     else "Same-state roommates")
        if row["apartment_occupancy_pct"] <= medians["apt_occ"]:
            parts.append("Lowest-occupancy apartment")
    else:
        # Priority 2 — lowest-occupancy fallback; do NOT pretend same-state.
        note = _same_state_note(False, customer_state)
        if note:
            parts.append(note)
        if row["current_occupancy_pct"] <= medians["room_occ"]:
            parts.append("Low occupancy room")

    # Compatibility is display-only unless explicitly enabled in ranking.
    if (compatibility_in_ranking
            and pd.notna(row.get("compatibility_score"))
            and row["compatibility_score"] >= medians["compat"]):
        parts.append("High roommate compatibility")

    if bed_mode == "nearest" and requested_bed_type:
        parts.append(f"Nearest bed type to '{requested_bed_type}'")
    elif bed_mode == "exact" and requested_bed_type:
        parts.append(f"Preferred bed type ({row['bed_type']})")

    if budget is not None and pd.notna(row.get("current_rent")):
        rent = float(row["current_rent"])
        parts.append("Budget match" if rent <= float(budget) else "Slightly above budget")

    if pd.notna(row.get("demand_score")) and row["demand_score"] >= medians["demand"]:
        parts.append("High demand score")

    return " + ".join(parts) if parts else "Available room"


def rank_recommendations(
    candidate_rooms: pd.DataFrame,
    *,
    state: Optional[str] = None,
    apartment_metrics: Optional[pd.DataFrame] = None,
    bed_type: Optional[str] = None,
    budget: Optional[float] = None,
    customer_gender: Optional[str] = None,
    customer_occupation: Optional[str] = None,
    customer_age: Optional[float] = None,
    use_compatibility_in_ranking: bool = False,
    top_n: int = 3,
) -> pd.DataFrame:
    """Rank available ROOMS and return the Top-N with reasons."""
    pool = candidate_rooms[candidate_rooms["available_beds"] >= 1].copy()
    if pool.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Step 6 — bed-type filter (with nearest-type fallback).
    pool, bed_mode = _apply_bed_type_filter(pool, bed_type)
    if pool.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Same-state flag (room qualification + tier-1 membership).
    target = normalize_text(state) if state else None
    pool["_same_state"] = pool["occupant_states"].apply(
        lambda c: bool(target) and target in {normalize_text(s) for s in _states_of(c)}
    )

    # Apartment occupancy for tier-1 ranking (revenue is intentionally ignored).
    if apartment_metrics is not None:
        cols = ["apartment_code", "apartment_occupancy_pct"]
        pool = pool.merge(apartment_metrics[cols], on="apartment_code", how="left")
    if "apartment_occupancy_pct" not in pool:
        pool["apartment_occupancy_pct"] = pool["current_occupancy_pct"]

    # Priority tiers + occupancy ordering column (revenue is NOT used here).
    pool["_tier"] = pool["_same_state"].map({True: 1, False: 2})
    # tier 1 uses APARTMENT occupancy (lowest-occupancy apartments first);
    # tier 2 uses ROOM occupancy.
    pool["_biz_occ"] = pool["apartment_occupancy_pct"].where(
        pool["_tier"] == 1, pool["current_occupancy_pct"]
    )

    # Budget distance — FINAL tie-breaker only.
    if budget is not None:
        pool["_budget_dist"] = (pool["current_rent"].astype(float) - float(budget)).abs()
        pool["_budget_dist"] = pool["_budget_dist"].fillna(pool["_budget_dist"].max())
    else:
        pool["_budget_dist"] = 0.0

    # Roommate compatibility (0-100). Computed for DISPLAY ONLY by default; it is
    # NOT part of the ranking unless `use_compatibility_in_ranking` is enabled
    # (a future business toggle). The engine stays modular either way.
    pool["compatibility_score"] = compatibility_scores(
        pool,
        customer_state=state,
        customer_gender=customer_gender,
        customer_occupation=customer_occupation,
        customer_age=customer_age,
    )
    pool["_compat"] = pool["compatibility_score"].fillna(-1)

    # Demand (business fill strategy) + recent tie-breaks (missing -> worst).
    pool["_demand"] = pool["demand_score"].fillna(-1)
    pool["_recent_occ"] = pool["recent_occupancy_pct"].fillna(-1)
    pool["_recent_fill"] = pool["recent_fill_rate"].fillna(-1)
    pool["_vac"] = pool["average_vacancy_days_recent"].fillna(float("inf"))

    # Active ranking: tier -> occupancy -> demand -> budget -> recent tie-breaks.
    # Revenue is excluded; budget is the final tie-breaker. Compatibility is
    # inserted right after occupancy ONLY when explicitly enabled (future toggle).
    sort_by = ["_tier", "_biz_occ"]
    sort_asc = [True, True]
    if use_compatibility_in_ranking:
        sort_by.append("_compat")
        sort_asc.append(False)
    sort_by += ["_demand", "_budget_dist", "_recent_occ", "_recent_fill", "_vac"]
    sort_asc += [False, True, False, False, True]
    pool = pool.sort_values(by=sort_by, ascending=sort_asc).reset_index(drop=True)

    medians = {
        "apt_occ": pool["apartment_occupancy_pct"].median(),
        "room_occ": pool["current_occupancy_pct"].median(),
        "compat": pool["compatibility_score"].median(),
        "demand": pool["demand_score"].median(),
    }

    top = pool.head(top_n).copy()
    rows = []
    for _, r in top.iterrows():
        genders = r["occupant_genders"] if "occupant_genders" in r.index else []
        states = r["occupant_states"] if "occupant_states" in r.index else []
        same = bool(r["_same_state"])
        ga = r.get("gender_allowed")
        if ga is None or (isinstance(ga, float) and pd.isna(ga)) or str(ga).strip() in ("", "nan"):
            ga_label = "Unknown"
        else:
            ga_label = "Mixed" if str(ga).strip() == "Unisex" else str(ga).strip()
        rows.append(
            {
                "apartment_code": r["apartment_code"],
                "room_code": r["room_code"],
                "bed_code": _first_available_bed(r["available_bed_codes"]),
                "bed_type": r["bed_type"],
                "gender_allowed": ga_label,
                "lifecycle_status": r.get("lifecycle_status", "Unknown"),
                "monthly_rent": r.get("current_rent"),
                "current_occupancy_pct": r["current_occupancy_pct"],
                "available_beds": int(r["available_beds"]),
                "occupant_states": ", ".join(_states_of(states)),
                "occupants_gender_label": occupants_gender_label(genders),
                "occupants_state_summary": occupants_state_summary(states),
                "same_state_roommate": "Yes" if same else "No",
                "same_state_note": _same_state_note(same, state),
                "match_similarity": customer_match_label(states, state),
                "compatibility_score": r.get("compatibility_score"),
                "demand_score": r.get("demand_score"),
                "reason": _build_reason(
                    r, medians=medians, budget=budget,
                    bed_mode=bed_mode, requested_bed_type=bed_type,
                    customer_state=state,
                    compatibility_in_ranking=use_compatibility_in_ranking,
                ),
            }
        )
    return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))
