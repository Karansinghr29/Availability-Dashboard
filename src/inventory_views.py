"""
inventory_views.py
==================

Single source of truth for ACTIVE vs INACTIVE inventory views.

All pages / recommendation filtering MUST use these helpers so lifecycle
logic is never duplicated. Status comes from ``annotate_lifecycle_status``
(DB columns + ``config/inactive_apartments.json`` overrides).

Two inventories
---------------
ACTIVE VACANT BEDS
  - apartment_status == Active
  - available bed(s)
  - eligible for recommendation

INACTIVE INVENTORY
  - apartment_status == Inactive
  - informational only — never recommended
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from utils import (
    LIFECYCLE_STATUS_ACTIVE,
    LIFECYCLE_STATUS_INACTIVE,
    LIFECYCLE_STATUS_UNKNOWN,
    annotate_lifecycle_status,
)

# Produced by this module — never feed back into annotate_lifecycle_status
# (name collisions with source fields would mark every room Inactive).
# NOTE: ``bed_status`` is a beds_master source column (Live / Not-Active) and
# must NOT be stripped or overwritten here.
_VIEW_COLUMNS = (
    "apartment_status",
    "room_status",
    "recommendable",
)


def _parse_bed_codes(cell) -> list:
    if isinstance(cell, list):
        return [str(b).strip() for b in cell if str(b).strip()]
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    return [s.strip() for s in str(cell).split(";") if s.strip()]


def prepare_inventory(room_inventory: pd.DataFrame) -> pd.DataFrame:
    """Annotate room inventory with shared lifecycle status columns.

    Adds / refreshes:
      lifecycle_status  — room-level Active / Inactive / Unknown
      apartment_status  — Inactive if the apartment has any Inactive room
                          (or is listed in config/inactive_apartments.json)
      room_status       — Inactive if room/apartment is Inactive; else Active
      recommendable     — True only for Active apartments

    Preserves beds_master ``bed_status`` (Live / Not-Active) and
    ``bed_lifecycle_status`` when present.
    """
    from config import get_temporarily_inactive_apartments

    if room_inventory is None or not isinstance(room_inventory, pd.DataFrame):
        return room_inventory
    if room_inventory.empty:
        out = room_inventory.copy()
        for col, default in (
            ("lifecycle_status", LIFECYCLE_STATUS_UNKNOWN),
            ("apartment_status", LIFECYCLE_STATUS_ACTIVE),
            ("room_status", LIFECYCLE_STATUS_ACTIVE),
            ("recommendable", False),
        ):
            out[col] = default
        return out

    # Preserve master bed fields across view annotation.
    master_bed_status = (
        room_inventory["bed_status"].copy()
        if "bed_status" in room_inventory.columns
        else None
    )
    master_bed_lifecycle = (
        room_inventory["bed_lifecycle_status"].copy()
        if "bed_lifecycle_status" in room_inventory.columns
        else None
    )

    # Strip prior view columns so they cannot poison lifecycle derivation.
    base = room_inventory.drop(
        columns=[c for c in _VIEW_COLUMNS if c in room_inventory.columns],
        errors="ignore",
    )
    out = annotate_lifecycle_status(base)

    # Apartments that are Inactive: any room already Inactive, plus config list.
    inactive_from_rooms = {
        str(c).strip().upper()
        for c in out.loc[
            out["lifecycle_status"].astype(str) == LIFECYCLE_STATUS_INACTIVE,
            "apartment_code",
        ]
    }
    inactive_apts = inactive_from_rooms | set(get_temporarily_inactive_apartments())

    apt_codes = out["apartment_code"].astype(str).str.strip().str.upper()
    out["apartment_status"] = [
        LIFECYCLE_STATUS_INACTIVE if c in inactive_apts else LIFECYCLE_STATUS_ACTIVE
        for c in apt_codes.tolist()
    ]

    room_inactive = out["lifecycle_status"].astype(str) == LIFECYCLE_STATUS_INACTIVE
    apt_mask = out["apartment_status"].astype(str) == LIFECYCLE_STATUS_INACTIVE
    out["room_status"] = [
        LIFECYCLE_STATUS_INACTIVE if (r or a) else LIFECYCLE_STATUS_ACTIVE
        for r, a in zip(room_inactive.tolist(), apt_mask.tolist())
    ]
    out.loc[apt_mask, "lifecycle_status"] = LIFECYCLE_STATUS_INACTIVE
    out["recommendable"] = ~apt_mask

    # Restore beds_master bed_status / lifecycle (do not replace with Active).
    if master_bed_status is not None:
        out["bed_status"] = master_bed_status.values
    if master_bed_lifecycle is not None:
        out["bed_lifecycle_status"] = master_bed_lifecycle.values
    return out


def active_inventory(room_inventory: pd.DataFrame) -> pd.DataFrame:
    """Rooms in Active apartments only."""
    inv = prepare_inventory(room_inventory)
    if inv is None or inv.empty:
        return inv
    return inv.loc[inv["apartment_status"] != LIFECYCLE_STATUS_INACTIVE].copy()


def inactive_inventory(room_inventory: pd.DataFrame) -> pd.DataFrame:
    """Rooms in Inactive apartments only (informational)."""
    inv = prepare_inventory(room_inventory)
    if inv is None or inv.empty:
        return inv
    return inv.loc[inv["apartment_status"] == LIFECYCLE_STATUS_INACTIVE].copy()


def active_vacant_rooms(room_inventory: pd.DataFrame) -> pd.DataFrame:
    """Active-apartment rooms that have at least one available bed (recommendable)."""
    active = active_inventory(room_inventory)
    if active is None or active.empty or "available_beds" not in active.columns:
        return active.iloc[0:0].copy() if active is not None else pd.DataFrame()
    return active.loc[active["available_beds"].fillna(0) >= 1].copy()


def active_vacant_beds(room_inventory: pd.DataFrame) -> pd.DataFrame:
    """One row per ACTIVE vacant bed (dashboard card / recommendation pool).

    Columns: apartment_code, room_code, bed_code, current_rent, occupancy,
    demand_score, apartment_status, room_status, bed_status.
    """
    rooms = active_vacant_rooms(room_inventory)
    if rooms is None or rooms.empty:
        return pd.DataFrame(
            columns=[
                "apartment_code", "room_code", "bed_code", "current_rent",
                "current_occupancy_pct", "demand_score",
                "apartment_status", "room_status", "bed_status",
            ]
        )

    rows = []
    for _, r in rooms.iterrows():
        codes = _parse_bed_codes(r.get("available_bed_codes"))
        if not codes:
            n = int(r.get("available_beds") or 0)
            codes = [f"(bed {i + 1})" for i in range(n)]
        for bed in codes:
            rows.append(
                {
                    "apartment_code": r["apartment_code"],
                    "room_code": r["room_code"],
                    "bed_code": bed,
                    "current_rent": r.get("current_rent"),
                    "current_occupancy_pct": r.get("current_occupancy_pct"),
                    "demand_score": r.get("demand_score"),
                    "apartment_status": r.get("apartment_status"),
                    "room_status": r.get("room_status"),
                    "bed_status": r.get("bed_status") or LIFECYCLE_STATUS_ACTIVE,
                    "bed_type": r.get("bed_type"),
                }
            )
    return pd.DataFrame(rows)


def active_vacant_summary(room_inventory: pd.DataFrame) -> dict:
    """Display-only KPIs for ACTIVE vacant inventory (never includes Inactive)."""
    vacant_rooms = active_vacant_rooms(room_inventory)
    vacant_beds = active_vacant_beds(room_inventory)
    n_rooms = 0 if vacant_rooms is None or vacant_rooms.empty else len(vacant_rooms)
    n_beds = 0 if vacant_beds is None or vacant_beds.empty else len(vacant_beds)
    n_apts = 0
    if vacant_rooms is not None and not vacant_rooms.empty:
        n_apts = int(vacant_rooms["apartment_code"].astype(str).nunique())
    return {
        "total_active_vacant_beds": n_beds,
        "total_active_vacant_rooms": n_rooms,
        "total_active_apartments_with_vacancies": n_apts,
    }


def vacant_beds_by_roommate_state(room_inventory: pd.DataFrame) -> pd.DataFrame:
    """Display-only: active vacant beds grouped by current roommate home state.

    Columns: State, Vacant Beds, Rooms, Apartments.
    ``Vacant Beds`` lists exact bed labels (``{apartment}-{bed}``, e.g. B33-C1)
    from ``available_bed_codes`` — one entry per vacant bed, not a count.
    Empty rooms (no occupant states) → ``Vacant / No Occupants``.
    Inactive apartments are never included.
    """
    from collections import defaultdict

    from roommate_matching import _states_of

    empty_label = "Vacant / No Occupants"
    rooms = active_vacant_rooms(room_inventory)
    empty_cols = ["State", "Vacant Beds", "Rooms", "Apartments"]
    if rooms is None or rooms.empty:
        return pd.DataFrame(columns=empty_cols)

    bed_labels: dict = defaultdict(set)
    room_sets: dict = defaultdict(set)
    apt_sets: dict = defaultdict(set)

    for _, r in rooms.iterrows():
        codes = _parse_bed_codes(r.get("available_bed_codes"))
        if not codes:
            n = int(r.get("available_beds") or 0)
            if n <= 0:
                continue
            codes = [f"(bed {i + 1})" for i in range(n)]
        room_code = str(r.get("room_code", "")).strip()
        apt_code = str(r.get("apartment_code", "")).strip()
        states = [s for s in _states_of(r.get("occupant_states")) if s]
        keys = [empty_label] if not states else list(dict.fromkeys(states))
        for bed in codes:
            bed = str(bed).strip()
            if not bed:
                continue
            label = f"{apt_code}-{bed}" if apt_code else bed
            for key in keys:
                bed_labels[key].add(label)
                if room_code:
                    room_sets[key].add(room_code)
                if apt_code:
                    apt_sets[key].add(apt_code)

    if not bed_labels:
        return pd.DataFrame(columns=empty_cols)

    def _join(codes: set) -> str:
        return ", ".join(sorted(codes)) if codes else "—"

    rows = []
    # Non-empty states first (by vacant-bed count desc), empty bucket last.
    ranked = sorted(
        ((k, v) for k, v in bed_labels.items() if k != empty_label),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )
    for state, labels in ranked:
        rows.append(
            {
                "State": state,
                "Vacant Beds": _join(labels),
                "Rooms": _join(room_sets[state]),
                "Apartments": _join(apt_sets[state]),
            }
        )
    if empty_label in bed_labels:
        rows.append(
            {
                "State": empty_label,
                "Vacant Beds": _join(bed_labels[empty_label]),
                "Rooms": _join(room_sets[empty_label]),
                "Apartments": _join(apt_sets[empty_label]),
            }
        )
    return pd.DataFrame(rows)


def print_active_vacant_inventory_summary(room_inventory: pd.DataFrame) -> dict:
    """Console verification that UI inventory is ACTIVE-only."""
    summary = active_vacant_summary(room_inventory)
    by_state = vacant_beds_by_roommate_state(room_inventory)
    print("\n" + "=" * 60)
    print("ACTIVE VACANT INVENTORY SUMMARY")
    print("=" * 60)
    print(f"Total Active Apartments: {summary['total_active_apartments_with_vacancies']}")
    print(f"Total Active Vacant Rooms: {summary['total_active_vacant_rooms']}")
    print(f"Total Active Vacant Beds: {summary['total_active_vacant_beds']}")
    print("-" * 60)
    print(f"{'State':<28} Vacant Beds (codes)  Rooms")
    if by_state is None or by_state.empty:
        print("(none)")
    else:
        for _, row in by_state.iterrows():
            print(
                f"{str(row['State']):<28} {row['Vacant Beds']}  |  {row['Rooms']}"
            )
    print("=" * 60)
    return summary


def has_same_state_vacant_rooms(
    room_inventory: pd.DataFrame, customer_state: Optional[str]
) -> bool:
    """True when at least one active vacant room has a roommate from customer_state."""
    from roommate_matching import same_state_rooms

    if not customer_state:
        return False
    vacant = active_vacant_rooms(room_inventory)
    if vacant is None or vacant.empty:
        return False
    return not same_state_rooms(vacant, customer_state).empty


def inactive_inventory_summary(room_inventory: pd.DataFrame) -> pd.DataFrame:
    """Apartment-level inactive summary for dashboard / occupancy section.

    Columns: apartment_code, apartment_status, rooms, beds (capacity),
    available_beds, current_occupancy_pct.
    """
    inactive = inactive_inventory(room_inventory)
    if inactive is None or inactive.empty:
        return pd.DataFrame(
            columns=[
                "apartment_code", "apartment_status", "rooms", "beds",
                "available_beds", "current_occupancy_pct",
            ]
        )

    rows = []
    for apt, g in inactive.groupby("apartment_code"):
        cap = float(g["capacity"].fillna(0).sum()) if "capacity" in g.columns else 0.0
        occ = float(g["occupied_beds"].fillna(0).sum()) if "occupied_beds" in g.columns else 0.0
        avail = float(g["available_beds"].fillna(0).sum()) if "available_beds" in g.columns else 0.0
        occ_pct = round(100.0 * occ / cap, 2) if cap > 0 else 0.0
        rows.append(
            {
                "apartment_code": apt,
                "apartment_status": LIFECYCLE_STATUS_INACTIVE,
                "rooms": len(g),
                "beds": int(cap),
                "available_beds": int(avail),
                "current_occupancy_pct": occ_pct,
            }
        )
    return pd.DataFrame(rows).sort_values("apartment_code").reset_index(drop=True)


def attach_apartment_status_to_blocked(
    blocked: pd.DataFrame, room_inventory: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Merge shared apartment_status onto blocked-room rows."""
    if blocked is None or blocked.empty:
        return blocked
    out = blocked.copy()
    if room_inventory is not None and not room_inventory.empty:
        inv = prepare_inventory(room_inventory)
        apt_map = (
            inv[["apartment_code", "apartment_status"]]
            .drop_duplicates("apartment_code")
        )
        if "apartment_status" in out.columns:
            out = out.drop(columns=["apartment_status"])
        out = out.merge(apt_map, on="apartment_code", how="left")
    if "apartment_status" not in out.columns:
        out["apartment_status"] = LIFECYCLE_STATUS_ACTIVE
    out["apartment_status"] = out["apartment_status"].fillna(LIFECYCLE_STATUS_ACTIVE)

    # Enforce inactive presentation (never Critical/Delayed/Normal).
    inactive = out["apartment_status"] == LIFECYCLE_STATUS_INACTIVE
    if "lifecycle_status" in out.columns:
        inactive = inactive | (out["lifecycle_status"] == LIFECYCLE_STATUS_INACTIVE)
    out.loc[inactive, "apartment_status"] = LIFECYCLE_STATUS_INACTIVE
    # Sync lifecycle_status to apartment_status so revenue aggregators see Inactive.
    out["lifecycle_status"] = out["apartment_status"]
    if "vacancy_status" in out.columns:
        out.loc[inactive, "vacancy_status"] = LIFECYCLE_STATUS_INACTIVE
    if "estimated_revenue_loss" in out.columns:
        out.loc[inactive, "estimated_revenue_loss"] = 0.0
    if "reason" in out.columns:
        out.loc[inactive, "reason"] = "Apartment is inactive"
    return out
