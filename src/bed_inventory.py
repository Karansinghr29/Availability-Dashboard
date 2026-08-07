"""Bed-level vacant inventory keyed by ``bed_id`` (read-only display helper).

ADDITIVE / display-only. This module does NOT change — and must never change —
the recommendation eligibility rule, ``active_vacant_beds``,
``build_room_inventory``, the bed-map derivation, or any vacancy/status logic.
It only provides a per-PHYSICAL-bed (``bed_id``) view of CURRENTLY VACANT beds
for the Room Search display, so that two bed records that share the same
``(apartment_code, bed_code)`` but have different ``bed_id`` are BOTH shown and
never deduplicated.

Vacancy rule (identical to the live bed-map / ``_reconstruct_bed_map``): a bed is
VACANT when it has NO open ``tenant_allotment`` — i.e. no allotment row with
``actual_exit_date`` null AND ``staying_status`` in {Staying, On-Notice, Notice,
Booked}. Inactive-apartment beds (apartment status Not-Active, e.g. A22) are
excluded, matching the existing recommendation inventory. Nothing is merged,
deduplicated, or renamed.
"""

from __future__ import annotations

import re
from typing import Optional

import pandas as pd

# Same open-stay vocabulary the loader uses to decide a bed is HELD.
_OPEN_STAY_STATUSES = frozenset({"staying", "on-notice", "notice", "booked"})
_NOT_ACTIVE = frozenset({"not-active", "inactive", "disabled", "blocked"})

_COLUMNS = [
    "bed_id", "apartment_code", "room_code", "bed_code", "bed_type",
    "toilet_type", "monthly_rate", "occupancy_status", "occupancy_history",
    "duplicate_bed_code",
]


def _s(v) -> str:
    return str(v).strip() if pd.notna(v) else ""


def _room_code(apartment_code: str, bed_code: str) -> str:
    """apartment_code + first alphabetic run of bed_code (e.g. A33 + B2 -> A33-B).

    Mirrors ``preprocessing._resolve_room_code`` so this list filters cleanly
    against the room-inventory ``room_code`` used elsewhere on the page.
    """
    apt = _s(apartment_code)
    m = re.match(r"^([A-Za-z]+)", _s(bed_code))
    return f"{apt}-{m.group(1)}" if (apt and m) else apt


def vacant_beds_by_bed_id(loader) -> pd.DataFrame:
    """One row per CURRENTLY VACANT physical bed, keyed by ``bed_id``.

    No deduplication by ``(apartment_code, bed_code)``: if two vacant bed records
    share a bed_code but differ by ``bed_id`` they are both returned. Inactive
    apartments are excluded (unchanged existing rule).
    """
    beds = loader.beds_master()
    if beds is None or beds.empty or "id" not in beds.columns:
        return pd.DataFrame(columns=_COLUMNS)

    try:
        al = loader.allotments()
    except Exception:  # noqa: BLE001
        al = pd.DataFrame()

    # --- Held (non-vacant) bed_ids from OPEN allotments + occupancy history ---
    open_ids: set = set()
    ever_ids: set = set()
    if al is not None and not al.empty and "bed_id" in al.columns:
        ever_ids = {_s(b) for b in al["bed_id"] if _s(b)}
        stay = al.get("staying_status", pd.Series(index=al.index, dtype=object)).map(
            lambda v: _s(v).lower()
        )
        exit_null = (
            al["actual_exit_date"].isna()
            if "actual_exit_date" in al.columns
            else pd.Series(True, index=al.index)
        )
        open_mask = exit_null & stay.isin(_OPEN_STAY_STATUSES)
        open_ids = {_s(b) for b in al.loc[open_mask, "bed_id"] if _s(b)}

    # apartment active/inactive: prefer the enriched _apt_status, else status.
    apt_status_col = "_apt_status" if "_apt_status" in beds.columns else "status"

    rate_col = next((c for c in ("current_rate", "bed_rate", "monthly_rental")
                     if c in beds.columns), None)

    # Count (apartment_code, bed_code) occurrences among VACANT beds to flag dups.
    rows = []
    for _, r in beds.iterrows():
        bid = _s(r.get("id"))
        apt = _s(r.get("apartment_code"))
        bed = _s(r.get("bed_code"))
        if not bid or not apt or not bed:
            continue
        if _s(r.get(apt_status_col)).lower().replace(" ", "-") in _NOT_ACTIVE:
            continue  # inactive-apartment exclusion (unchanged)
        if bid in open_ids:
            continue  # held -> not vacant
        rate = pd.to_numeric(r.get(rate_col), errors="coerce") if rate_col else pd.NA
        rows.append({
            "bed_id": bid,
            "apartment_code": apt,
            "room_code": _room_code(apt, bed),
            "bed_code": bed,
            "bed_type": _s(r.get("bed_type")) or pd.NA,
            "toilet_type": _s(r.get("toilet_type")) or pd.NA,
            "monthly_rate": rate,
            "occupancy_status": "Vacant",
            "occupancy_history": "Never occupied" if bid not in ever_ids else "Previously occupied",
        })

    out = pd.DataFrame(rows, columns=_COLUMNS)
    if out.empty:
        return out
    dup_counts = out.groupby(["apartment_code", "bed_code"])["bed_id"].transform("size")
    out["duplicate_bed_code"] = dup_counts > 1
    return out.sort_values(["apartment_code", "room_code", "bed_code", "bed_id"]).reset_index(drop=True)
