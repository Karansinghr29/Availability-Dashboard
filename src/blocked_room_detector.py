"""
blocked_room_detector.py  —  PHASE 5: BLOCKED ROOM DETECTION & REVENUE LEAKAGE
==============================================================================

A completely separate analytics module. It does NOT change the Recommendation
Engine, Room Inventory, Occupancy History or the Fill-Time engine — it only
CONSUMES their existing outputs:

  * room_inventory      (preprocessing.build_room_inventory)
  * occupancy_history   (preprocessing.build_occupancy_history)
  * room fill-time      (occupancy_analysis.vacancy_report / fill_time)

Business goal
-------------
Detect beds that stay vacant much longer than the room's OWN historical refill
pattern — a business-loss / "possible occupancy block" signal, not merely
"vacant". Everything is room/bed level and fully dynamic (no hardcoded room
codes, rents, fill days or apartment names). Works unchanged when PostgreSQL
replaces the CSVs, because it only touches DataFrames.

Public API
----------
detect_blocked_rooms(room_inventory, occupancy_history=None, vacancy=None, ...)
    -> DataFrame with the dashboard-ready columns (see OUTPUT_COLUMNS).
generate_and_save() -> builds inputs via the existing modules and writes
    outputs/blocked_rooms.csv.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from occupancy_analysis import (
    vacancy_report as build_vacancy_report,
    fill_time as build_fill_time,
    _classify_vacancy,
    _DEFAULT_CRITICAL_MULTIPLIER,
)
from inventory_views import prepare_inventory
from utils import (
    ACTIVE_STATUS_COLUMNS,
    LIFECYCLE_STATUS_ACTIVE,
    LIFECYCLE_STATUS_INACTIVE,
)

logger = logging.getLogger(__name__)

# Average days per month (calendar constant) used to prorate monthly rent into a
# daily loss. Not a business threshold — purely a units conversion.
_DAYS_PER_MONTH = 30.44

# Severity ordering for sorting ("Critical first"). Higher-severity = smaller key.
_SEVERITY_ORDER = {
    "Critical": 0, "Delayed": 1, "Normal": 2, "Unknown": 3, "Inactive": 4,
}

OUTPUT_COLUMNS = (
    "apartment_code", "room_code", "bed_code", "bed_type", "capacity",
    "occupied_beds", "available_beds", "lifecycle_status", "apartment_status",
    "current_vacant_days", "expected_fill_days", "vacancy_status",
    "estimated_revenue_loss", "demand_score", "reason",
)

# Inventory columns pulled in per room (bed rent + occupancy context).
_INVENTORY_COLUMNS = (
    "apartment_code", "room_code", "bed_type", "capacity", "occupied_beds",
    "available_beds", "current_rent", "demand_score",
)


def _estimate_revenue_loss(bed_rent, vacant_days) -> Optional[float]:
    """Room/bed-level leakage = monthly rent x (vacant_days / days-per-month)."""
    if bed_rent is None or pd.isna(bed_rent) or pd.isna(vacant_days):
        return None
    return round(float(bed_rent) * (float(vacant_days) / _DAYS_PER_MONTH), 2)


def _fmt_days(value) -> str:
    return "?" if pd.isna(value) else f"{float(value):.0f}"


def _build_reason(row) -> str:
    """Explainable, business-friendly reason string (fully data-driven)."""
    status = row["vacancy_status"]
    days = row["current_vacant_days"]
    expected = row["expected_fill_days"]
    parts = []

    if status == "Unknown":
        parts.append(f"Vacant {_fmt_days(days)} days. No historical refill data to compare.")
    elif status == "Normal":
        parts.append(f"Vacant {_fmt_days(days)} days (within expected {_fmt_days(expected)} days).")
    else:  # Delayed / Critical
        parts.append(
            f"Vacant {_fmt_days(days)} days (expected {_fmt_days(expected)} days). "
            "Possible occupancy block."
        )

    # Last remaining bed only: exactly one vacant bed in a multi-bed room.
    # Owner rule: occupied_beds == capacity - 1  (equiv. vacant / available_beds == 1).
    capacity = row.get("capacity")
    occupied = row.get("occupied_beds")
    if (
        pd.notna(capacity)
        and pd.notna(occupied)
        and capacity > 1
        and int(occupied) == int(capacity) - 1
    ):
        bed_type = row.get("bed_type") or "Multi-bed"
        parts.append(
            f"{bed_type} room partially occupied "
            f"({int(occupied)}/{int(capacity)} beds). Remaining bed not filling."
        )

    if status in ("Delayed", "Critical"):
        parts.append(
            "Revenue leakage increasing because vacancy exceeds historical refill time."
        )

    return " ".join(parts)


def detect_blocked_rooms(
    room_inventory: pd.DataFrame,
    occupancy_history: Optional[pd.DataFrame] = None,
    vacancy: Optional[pd.DataFrame] = None,
    *,
    as_of: Optional[pd.Timestamp] = None,
    bed_map: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Return one dashboard-ready row per currently vacant bed.

    Parameters
    ----------
    room_inventory : the Room Master (source of bed rent + occupancy context).
    occupancy_history : optional; used to build the vacancy report when one is
        not supplied directly.
    vacancy : optional pre-computed vacancy report (occupancy_analysis.vacancy_report).
        If omitted, it is derived from ``occupancy_history``.
    as_of : optional reference "today" passed through to the vacancy report.
    """
    # Shared lifecycle views (config + DB) — keep Inactive visible but labeled.
    room_inventory = prepare_inventory(room_inventory)

    # Vacancy END anchor = live/server date (match Vishful), NOT the max date
    # inside Q35. The vacancy START still comes from Q35 exit history unchanged.
    if as_of is None:
        as_of = pd.Timestamp.today().normalize()

    # ---- SINGLE-SOURCE vacant set: from the shared Room Inventory ----------
    # ``available_bed_codes`` already reflects the Bed Map (the one current
    # occupancy source), so the vacant SET here is identical to Inventory
    # Overview / Occupancy Analytics. Current vacancy DAYS come from the Bed Map;
    # EXPECTED fill days stay Q35 (fill-time). The Q35 vacancy report is only a
    # days fallback for beds the Bed Map does not cover (e.g. inactive apts).
    fill = build_fill_time(occupancy_history) if occupancy_history is not None else None
    ft_map: dict = {}
    if fill is not None and not fill.empty:
        for _, fr in fill.iterrows():
            ft_map[(str(fr["apartment_code"]), str(fr["room_code"]))] = (
                fr.get("expected_fill_days"),
                fr.get("recent_average_fill_days"),
                fr.get("historical_average_fill_days"),
            )

    def _key(a, b):
        return (str(a).strip().upper(), str(b).strip().upper())

    bm_days: dict = {}
    if bed_map is not None and isinstance(bed_map, pd.DataFrame) and not bed_map.empty:
        for _, br in bed_map.iterrows():
            bm_days[_key(br.get("apartment_code"), br.get("bed_code"))] = br.get("days_vacant")

    q35_days: dict = {}
    if vacancy is None and occupancy_history is not None:
        vacancy = build_vacancy_report(occupancy_history, as_of=as_of)
    if vacancy is not None and not vacancy.empty:
        for _, vr in vacancy.iterrows():
            q35_days[_key(vr["apartment_code"], vr["bed_code"])] = vr.get("current_vacant_days")

    def _parse_codes(cell):
        if isinstance(cell, list):
            return [str(x).strip() for x in cell if str(x).strip()]
        if cell is None or (isinstance(cell, float) and pd.isna(cell)):
            return []
        return [x.strip() for x in str(cell).split(";") if x.strip()]

    rows = []
    for _, r in room_inventory.iterrows():
        apt = str(r["apartment_code"])
        room = str(r["room_code"])
        # OCCUPANCY-BLOCK rule: a room is "blocked" only when it is PARTIALLY
        # occupied — occupied_beds > 0 AND available_beds > 0 — and its live
        # vacancy is measured against the room's Q35 expected fill time.
        # Entirely-vacant rooms (occupied_beds == 0) are general vacancy, not
        # occupancy blocks: they belong to Inventory/Vacancy reporting only and
        # are excluded from Blocked Rooms (and therefore Revenue Leakage).
        _occ = r.get("occupied_beds")
        _avail = r.get("available_beds")
        _occ = int(_occ) if pd.notna(_occ) else 0
        _avail = int(_avail) if pd.notna(_avail) else 0
        if _occ <= 0 or _avail <= 0:
            continue
        for bed in _parse_codes(r.get("available_bed_codes")):
            k = _key(apt, bed)
            days = bm_days.get(k)
            if days is None or (isinstance(days, float) and pd.isna(days)):
                days = q35_days.get(k)
            exp, rec, hist = ft_map.get((apt, room), (None, None, None))
            rows.append(
                {
                    "apartment_code": apt, "room_code": room, "bed_code": bed,
                    "current_vacant_days": days,
                    "expected_fill_days": exp,
                    "recent_average_fill_days": rec,
                    "historical_average_fill_days": hist,
                    "bed_type": r.get("bed_type"), "capacity": r.get("capacity"),
                    "occupied_beds": r.get("occupied_beds"),
                    "available_beds": r.get("available_beds"),
                    "current_rent": r.get("current_rent"),
                    "demand_score": r.get("demand_score"),
                    "lifecycle_status": r.get("lifecycle_status"),
                    "apartment_status": r.get("apartment_status"),
                    "bed_status": r.get("bed_status"),
                }
            )

    merged = pd.DataFrame(rows)
    if merged.empty:
        empty = pd.DataFrame(columns=list(OUTPUT_COLUMNS))
        empty.attrs["as_of"] = as_of
        return empty
    merged["apartment_status"] = merged["apartment_status"].fillna(LIFECYCLE_STATUS_ACTIVE)

    # Vacancy status: live days (Bed Map) vs Q35 expected fill. Formula unchanged.
    merged["vacancy_status"] = [
        _classify_vacancy(vd, exp, _DEFAULT_CRITICAL_MULTIPLIER)
        for vd, exp in zip(merged["current_vacant_days"], merged["expected_fill_days"])
    ]

    # Bed rent = the room's current monthly rent (never apartment/invoice revenue).
    rent_series = (
        merged["current_rent"] if "current_rent" in merged.columns
        else pd.Series([None] * len(merged), index=merged.index)
    )
    merged["estimated_revenue_loss"] = [
        _estimate_revenue_loss(rent, days)
        for rent, days in zip(rent_series, merged["current_vacant_days"])
    ]
    merged["reason"] = merged.apply(_build_reason, axis=1)

    # Inactive apartments: never Critical / Delayed / Normal.
    inactive = merged["apartment_status"] == LIFECYCLE_STATUS_INACTIVE
    if "lifecycle_status" in merged.columns:
        inactive = inactive | (merged["lifecycle_status"] == LIFECYCLE_STATUS_INACTIVE)
    merged.loc[inactive, "apartment_status"] = LIFECYCLE_STATUS_INACTIVE
    merged.loc[inactive, "lifecycle_status"] = LIFECYCLE_STATUS_INACTIVE
    merged.loc[inactive, "vacancy_status"] = "Inactive"
    merged.loc[inactive, "estimated_revenue_loss"] = 0.0
    merged.loc[inactive, "reason"] = "Apartment is inactive"

    # Sort: Critical first -> highest revenue loss -> highest vacant days.
    merged["_sev"] = merged["vacancy_status"].map(_SEVERITY_ORDER).fillna(len(_SEVERITY_ORDER))
    merged["_loss_sort"] = merged["estimated_revenue_loss"].fillna(0.0)
    merged = merged.sort_values(
        ["_sev", "_loss_sort", "current_vacant_days"],
        ascending=[True, False, False],
    ).reset_index(drop=True)

    for col in OUTPUT_COLUMNS:
        if col not in merged.columns:
            merged[col] = pd.NA

    result = merged[list(OUTPUT_COLUMNS)]
    result.attrs["as_of"] = as_of
    return result


# --------------------------------------------------------------------------- #
# Script entry point: build inputs via the existing modules, then save.
# --------------------------------------------------------------------------- #

def generate_and_save(output_name: str = "blocked_rooms.csv") -> pd.DataFrame:
    from pathlib import Path

    from data_loader import DataLoader
    from preprocessing import build_occupancy_history, build_room_inventory

    loader = DataLoader()
    bookings = loader.bookings()
    tenants = None
    try:
        tenants = loader.tenants()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tenants table unavailable: %s", exc)

    beds_master = None
    try:
        beds_master = loader.beds_master()
    except Exception as exc:  # noqa: BLE001
        logger.warning("beds_master unavailable: %s", exc)

    bed_map = None
    try:
        bed_map = loader.bed_map()
    except Exception as exc:  # noqa: BLE001
        logger.warning("bed-map unavailable: %s", exc)

    room_inventory = build_room_inventory(
        bookings, tenants=tenants, beds_master=beds_master, bed_map=bed_map
    )
    occupancy_history = build_occupancy_history(bookings)
    blocked = detect_blocked_rooms(
        room_inventory, occupancy_history=occupancy_history, bed_map=bed_map
    )

    outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    blocked.to_csv(outputs_dir / output_name, index=False)
    logger.info("Saved blocked-room analytics -> %s", outputs_dir / output_name)
    return blocked


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    df = generate_and_save()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    print(f"\nVacant beds analysed: {len(df)}  |  as_of: {df.attrs.get('as_of')}")
    if not df.empty:
        print("\nStatus breakdown:")
        print(df["vacancy_status"].value_counts().to_string())
        total_loss = df["estimated_revenue_loss"].dropna().sum()
        print(f"\nEstimated total monthly-run-rate revenue leakage: {round(total_loss, 2)}")
        print("\n=== Blocked rooms (Critical first, highest loss) ===")
        show = ["apartment_code", "room_code", "bed_code", "bed_type",
                "current_vacant_days", "expected_fill_days", "vacancy_status",
                "estimated_revenue_loss", "demand_score", "reason"]
        print(df[show].head(20).to_string(index=False))
