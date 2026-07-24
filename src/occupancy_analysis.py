"""
occupancy_analysis.py  —  PHASE 2 STEP 3: FILL-TIME ENGINE
==========================================================

Turns the canonical Occupancy History (preprocessing.build_occupancy_history)
into per-room fill-time behaviour and a per-vacant-bed vacancy report.

It treats ``room_code`` as opaque and never derives rooms from bed_code; room
identity comes exclusively from the Occupancy History / Room Inventory.

Two windows (see utils config) are used everywhere:
  * RECENT baseline (2025+)      -> primary signal for expected fill time.
  * HISTORICAL baseline (2023+)  -> reporting + fallback when recent data is thin.
Pre-2023 data stays as reference only (already excluded upstream by window tags).

Public functions
----------------
fill_time(occupancy_history, ...)          -> per-room fill-time table
current_vacant_days(occupancy_history, ...) -> per currently-vacant bed
vacancy_report(occupancy_history, ...)      -> the Step-3 deliverable (per vacant bed)

apartment_occupancy(...) is a later-step stub (internal ranking).

This module is the FOUNDATION for Blocked-Room Detection, Revenue Leakage,
Pricing Opportunity and the Management Dashboard — none of which are built here.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# A refill's vacancy event is bucketed by its exit date in build_occupancy_history:
#   exit >= 2025 -> "recent";  exit >= 2023 -> "historical";  older -> "pre".
# The historical baseline (2023+) therefore includes the recent window too.
_HISTORICAL_WINDOWS = {"recent", "historical"}
_RECENT_WINDOWS = {"recent"}

# Minimum number of recent refill events for the recent average to be trusted
# as the expected fill time (else we fall back to the historical average).
# This is an event COUNT, not a day threshold — no fill-day value is hardcoded.
_DEFAULT_RECENT_MIN_EVENTS = 1

# Multiplier defining the "Critical" band. Comes straight from the business
# rule (>= 2x expected). Kept as a named, overridable parameter.
_DEFAULT_CRITICAL_MULTIPLIER = 2.0


def _resolve_as_of(occupancy_history: pd.DataFrame, as_of) -> pd.Timestamp:
    """Reference 'today': explicit arg, else the frame's attrs, else max date."""
    if as_of is not None:
        return pd.Timestamp(as_of)
    if occupancy_history.attrs.get("as_of") is not None:
        return pd.Timestamp(occupancy_history.attrs["as_of"])
    candidates = [
        occupancy_history[col]
        for col in ("actual_exit_date", "onboarding_date", "next_onboarding_date")
        if col in occupancy_history.columns
    ]
    return pd.Timestamp(pd.concat(candidates).max()) if candidates else pd.Timestamp.today()


def fill_time(
    occupancy_history: pd.DataFrame,
    recent_min_events: int = _DEFAULT_RECENT_MIN_EVENTS,
) -> pd.DataFrame:
    """Per-room average fill time over both windows, plus the expected value.

    Fill time = the vacancy gap (previous exit -> next onboarding) of a REFILLED
    vacancy event. Each event is attributed to a window by its exit date
    (``exit_window`` from the Occupancy History).

    Returns per room:
        apartment_code, room_code,
        recent_average_fill_days, historical_average_fill_days,
        n_recent_refills, n_historical_refills, expected_fill_days,
        expected_source ("recent" | "historical" | "none").
    """
    oh = occupancy_history
    refilled = oh[oh["refilled"] == True].copy()  # noqa: E712 (works on bool/obj)

    hist = refilled[refilled["exit_window"].isin(_HISTORICAL_WINDOWS)]
    recent = refilled[refilled["exit_window"].isin(_RECENT_WINDOWS)]

    keys = ["apartment_code", "room_code"]
    # All rooms seen anywhere (so fully-occupied rooms still appear).
    rooms = oh[keys].drop_duplicates()

    def _agg(frame: pd.DataFrame, avg_name: str, cnt_name: str) -> pd.DataFrame:
        if frame.empty:
            return pd.DataFrame(columns=keys + [avg_name, cnt_name])
        g = frame.groupby(keys)["vacancy_days"]
        return (
            g.agg(**{avg_name: "mean", cnt_name: "size"})
            .reset_index()
        )

    hist_agg = _agg(hist, "historical_average_fill_days", "n_historical_refills")
    recent_agg = _agg(recent, "recent_average_fill_days", "n_recent_refills")

    out = rooms.merge(hist_agg, on=keys, how="left").merge(recent_agg, on=keys, how="left")
    out["n_historical_refills"] = out["n_historical_refills"].fillna(0).astype(int)
    out["n_recent_refills"] = out["n_recent_refills"].fillna(0).astype(int)

    use_recent = (out["n_recent_refills"] >= recent_min_events) & out[
        "recent_average_fill_days"
    ].notna()
    out["expected_fill_days"] = out["historical_average_fill_days"].where(~use_recent)
    out.loc[use_recent, "expected_fill_days"] = out.loc[
        use_recent, "recent_average_fill_days"
    ]
    out["expected_source"] = "none"
    out.loc[out["historical_average_fill_days"].notna(), "expected_source"] = "historical"
    out.loc[use_recent, "expected_source"] = "recent"

    for col in ("recent_average_fill_days", "historical_average_fill_days",
                "expected_fill_days"):
        out[col] = out[col].round(1)
    return out.sort_values(keys).reset_index(drop=True)


def current_vacant_days(
    occupancy_history: pd.DataFrame, as_of: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """One row per CURRENTLY VACANT bed with days vacant since its last exit.

    A bed is occupied if it has any active span; otherwise it is vacant and its
    vacancy started at its most recent ``actual_exit_date``.
    """
    as_of = _resolve_as_of(occupancy_history, as_of)
    keys = ["apartment_code", "room_code", "bed_code"]

    grouped = occupancy_history.groupby(keys)
    rows = []
    for (apartment_code, room_code, bed_code), g in grouped:
        if g["is_active"].any():
            continue  # currently occupied
        last_exit = g["actual_exit_date"].max()
        if pd.isna(last_exit):
            continue
        rows.append(
            {
                "apartment_code": apartment_code,
                "room_code": room_code,
                "bed_code": bed_code,
                "last_exit_date": last_exit,
                "current_vacant_days": int((as_of - last_exit).days),
            }
        )
    result = pd.DataFrame(rows, columns=[*keys, "last_exit_date", "current_vacant_days"])
    result.attrs["as_of"] = as_of
    return result


def _classify_vacancy(vacant_days: float, expected: float, critical_multiplier: float) -> str:
    """Dynamic per-bed classification against the room's OWN expected fill time."""
    if pd.isna(expected):
        return "Unknown"  # room has no refill history to compare against
    if expected > 0 and vacant_days >= critical_multiplier * expected:
        return "Critical"
    if vacant_days > expected:
        return "Delayed"
    return "Normal"


def vacancy_report(
    occupancy_history: pd.DataFrame,
    as_of: Optional[pd.Timestamp] = None,
    recent_min_events: int = _DEFAULT_RECENT_MIN_EVENTS,
    critical_multiplier: float = _DEFAULT_CRITICAL_MULTIPLIER,
) -> pd.DataFrame:
    """PHASE 2 STEP 3 deliverable: per vacant bed, expected fill + status.

    Columns:
        apartment_code, room_code, bed_code, current_vacant_days,
        expected_fill_days, vacancy_status,
        recent_average_fill_days, historical_average_fill_days.
    """
    room_ft = fill_time(occupancy_history, recent_min_events=recent_min_events)
    vacant = current_vacant_days(occupancy_history, as_of=as_of)

    report = vacant.merge(room_ft, on=["apartment_code", "room_code"], how="left")
    report["vacancy_status"] = [
        _classify_vacancy(vd, exp, critical_multiplier)
        for vd, exp in zip(report["current_vacant_days"], report["expected_fill_days"])
    ]

    columns = [
        "apartment_code", "room_code", "bed_code", "current_vacant_days",
        "expected_fill_days", "vacancy_status",
        "recent_average_fill_days", "historical_average_fill_days",
    ]
    report = report[columns].sort_values(
        ["vacancy_status", "current_vacant_days"], ascending=[True, False]
    ).reset_index(drop=True)
    report.attrs["as_of"] = vacant.attrs.get("as_of")
    return report


def apartment_occupancy(room_inventory: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Phase 2 — later step (internal apartment ranking).")


# --------------------------------------------------------------------------- #
# Script entry point: load via data_loader -> occupancy history -> save report.
# --------------------------------------------------------------------------- #


def generate_and_save(output_name: str = "vacancy_report.csv") -> pd.DataFrame:
    from pathlib import Path

    from data_loader import DataLoader
    from preprocessing import build_occupancy_history

    loader = DataLoader()
    history = build_occupancy_history(loader.bookings())
    report = vacancy_report(history)

    outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(outputs_dir / output_name, index=False)
    fill_time(history).to_csv(outputs_dir / "room_fill_time.csv", index=False)
    logger.info("Saved vacancy report -> %s", outputs_dir / output_name)
    logger.info("Saved room fill-time -> %s", outputs_dir / "room_fill_time.csv")
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rpt = generate_and_save()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(f"\nVacant beds: {len(rpt)}  |  as_of: {rpt.attrs.get('as_of')}")
    print("\nStatus breakdown:")
    print(rpt["vacancy_status"].value_counts().to_string())
    print("\n=== Vacancy report (most overdue first) ===")
    print(rpt.head(20).to_string(index=False))
