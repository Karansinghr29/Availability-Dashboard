"""
revenue_analytics.py  —  PHASE 6: PORTFOLIO REVENUE-LEAKAGE ANALYTICS
=====================================================================

A lightweight AGGREGATION-ONLY module. It does NOT compute or recalculate any
revenue leakage — ``blocked_room_detector`` is the single source of truth for
per-bed ``estimated_revenue_loss`` and ``vacancy_status``. This module simply
rolls those existing values up into portfolio-level KPIs and per-apartment
breakdowns for the (future) Streamlit dashboard.

Inputs (either form):
  * a blocked-rooms DataFrame (blocked_room_detector.detect_blocked_rooms), or
  * outputs/blocked_rooms.csv (auto-regenerated via the detector if missing).

Output:
  * outputs/revenue_summary.csv                — portfolio KPIs (metric, value)
  * outputs/revenue_leakage_by_apartment.csv   — per-apartment breakdown

Fully dynamic (no hardcoded rooms/rents/apartments). When PostgreSQL replaces
the CSVs, only the upstream modules change; this aggregator is unaffected.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import pandas as pd

from utils import (
    LIFECYCLE_STATUS_ACTIVE,
    LIFECYCLE_STATUS_INACTIVE,
    LIFECYCLE_STATUS_UNKNOWN,
    annotate_lifecycle_status,
)

logger = logging.getLogger(__name__)

_STATUSES = ("Critical", "Delayed", "Normal", "Unknown", "Inactive")

_BY_APARTMENT_COLUMNS = (
    "apartment_code", "lifecycle_status", "estimated_revenue_loss",
    "critical_rooms", "delayed_rooms", "normal_rooms", "unknown_rooms",
    "inactive_rooms", "vacant_beds", "avg_vacancy_days",
)


def load_blocked_rooms(source: Union[pd.DataFrame, str, None] = None) -> pd.DataFrame:
    """Return the blocked-rooms table from a DataFrame, a path, or the default CSV.

    Falls back to regenerating via ``blocked_room_detector`` (the single source
    of truth) when the CSV does not yet exist — no leakage logic is duplicated.
    Inactive rows stay visible; leakage for them is forced to 0.
    """
    if isinstance(source, pd.DataFrame):
        blocked = source
    else:
        from pathlib import Path

        path = (
            Path(source) if source
            else Path(__file__).resolve().parents[1] / "outputs" / "blocked_rooms.csv"
        )
        if path.exists():
            blocked = pd.read_csv(path)
        else:
            logger.info("blocked_rooms.csv not found; regenerating via blocked_room_detector.")
            from blocked_room_detector import generate_and_save
            blocked = generate_and_save()

    blocked = annotate_lifecycle_status(blocked)
    inactive = blocked["lifecycle_status"] == LIFECYCLE_STATUS_INACTIVE
    if inactive.any():
        blocked = blocked.copy()
        blocked.loc[inactive, "estimated_revenue_loss"] = 0.0
        if "vacancy_status" in blocked.columns:
            blocked.loc[inactive, "vacancy_status"] = "Inactive"
    return blocked


def _rooms_with_status(frame: pd.DataFrame, status: str) -> int:
    """Distinct rooms carrying a given vacancy status."""
    return int(frame.loc[frame["vacancy_status"] == status, "room_code"].nunique())


def _apartment_lifecycle_status(g: pd.DataFrame) -> str:
    """Apartment Status from its rooms: Inactive if all Inactive, else Active/Unknown."""
    statuses = set(g.get("lifecycle_status", pd.Series(dtype=object)).dropna().tolist())
    if not statuses:
        return LIFECYCLE_STATUS_UNKNOWN
    if statuses == {LIFECYCLE_STATUS_INACTIVE}:
        return LIFECYCLE_STATUS_INACTIVE
    if LIFECYCLE_STATUS_ACTIVE in statuses:
        return LIFECYCLE_STATUS_ACTIVE
    if LIFECYCLE_STATUS_INACTIVE in statuses and LIFECYCLE_STATUS_UNKNOWN in statuses:
        return LIFECYCLE_STATUS_INACTIVE
    return next(iter(statuses))


def leakage_by_apartment(blocked: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the existing per-bed leakage up to one row per apartment."""
    if blocked.empty:
        return pd.DataFrame(columns=list(_BY_APARTMENT_COLUMNS))

    rows = []
    for apartment_code, g in blocked.groupby("apartment_code"):
        apt_status = _apartment_lifecycle_status(g)
        # Inactive apartments contribute 0 leakage to the portfolio.
        loss = (
            0.0 if apt_status == LIFECYCLE_STATUS_INACTIVE
            else round(g["estimated_revenue_loss"].dropna().sum(), 2)
        )
        rows.append(
            {
                "apartment_code": apartment_code,
                "lifecycle_status": apt_status,
                "estimated_revenue_loss": loss,
                "critical_rooms": _rooms_with_status(g, "Critical"),
                "delayed_rooms": _rooms_with_status(g, "Delayed"),
                "normal_rooms": _rooms_with_status(g, "Normal"),
                "unknown_rooms": _rooms_with_status(g, "Unknown"),
                "inactive_rooms": _rooms_with_status(g, "Inactive"),
                "vacant_beds": int(len(g)),
                "avg_vacancy_days": round(float(g["current_vacant_days"].mean()), 1),
            }
        )
    return (
        pd.DataFrame(rows, columns=list(_BY_APARTMENT_COLUMNS))
        .sort_values(["estimated_revenue_loss", "critical_rooms"], ascending=[False, False])
        .reset_index(drop=True)
    )


def portfolio_kpis(blocked: pd.DataFrame, by_apartment: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Portfolio-level KPIs as a tidy (metric, value) table."""
    if by_apartment is None:
        by_apartment = leakage_by_apartment(blocked)

    # Highest-loss calcs ignore explicit Inactive inventory.
    countable = blocked
    if "lifecycle_status" in blocked.columns and not blocked.empty:
        countable = blocked[blocked["lifecycle_status"] != LIFECYCLE_STATUS_INACTIVE]

    kpis: dict = {}
    kpis["total_estimated_revenue_leakage"] = round(
        float(countable["estimated_revenue_loss"].dropna().sum()), 2
    ) if not countable.empty else 0.0
    kpis["total_vacant_beds"] = int(len(blocked))

    for status in _STATUSES:
        kpis[f"total_{status.lower()}_rooms"] = (
            _rooms_with_status(blocked, status) if not blocked.empty else 0
        )

    kpis["average_vacancy_days"] = (
        round(float(blocked["current_vacant_days"].mean()), 1) if not blocked.empty else 0.0
    )

    loss = (
        countable["estimated_revenue_loss"]
        if not countable.empty
        else pd.Series(dtype="float64")
    )
    if loss.notna().any() and float(loss.max()) > 0:
        top = countable.loc[loss.idxmax()]
        kpis["highest_loss_room"] = (
            f"{top['apartment_code']} / {top['room_code']} / {top['bed_code']}"
        )
        kpis["highest_loss_room_value"] = round(float(top["estimated_revenue_loss"]), 2)
    else:
        kpis["highest_loss_room"] = None
        kpis["highest_loss_room_value"] = 0.0

    active_apts = by_apartment
    if "lifecycle_status" in by_apartment.columns and not by_apartment.empty:
        active_apts = by_apartment[
            by_apartment["lifecycle_status"] != LIFECYCLE_STATUS_INACTIVE
        ]
    if not active_apts.empty and float(active_apts["estimated_revenue_loss"].max()) > 0:
        top_apt = active_apts.iloc[0]
        kpis["highest_loss_apartment"] = top_apt["apartment_code"]
        kpis["highest_loss_apartment_value"] = round(
            float(top_apt["estimated_revenue_loss"]), 2
        )
    else:
        kpis["highest_loss_apartment"] = None
        kpis["highest_loss_apartment_value"] = 0.0

    return pd.DataFrame({"metric": list(kpis.keys()), "value": list(kpis.values())})


def summarize(source: Union[pd.DataFrame, str, None] = None):
    """Return (portfolio_kpis, leakage_by_apartment) from a blocked-rooms source."""
    blocked = load_blocked_rooms(source)
    by_apartment = leakage_by_apartment(blocked)
    kpis = portfolio_kpis(blocked, by_apartment)
    return kpis, by_apartment


# --------------------------------------------------------------------------- #
# Script entry point
# --------------------------------------------------------------------------- #

def generate_and_save(
    source: Union[pd.DataFrame, str, None] = None,
    summary_name: str = "revenue_summary.csv",
    by_apartment_name: str = "revenue_leakage_by_apartment.csv",
):
    from pathlib import Path

    kpis, by_apartment = summarize(source)

    outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    kpis.to_csv(outputs_dir / summary_name, index=False)
    by_apartment.to_csv(outputs_dir / by_apartment_name, index=False)
    logger.info("Saved portfolio KPIs -> %s", outputs_dir / summary_name)
    logger.info("Saved apartment breakdown -> %s", outputs_dir / by_apartment_name)
    return kpis, by_apartment


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    kpis, by_apt = generate_and_save()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    print("\n=== Portfolio Revenue-Leakage KPIs ===")
    print(kpis.to_string(index=False))
    print("\n=== Revenue Leakage by Apartment (highest first) ===")
    print(by_apt.head(20).to_string(index=False))
