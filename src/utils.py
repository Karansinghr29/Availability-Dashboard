"""
utils.py
========

Small, dependency-light helpers shared across the project. Everything here is
pure and generic: NO business rules, NO hardcoded apartment/tenant/city values.

These helpers operate only on pandas DataFrames / plain values so they remain
valid after the CSV -> PostgreSQL/Supabase migration (data_loader.py is the only
source-aware module).

NOTE: room identity (room_code) and room capacity are intentionally NOT derived
here. Deriving a room from bed_code only works for today's CSV convention and
would break once the database provides room_id / room_code / capacity columns.
ALL room-identity logic lives in exactly one place: preprocessing.build_room_inventory().
Every other module consumes the generated Room Inventory DataFrame.
"""

from __future__ import annotations

from typing import Iterable, Optional

import pandas as pd

# Statuses that mean a bed is currently held by a tenant. Kept here (not
# hardcoded per-module) and matched case-insensitively so new data adapts.
OCCUPIED_STATUSES = {"staying", "on-notice", "onnotice", "reserved"}

# --------------------------------------------------------------------------- #
# Analytics time windows (business configuration — the single place to change).
#
# Two baselines are used across all occupancy analytics:
#   * HISTORICAL baseline (2023+)  -> long-term behaviour, kept for REPORTING.
#   * RECENT operational baseline (2025+) -> used for recommendations and for
#     blocked-room / anomaly detection, because operations stabilised in 2025.
# Data older than the historical start is never discarded: it still informs
# room capacity, room age and first-occupancy references — it is just not the
# baseline for occupancy/revenue/fill-time metrics.
# --------------------------------------------------------------------------- #
HISTORICAL_BASELINE_START = pd.Timestamp("2023-01-01")
RECENT_BASELINE_START = pd.Timestamp("2025-01-01")

# Relative weights (NOT thresholds) for the composite room demand_score. They
# only decide how the four already-normalised signals are blended; the signals
# themselves are min-max normalised dynamically across the current room set, so
# the score adapts automatically to whatever data is loaded. Must sum to 1.
DEMAND_SCORE_WEIGHTS = {
    "recent_occupancy": 0.25,
    "recent_fill_rate": 0.25,
    "low_vacancy": 0.25,       # lower average vacancy => higher demand
    "revenue_performance": 0.25,
}


def normalize_text(value) -> Optional[str]:
    """Lower-case, trim and collapse a scalar for robust matching."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if pd.isna(value):
        return None
    return " ".join(str(value).strip().lower().split())


def normalize_series(series: pd.Series) -> pd.Series:
    """Vectorised :func:`normalize_text` for a whole column."""
    return series.map(normalize_text)


# --------------------------------------------------------------------------- #
# Inventory lifecycle Status (shared by EVERY module for consistency).
#
# Status values: Active | Inactive | Unknown
# Derived ONLY from explicit lifecycle/status columns — never from occupancy.
# When NONE of those columns exist (today's CSVs) -> Status = Unknown for all
# rows. Optional overrides in ``config/inactive_apartments.json`` (loaded by
# ``config.py``) may still mark listed apartments Inactive until the DB
# exports a real apartment_status field — then empty that JSON and remove it.
#
# ``beds_master.bed_status`` uses Live / Not-Active. Those tokens drive
# Inactive detection once the master is joined onto Room Inventory.
#
# Customer Recommendation / Room Search / Occupancy Analytics default to
# excluding Inactive. Inactive appears only when the user opts in
# ("Show Inactive"). Blocked-room / revenue modules may still label Inactive.
# --------------------------------------------------------------------------- #
# Column -> token(s) that mean the unit is Active / Live for leasing.
# ``bed_lifecycle_status`` (occupied/notice/vacant/booked) is intentionally
# NOT listed — it is occupancy state, not apartment Inactive.
ACTIVE_STATUS_TOKENS = {
    "room_status": frozenset({"active"}),
    "room_lifecycle_status": frozenset({"active"}),
    "apartment_status": frozenset({"active"}),
    "apartment_lifecycle_status": frozenset({"active"}),
    "bed_status": frozenset({"live", "available", "active"}),
}

# Back-compat single-token map (first/canonical active token per column).
ACTIVE_STATUS_COLUMNS = {col: next(iter(tokens)) for col, tokens in ACTIVE_STATUS_TOKENS.items()}

LIFECYCLE_STATUS_ACTIVE = "Active"
LIFECYCLE_STATUS_INACTIVE = "Inactive"
LIFECYCLE_STATUS_UNKNOWN = "Unknown"


def present_lifecycle_columns(df_or_row) -> list:
    """Lifecycle/status columns that actually exist on a DataFrame or row."""
    if df_or_row is None:
        return []
    if isinstance(df_or_row, pd.DataFrame):
        return [c for c in ACTIVE_STATUS_TOKENS if c in df_or_row.columns]
    if isinstance(df_or_row, pd.Series):
        return [c for c in ACTIVE_STATUS_TOKENS if c in df_or_row.index]
    if isinstance(df_or_row, dict):
        return [c for c in ACTIVE_STATUS_TOKENS if c in df_or_row]
    return []


def derive_lifecycle_status(df: pd.DataFrame) -> pd.Series:
    """Derive Active / Inactive / Unknown for every row.

    - No lifecycle columns present  -> Unknown
    - Explicit value in the column's active-token set -> Active
    - Explicit non-active value     -> Inactive
    - All lifecycle values null     -> Unknown
    Never uses occupancy.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=object)
    present = present_lifecycle_columns(df)
    if not present:
        # Prefer an existing annotation (e.g. reloaded CSV); else Unknown.
        if "lifecycle_status" in df.columns:
            return (
                df["lifecycle_status"]
                .fillna(LIFECYCLE_STATUS_UNKNOWN)
                .astype(object)
            )
        return pd.Series(LIFECYCLE_STATUS_UNKNOWN, index=df.index, dtype=object)

    saw = pd.Series(False, index=df.index)
    inactive = pd.Series(False, index=df.index)
    for col in present:
        norm = df[col].map(normalize_text)
        has = norm.notna()
        saw |= has
        active_tokens = ACTIVE_STATUS_TOKENS[col]
        inactive |= has & ~norm.isin(active_tokens)

    out = pd.Series(LIFECYCLE_STATUS_UNKNOWN, index=df.index, dtype=object)
    out = out.mask(saw & ~inactive, LIFECYCLE_STATUS_ACTIVE)
    out = out.mask(saw & inactive, LIFECYCLE_STATUS_INACTIVE)
    return out


def apply_temporary_inactive_apartments(df: pd.DataFrame) -> pd.DataFrame:
    """Apply config-driven temporary Inactive apartment overrides.

    Codes come from ``config/inactive_apartments.json`` via ``config.py``.
    Empty config → no-op (ready for DB ``apartment_status``).
    """
    from config import get_temporarily_inactive_apartments

    inactive_codes = get_temporarily_inactive_apartments()
    if not inactive_codes:
        return df
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    if "apartment_code" not in df.columns:
        return df
    out = df.copy()
    if "lifecycle_status" not in out.columns:
        out["lifecycle_status"] = LIFECYCLE_STATUS_UNKNOWN
    codes = out["apartment_code"].astype(str).str.strip().str.upper()
    mask = codes.isin(inactive_codes)
    if mask.any():
        out.loc[mask, "lifecycle_status"] = LIFECYCLE_STATUS_INACTIVE
    return out


def annotate_lifecycle_status(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with ``lifecycle_status`` (Active/Inactive/Unknown).

    Column-derived status first; then optional config overrides for apartments
    that are known Inactive until the DB exports real status fields.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return df
    out = df.copy()
    out["lifecycle_status"] = derive_lifecycle_status(out)
    return apply_temporary_inactive_apartments(out)


def is_active_room(row) -> bool:
    """True when the row is recommendable / countable as active inventory.

    Unknown (no lifecycle columns) is treated as eligible. Only explicit
    Inactive is excluded. Never infers from occupancy.
    """
    if isinstance(row, pd.Series) and "lifecycle_status" in row.index:
        return normalize_text(row["lifecycle_status"]) != "inactive"
    if isinstance(row, dict) and "lifecycle_status" in row:
        return normalize_text(row.get("lifecycle_status")) != "inactive"
    status = derive_lifecycle_status(pd.DataFrame([row]))
    return status.empty or status.iloc[0] != LIFECYCLE_STATUS_INACTIVE


def exclude_inactive_rooms(df: pd.DataFrame) -> pd.DataFrame:
    """Keep Active + Unknown rows; drop only explicit Inactive.

    Used by recommendations, room search, and occupancy analytics by default.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    work = df if "lifecycle_status" in df.columns else annotate_lifecycle_status(df)
    status = work["lifecycle_status"]
    return work.loc[status != LIFECYCLE_STATUS_INACTIVE].copy()


# Back-compat alias: historically dropped inactive rows. Now keeps Unknown and
# only drops explicit Inactive (same as exclude_inactive_rooms).
def filter_active_rooms(df: pd.DataFrame) -> pd.DataFrame:
    return exclude_inactive_rooms(df)


def summarize_inventory_lifecycle(df: pd.DataFrame) -> dict:
    """Active vacant inventory summary (Step 1). Does not change calculations."""
    empty = {
        "total_apartments": 0,
        "active_apartments": 0,
        "inactive_apartments": 0,
        "total_rooms": 0,
        "active_rooms": 0,
        "inactive_rooms": 0,
        "total_beds": 0,
        "occupied_beds": 0,
        "vacant_beds": 0,
        "vacant_rooms_active": 0,
    }
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return empty

    # Prefer shared apartment_status when present (from inventory_views).
    # Never re-annotate frames that already carry view columns — those names
    # collide with source lifecycle fields and would mark everything Inactive.
    view_cols = ("apartment_status", "room_status", "bed_status", "recommendable")
    if "apartment_status" in df.columns:
        work = df
        inactive_mask = work["apartment_status"].astype(str) == LIFECYCLE_STATUS_INACTIVE
    else:
        base = df.drop(columns=[c for c in view_cols if c in df.columns], errors="ignore")
        work = annotate_lifecycle_status(base)
        inactive_mask = work["lifecycle_status"] == LIFECYCLE_STATUS_INACTIVE

    active = work.loc[~inactive_mask]
    inactive = work.loc[inactive_mask]

    all_apts = set(work["apartment_code"].astype(str).unique()) if "apartment_code" in work else set()
    inactive_apts = (
        set(inactive["apartment_code"].astype(str).unique()) if not inactive.empty else set()
    )
    active_apts = (
        set(active["apartment_code"].astype(str).unique()) if not active.empty else set()
    )
    inactive_only_apts = inactive_apts - active_apts

    total_beds = int(work["capacity"].fillna(0).sum()) if "capacity" in work.columns else 0
    occupied = int(work["occupied_beds"].fillna(0).sum()) if "occupied_beds" in work.columns else 0
    vacant = int(work["available_beds"].fillna(0).sum()) if "available_beds" in work.columns else max(
        total_beds - occupied, 0
    )
    vacant_rooms_active = 0
    if not active.empty and "available_beds" in active.columns:
        vacant_rooms_active = int((active["available_beds"].fillna(0) >= 1).sum())

    return {
        "total_apartments": len(all_apts),
        "active_apartments": len(active_apts),
        "inactive_apartments": len(inactive_only_apts),
        "total_rooms": len(work),
        "active_rooms": len(active),
        "inactive_rooms": int(inactive_mask.sum()),
        "total_beds": total_beds,
        "occupied_beds": occupied,
        "vacant_beds": vacant,
        "vacant_rooms_active": vacant_rooms_active,
    }


def print_inventory_lifecycle_summary(df: pd.DataFrame) -> dict:
    """Print Step 1 active/inactive inventory summary; return the same dict."""
    summary = summarize_inventory_lifecycle(df)
    print("\n" + "=" * 60)
    print("ACTIVE VACANT INVENTORY SUMMARY")
    print("=" * 60)
    print(f"Total Apartments          : {summary['total_apartments']}")
    print(f"Active Apartments         : {summary['active_apartments']}")
    print(f"Inactive Apartments       : {summary['inactive_apartments']}")
    print(f"Total Rooms               : {summary['total_rooms']}")
    print(f"Active Rooms              : {summary['active_rooms']}")
    print(f"Inactive Rooms            : {summary['inactive_rooms']}")
    print(f"Total Beds                : {summary['total_beds']}")
    print(f"Occupied Beds             : {summary['occupied_beds']}")
    print(f"Vacant Beds               : {summary['vacant_beds']}")
    print(f"Vacant Rooms (Active only): {summary['vacant_rooms_active']}")
    print("=" * 60)
    return summary


def print_candidate_pool_after_inactive_filter(
    full_inventory: pd.DataFrame, active_inventory: pd.DataFrame
) -> None:
    """Print Step 3 candidate pool (active rooms only) before ranking."""
    total = 0 if full_inventory is None else len(full_inventory)
    summary = summarize_inventory_lifecycle(full_inventory)
    remaining = 0 if active_inventory is None else len(active_inventory)
    removed = summary["inactive_rooms"]

    print("\n" + "=" * 60)
    print("CANDIDATE POOL AFTER INACTIVE FILTERING")
    print("=" * 60)
    print(f"Total rooms              : {total}")
    print(f"Inactive apartments      : {summary['inactive_apartments']}")
    print(f"Inactive rooms removed   : {removed}")
    print(f"Remaining active rooms   : {remaining}")
    print(f"Candidate rooms after inactive filtering = {remaining}")

    if active_inventory is not None and not active_inventory.empty and "available_beds" in active_inventory.columns:
        vacant = active_inventory[active_inventory["available_beds"].fillna(0) >= 1].copy()
        print("\nVacant active rooms:")
        if vacant.empty:
            print("(none)")
        else:
            print(f"{'Room':<10} {'Apartment':<12} {'Occupancy'}")
            sort_cols = [c for c in ("current_occupancy_pct", "room_code") if c in vacant.columns]
            vacant = vacant.sort_values(sort_cols) if sort_cols else vacant
            for _, row in vacant.iterrows():
                occ = row.get("current_occupancy_pct", 0)
                try:
                    occ_s = f"{float(occ):g}%"
                except (TypeError, ValueError):
                    occ_s = str(occ)
                print(f"{str(row.get('room_code', '')):<10} {str(row.get('apartment_code', '')):<12} {occ_s}")
    print("=" * 60)


def build_city_state_map(tenants: pd.DataFrame) -> dict:
    """Learn a city -> state lookup from the tenant table (data-driven).

    This is intentionally derived from live data so it adapts automatically.
    It will NOT cover cities that have never appeared among tenants; callers
    should layer a static fallback map on top (see the gap report in README).
    """
    if not {"city", "state"}.issubset(tenants.columns):
        return {}
    pairs = tenants[["city", "state"]].dropna()
    pairs = pairs.assign(_city=normalize_series(pairs["city"]))
    # Most frequent state per city wins.
    out = (
        pairs.groupby("_city")["state"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )
    return out


def require_columns(df: pd.DataFrame, columns: Iterable[str], table: str = "") -> None:
    """Raise a clear error if expected columns are missing (fail fast)."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"Table '{table}' is missing required columns: {missing}")
