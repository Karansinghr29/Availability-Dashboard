"""
preprocessing.py
================

Turns the raw logical tables from :mod:`data_loader` into the clean, derived
frames the engines need. Works ONLY on DataFrames (source-agnostic).

SINGLE SOURCE OF TRUTH FOR ROOM IDENTITY
----------------------------------------
`build_room_inventory()` is the ONLY function in the entire project allowed to
decide what a "room" is. No other module may derive a room from `bed_code` (or
anything else). Every downstream module — recommendation engine, occupancy,
revenue, blocked-room detection, dashboard — consumes the Room Inventory
DataFrame it returns and treats `room_code` as opaque.

Why: `room = apartment_code + first char of bed_code` is only valid for today's
CSV convention. Tomorrow the database may expose `room_id` / `room_code` /
`capacity` columns, or a different bed-naming scheme. Because room identity is
centralised here, that migration touches THIS function only (plus data_loader).

Room Master schema (one row per room, returned by build_room_inventory)
-----------------------------------------------------------------------
    apartment_code            str
    room_code                 str    (opaque room id)
    bed_type                  str
    capacity                  int    (distinct beds ever observed / source col)
    occupied_beds             int    (active bookings only)
    available_beds            int
    available_bed_codes       list
    current_occupancy_pct     float  (snapshot: occupied / capacity)
    current_revenue           float  (sum of current occupied-bed rents)
    current_rent              float  (latest positive rent seen in the room)
    recent_occupancy_pct      float  (RECENT window, default 2025+)
    recent_revenue            float  (RECENT window accrued)
    recent_vacancy_events     int    (real tenancies ended in RECENT window)
    recent_refills            int    (of those, beds a new tenant took)
    recent_fill_rate          float  (refills / vacancy_events, or NaN)
    average_vacancy_days_recent float (mean exit->refill gap, or NaN)
    historical_occupancy_pct  float  (business-priority occupancy; same window as Demand: 2025+)
    historical_revenue        float  (accrued rent over the same business-priority window)
    occupant_states           list   (home states of current occupants, unique)
    occupant_genders          list   (per-occupant gender, when available)
    occupant_occupations      list   (per-occupant 'student'/'working', when available)
    occupant_ages             list   (per-occupant age in years, when available)
    gender_allowed            str    (from beds_master only; else Unknown)
    bed_status                str    (from beds_master: Live / Not-Active)
    bed_lifecycle_status      str    (from beds_master: occupied/notice/vacant/booked)
    toilet_type               str    (from beds_master: Attached / Common)
    current_rate              float  (from beds_master when present)
    bed_codes                 list   (all beds in the room)
    lifecycle_status          str    (Active / Inactive / Unknown)
    demand_score              float  (0-100 composite, normalised across rooms)

Occupancy analytics use two windows (configured in utils):
  * historical baseline 2023+  -> long-term reporting.
  * recent operational 2025+   -> recommendations & anomaly detection.
Pre-window data is never discarded; it still sets capacity and room age.

Phase-2 status: STEP 1 (build_room_inventory) and STEP 2 (build_occupancy_history)
are IMPLEMENTED. The remaining builders are still interface stubs.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

from utils import (
    ACTIVE_STATUS_COLUMNS,
    ACTIVE_STATUS_TOKENS,
    DEMAND_SCORE_WEIGHTS,
    HISTORICAL_BASELINE_START,
    OCCUPIED_STATUSES,
    RECENT_BASELINE_START,
    annotate_lifecycle_status,
    normalize_series,
    normalize_text,
    require_columns,
)

logger = logging.getLogger(__name__)

# Approx. days per month, used to convert a monthly rent into a daily accrual
# when estimating historical (realised) revenue over an occupancy span.
_DAYS_PER_MONTH = 30.44

# Candidate source columns that, when present, are authoritative and override
# any derivation (future PostgreSQL/Supabase schema).
_SOURCE_ROOM_ID_COLUMNS = ("room_id", "room_code", "room_number")
_SOURCE_CAPACITY_COLUMNS = ("capacity", "room_capacity", "total_beds")

# bed_type -> expected capacity. Used ONLY to raise data-quality warnings when
# the label disagrees with the dynamically-derived capacity. It is NEVER the
# source of the capacity value itself (capacity is always derived from data).
_BED_TYPE_HINT = {"single": 1, "double": 2, "triple": 3, "quad": 4, "quadruple": 4}


# --------------------------------------------------------------------------- #
# Room-identity derivation — PRIVATE, used ONLY by build_room_inventory().
# Do not import these anywhere else; consume the Room Inventory instead.
# --------------------------------------------------------------------------- #


def _resolve_room_code(frame: pd.DataFrame) -> pd.Series:
    """Return an opaque ``room_code`` per row (the ONLY place this is decided).

    Strategy (future-proof):
      1. If the source already provides a room identifier column
         (``_SOURCE_ROOM_ID_COLUMNS``) with data, use it verbatim.
      2. Otherwise fall back to today's CSV convention:
         ``apartment_code`` + first alphabetic run of ``bed_code``
         (e.g. apartment ``B24`` + bed ``B3`` -> room ``B24-B``). When a
         bed_code carries no letter, the room degrades to the apartment itself.
    """
    for col in _SOURCE_ROOM_ID_COLUMNS:
        if col in frame.columns and frame[col].notna().any():
            logger.info("Using source-provided room identifier column '%s'.", col)
            return frame[col].astype("string")

    apartment = frame["apartment_code"].astype("string").str.strip()
    letter = (
        frame["bed_code"]
        .astype("string")
        .str.strip()
        .str.extract(r"^([A-Za-z]+)", expand=False)
    )
    room = apartment.str.cat(letter, sep="-", na_rep="")
    # If there was no letter, cat leaves a trailing '-'; fall back to apartment.
    room = room.where(letter.notna() & (letter != ""), apartment)
    return room.where(apartment.notna())


def _resolve_capacity(room_rows: pd.DataFrame) -> int:
    """Return one room's capacity (total physical beds).

    Strategy:
      1. Use a source capacity column if present (``_SOURCE_CAPACITY_COLUMNS``).
      2. Else derive it dynamically as the number of DISTINCT ``bed_code`` values
         ever observed in the room across the full booking history. Never a
         hardcoded per-type constant.
    """
    for col in _SOURCE_CAPACITY_COLUMNS:
        if col in room_rows.columns:
            vals = pd.to_numeric(room_rows[col], errors="coerce").dropna()
            if not vals.empty:
                return int(vals.max())
    return int(room_rows["bed_code"].dropna().nunique())


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _normalize_phone(series: pd.Series) -> pd.Series:
    """Reduce phone numbers to their last 10 digits for robust joining.

    Handles Excel-style scientific notation (e.g. ``9.16239E+11``) and stray
    punctuation that appears in the exports.
    """

    def clean(value) -> Optional[str]:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        if not text:
            return None
        if "e" in text.lower():  # scientific notation -> expand
            try:
                text = str(int(float(text)))
            except (ValueError, OverflowError):
                pass
        digits = "".join(ch for ch in text if ch.isdigit())
        if not digits:
            return None
        return digits[-10:] if len(digits) >= 10 else digits

    return series.map(clean)


def _windowed_metrics(
    group: pd.DataFrame, window_start: pd.Timestamp, as_of: pd.Timestamp, capacity: int,
    activation: Optional[pd.Timestamp] = None,
):
    """Occupancy % and accrued revenue for one room within [window_start, as_of].

    Each booking's span is clipped to the window. The denominator starts at the
    later of the window start and the room's first-ever occupancy, so a room is
    never penalised for time before it existed. Older (pre-window) data is used
    only to establish that first-occupancy reference — never discarded.

    Business rule (new apartments): when ``activation`` (apartments.start_date) is
    AFTER ``window_start`` the apartment is NEW — occupancy is measured ONLY from
    start_date onward and months before it are ignored. A new apartment with no
    occupancy in its own lifetime has NO fill history to judge, so occupancy is
    returned as UNKNOWN (NaN) rather than 0% (it must not be treated as
    "difficult to fill"). Existing apartments (start_date <= window_start) keep the
    original first-occupancy anchor exactly unchanged.
    """
    onboard = group["onboarding_date"]
    span_end = group["_span_end"]
    room_first = onboard.min()

    is_new = activation is not None and pd.notna(activation) and pd.Timestamp(activation) > window_start
    if is_new:
        eff_window_start = pd.Timestamp(activation)
    else:
        eff_window_start = max(window_start, room_first) if pd.notna(room_first) else window_start

    if pd.isna(eff_window_start) or eff_window_start > as_of:
        # New apartment not yet active within the data window -> no baseline.
        return (float("nan"), 0.0) if is_new else (0.0, 0.0)

    eff_start = onboard.clip(lower=eff_window_start)
    days = (span_end - eff_start).dt.days
    days = days.where(onboard.notna() & (span_end >= eff_window_start)).clip(lower=0)
    days = days.fillna(0)

    occupied_days = float(days.sum())
    revenue = float((group["monthly_rental"].fillna(0) * days / _DAYS_PER_MONTH).sum())

    if is_new and occupied_days == 0:
        # New apartment, zero occupancy since activation -> UNKNOWN, not 0%.
        return float("nan"), round(revenue, 2)

    window_len = max((as_of - eff_window_start).days, 1)
    occ_pct = min(100.0, 100 * occupied_days / (capacity * window_len)) if capacity else 0.0
    return round(occ_pct, 2), round(revenue, 2)


def _recent_demand(group: pd.DataFrame, recent_start: pd.Timestamp, as_of: pd.Timestamp):
    """Reconstruct per-bed vacancy/refill events for a room inside the recent window.

    Returns (vacancy_events, refills, fill_rate, avg_vacancy_days):
      * vacancy_event  = a real tenancy ended (actual_exit_date in the window).
      * refill         = that vacancy was followed by a later onboarding on the
                         same bed (a new tenant took it).
      * fill_rate      = refills / vacancy_events (None if no vacancy events).
      * avg_vacancy_days = mean gap (exit -> next onboarding) over refilled
                         events (None if none).

    Same-day "no-show" spans (stay <= 1 day) are treated as data noise and
    dropped before sequencing, matching the fill-time reconstruction rules.
    """
    spans = group[group["onboarding_date"].notna()].copy()
    if spans.empty:
        return 0, 0, None, None

    end = spans["actual_exit_date"].fillna(as_of)
    stay_days = (end - spans["onboarding_date"]).dt.days
    spans = spans[spans["actual_exit_date"].isna() | (stay_days > 1)]

    vacancy_events = 0
    refills = 0
    vacancy_days: list[int] = []

    for _, bed_rows in spans.groupby("bed_code"):
        bed_rows = bed_rows.sort_values("onboarding_date")
        exits = bed_rows["actual_exit_date"].tolist()
        onboards = bed_rows["onboarding_date"].tolist()
        for i, exit_date in enumerate(exits):
            if pd.isna(exit_date):
                continue  # still occupied -> not a vacancy event
            if exit_date < recent_start or exit_date > as_of:
                continue  # attribute the event to the window by its exit date
            vacancy_events += 1
            if i + 1 < len(onboards):  # a later tenant took the bed
                refills += 1
                vacancy_days.append(max((onboards[i + 1] - exit_date).days, 0))

    fill_rate = (refills / vacancy_events) if vacancy_events else None
    avg_vacancy = (sum(vacancy_days) / len(vacancy_days)) if vacancy_days else None
    return vacancy_events, refills, fill_rate, avg_vacancy


def _minmax(series: pd.Series, basis: Optional[pd.Series] = None) -> pd.Series:
    """Min-max scale to [0,1] across rooms. Constant/empty -> neutral 0.5.

    ``basis`` (optional boolean mask) restricts which rows define the min/max
    scale; every row is still scaled against that range. Used so never-booked
    seeded rooms (no demand history) do NOT shift the demand normalisation of
    rooms that have history — their scores stay identical.
    """
    s = series.astype(float)
    ref = s[basis] if basis is not None and basis.any() else s
    lo, hi = ref.min(), ref.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(0.5, index=s.index)
    return (s - lo) / (hi - lo)


def _compute_demand_score(inv: pd.DataFrame) -> pd.Series:
    """Composite 0-100 demand score, normalised dynamically across the room set.

    Blends four min-max-normalised signals (no hardcoded thresholds):
      recent occupancy %, recent fill rate, LOW average vacancy (inverted),
      and revenue performance (recent revenue). Missing signals -> neutral 0.5.
    """
    w = DEMAND_SCORE_WEIGHTS
    # Normalise the demand signals over rooms WITH booking history only, so
    # never-booked seeded rooms (A33/A34) leave existing rooms' scores unchanged.
    basis = None
    if "_never_booked" in inv.columns:
        booked = ~inv["_never_booked"].astype(bool)
        if booked.any():
            basis = booked
    occ = _minmax(inv["recent_occupancy_pct"], basis).fillna(0.5)
    fill = _minmax(inv["recent_fill_rate"], basis).fillna(0.5)
    low_vac = (1 - _minmax(inv["average_vacancy_days_recent"], basis)).fillna(0.5)
    rev = _minmax(inv["recent_revenue"], basis).fillna(0.5)

    score = (
        w["recent_occupancy"] * occ
        + w["recent_fill_rate"] * fill
        + w["low_vacancy"] * low_vac
        + w["revenue_performance"] * rev
    ) * 100
    return score.round(2)


# Occupant attributes enriched onto bookings for occupancy + compatibility.
# The set is dynamic: only attributes whose source column exists are populated,
# so compatibility automatically strengthens as more tenant data is available.
_OCCUPANT_ATTRS = ("state", "gender", "occupation", "age")


def _derive_occupation_series(tenants: pd.DataFrame) -> pd.Series:
    """Classify each tenant as 'student' / 'working' from whatever fields exist.

    Fully dynamic: uses course/profession/company/designation columns when
    present, returns None when nothing indicative is available.
    """
    cols = set(tenants.columns)

    def _val(row, c):
        if c in cols:
            v = row.get(c)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return None

    def _classify(row):
        if _val(row, "course"):
            return "student"
        prof = _val(row, "profession")
        if prof:
            return "student" if "student" in prof.lower() else "working"
        if _val(row, "company_name") or _val(row, "designation") or _val(row, "company_state"):
            return "working"
        return None

    if not ({"course", "profession", "company_name", "designation", "company_state"} & cols):
        return pd.Series(pd.NA, index=tenants.index, dtype="object")
    return tenants.apply(_classify, axis=1)


def _tenant_profile(tenants: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Per-tenant normalised attributes (keyed by phone & name) for enrichment."""
    idx = tenants.index
    prof = pd.DataFrame(index=idx)
    prof["_phone"] = (
        _normalize_phone(tenants["phone"]) if "phone" in tenants.columns
        else pd.Series(pd.NA, index=idx, dtype="object")
    )
    prof["_name"] = (
        normalize_series(tenants["full_name"]) if "full_name" in tenants.columns
        else pd.Series(pd.NA, index=idx, dtype="object")
    )
    prof["state"] = tenants["state"] if "state" in tenants.columns else pd.NA
    prof["gender"] = (
        normalize_series(tenants["gender"]) if "gender" in tenants.columns else pd.NA
    )

    age = pd.Series(np.nan, index=idx, dtype="float64")
    if "age" in tenants.columns:
        age = pd.to_numeric(tenants["age"], errors="coerce")
    if "date_of_birth" in tenants.columns:
        dob = pd.to_datetime(tenants["date_of_birth"], errors="coerce")
        derived = (as_of - dob).dt.days / 365.25
        age = age.fillna(derived)
    # Guard against absurd ages from bad DOBs.
    prof["age"] = age.where((age >= 10) & (age <= 100))

    prof["occupation"] = _derive_occupation_series(tenants)
    return prof


def _attr_maps(profile: Optional[pd.DataFrame], attr: str):
    """Return (phone->attr, name->attr) dicts for one attribute."""
    if profile is None or attr not in profile.columns:
        return {}, {}
    pm = profile.dropna(subset=["_phone", attr])
    nm = profile.dropna(subset=["_name", attr])
    return (dict(zip(pm["_phone"], pm[attr])), dict(zip(nm["_name"], nm[attr])))


def _lifecycle_attrs(frame: pd.DataFrame) -> dict:
    """Carry forward any explicit lifecycle/status columns (mode of non-null).

    Empty when the source has no lifecycle columns (today's CSVs). Once the DB
    or beds_master supplies them, they land on Room Inventory so
    ``annotate_lifecycle_status`` can label Active / Inactive / Unknown.
    """
    out = {}
    for col in ACTIVE_STATUS_TOKENS:
        if col not in frame.columns:
            continue
        vals = frame[col].dropna()
        if vals.empty:
            continue
        mode = vals.astype(str).str.strip().mode()
        out[col] = mode.iloc[0] if not mode.empty else vals.iloc[-1]
    return out


# --------------------------------------------------------------------------- #
# beds_master — official bed inventory metadata (NOT used in ranking)
# --------------------------------------------------------------------------- #

_BEDS_MASTER_KEEP = (
    "apartment_code",
    "bed_code",
    "gender_allowed",
    "bed_status",
    "bed_lifecycle_status",
    "toilet_type",
    "current_rate",
)


def _canonical_master_gender(value) -> Optional[str]:
    """Normalize beds_master gender_allowed; None if blank."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none", "null", ""}:
        return None
    n = normalize_text(raw)
    if not n:
        return None
    if n in {"any", "all", "both", "open"}:
        return "Any"
    if n in {"unisex", "mixed"}:
        return "Mixed" if n == "mixed" else "Unisex"
    if n.startswith("m") and "female" not in n:
        return "Male"
    if n.startswith("f"):
        return "Female"
    return raw.title()


def _is_inactive_bed_status(value) -> bool:
    n = normalize_text(value)
    if not n:
        return False
    return n in {
        "not-active",
        "not active",
        "inactive",
        "disabled",
        "blocked",
    } or n.startswith("not-active") or n.startswith("inactive")


def normalize_beds_master(raw: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Keep/rename the official beds_master columns used by Room Inventory."""
    empty = pd.DataFrame(columns=list(_BEDS_MASTER_KEEP))
    if raw is None or not isinstance(raw, pd.DataFrame) or raw.empty:
        return empty

    out = raw.copy()
    if "current_rate" not in out.columns and "monthly_rate" in out.columns:
        out["current_rate"] = out["monthly_rate"]
    # Production export: Occupied/Vacant in `occupancy` (may already be aliased).
    if "bed_lifecycle_status" not in out.columns and "occupancy" in out.columns:
        out["bed_lifecycle_status"] = out["occupancy"]
    # Snapshot file uses bed_lifecycle_status; catalog uses Occupied/Vacant status.
    if "bed_lifecycle_status" not in out.columns and "status" in out.columns:
        mapped = normalize_series(out["status"]).map(
            {
                "occupied": "occupied",
                "vacant": "vacant",
                "live": "occupied",
                "notice": "notice",
                "booked": "booked",
            }
        )
        out["bed_lifecycle_status"] = mapped
    # Canonicalize lifecycle labels from Occupied/Vacant text.
    if "bed_lifecycle_status" in out.columns:
        life_n = normalize_series(out["bed_lifecycle_status"])
        out["bed_lifecycle_status"] = life_n.map(
            {
                "occupied": "occupied",
                "vacant": "vacant",
                "live": "occupied",
                "notice": "notice",
                "booked": "booked",
                "not-active": "vacant",
            }
        ).fillna(out["bed_lifecycle_status"])

    # Production master may omit bed_status; DataLoader fills Not-Active from
    # Backup_old query (8) before this runs. Remaining blanks default to Live.
    if "bed_status" not in out.columns or out["bed_status"].isna().all():
        out["bed_status"] = "Live"

    for col in _BEDS_MASTER_KEEP:
        if col not in out.columns:
            out[col] = pd.NA

    out = out[list(_BEDS_MASTER_KEEP)].copy()
    out["apartment_code"] = out["apartment_code"].astype("string").str.strip()
    out["bed_code"] = out["bed_code"].astype("string").str.strip()
    out = out[out["apartment_code"].notna() & out["bed_code"].notna()].copy()
    out["gender_allowed"] = out["gender_allowed"].map(_canonical_master_gender)
    out["current_rate"] = pd.to_numeric(out["current_rate"], errors="coerce")
    for col in ("bed_status", "bed_lifecycle_status", "toilet_type"):
        out[col] = out[col].map(
            lambda v: str(v).strip()
            if v is not None and not (isinstance(v, float) and pd.isna(v)) and str(v).strip()
            else pd.NA
        )
    # Fill remaining null bed_status with Live after strip.
    out["bed_status"] = out["bed_status"].fillna("Live")
    out.loc[
        out["bed_status"].astype(str).str.strip().isin({"", "<NA>", "nan", "None"}),
        "bed_status",
    ] = "Live"
    return out.reset_index(drop=True)


def _beds_master_key(df: pd.DataFrame) -> pd.Series:
    return (
        df["apartment_code"].astype(str).str.strip().str.upper()
        + "|"
        + df["bed_code"].astype(str).str.strip().str.upper()
    )


def _toilet_type_preference(value) -> int:
    """Lower is better for duplicate resolve: Common (0) > Attached (1) > other (2)."""
    n = normalize_text(value) or ""
    if n == "common":
        return 0
    if n == "attached":
        return 1
    return 2


def _pick_duplicate_bed_row(group: pd.DataFrame, bedmap_rate=None) -> pd.Series:
    """Choose one beds_master row for a duplicate apartment+bed key.

    Selects the row that represents the ACTIVE / current bed used by Vishful:
      1. **Bed Map** — the row whose ``current_rate`` equals the live Bed Map rate
         (the active bed_rates record the Vishful application shows). Preferred
         whenever the Bed Map value is available.
      2. **Effective dates** — if the source carries ``from_date`` / ``to_date``,
         the row active today (from_date <= today AND (to_date null OR >= today)).
      3. **Fallback** — the latest valid source row (highest source order).
    Toilet-type is NEVER used to pick the rate.
    """
    g = group
    pool = g

    if bedmap_rate is not None and not (
        isinstance(bedmap_rate, float) and pd.isna(bedmap_rate)
    ):
        # 1. Match the live Bed Map active rate.
        rate = pd.to_numeric(g["current_rate"], errors="coerce")
        match = g.loc[(rate - float(bedmap_rate)).abs() <= 0.5]
        if not match.empty:
            pool = match
    elif {"from_date", "to_date"}.issubset(g.columns):
        # 2. Bed Map unavailable -> active bed_rates row by effective dates.
        today = pd.Timestamp.today().normalize()
        fd = pd.to_datetime(g["from_date"], errors="coerce")
        td = pd.to_datetime(g["to_date"], errors="coerce")
        active = g.loc[(fd.isna() | (fd <= today)) & (td.isna() | (td >= today))]
        if not active.empty:
            pool = active

    # 3. Latest valid source row within the selected pool (final tiebreak).
    has_rate = pool["current_rate"].notna()
    if has_rate.any():
        pool = pool.loc[has_rate]
    return pool.loc[pool["_src_order"].idxmax()]


def resolve_beds_master(
    raw: Optional[pd.DataFrame], bed_map: Optional[pd.DataFrame] = None
) -> tuple[pd.DataFrame, dict]:
    """Normalize beds_master, report duplicate keys, return one row per bed.

    Duplicate ``apartment_code + bed_code`` rows are always reported. For each
    duplicate the ACTIVE/current bed is selected: the row matching the live
    ``bed_map`` rate when available, else the effective-date-active row
    (from_date/to_date), else the latest valid source row. Toilet-type preference
    is not used. ``bed_map`` is the DataLoader Bed Map (``apartment_code`` /
    ``bed_code`` / ``bed_rate``); when None the date/latest fallbacks apply.
    """
    report = {
        "beds_loaded": 0,
        "unique_keys": 0,
        "duplicate_keys": [],
        "conflict_keys": [],
        "rows_before_resolve": 0,
        "rows_after_resolve": 0,
    }
    master = normalize_beds_master(raw)
    report["beds_loaded"] = int(len(master))
    report["rows_before_resolve"] = int(len(master))
    if master.empty:
        return master, report

    # Live Bed Map active rate per APT|BED (drives duplicate selection).
    bedmap_rate: Dict[str, float] = {}
    if (
        bed_map is not None
        and isinstance(bed_map, pd.DataFrame)
        and not bed_map.empty
        and "bed_rate" in bed_map.columns
    ):
        for _, _br in bed_map.iterrows():
            _a = str(_br.get("apartment_code") or "").strip().upper()
            _b = str(_br.get("bed_code") or "").strip().upper()
            _rt = _br.get("bed_rate")
            if _a and _b and _rt is not None and not (
                isinstance(_rt, float) and pd.isna(_rt)
            ):
                bedmap_rate[f"{_a}|{_b}"] = float(_rt)

    master = master.copy()
    master["_key"] = _beds_master_key(master)
    master["_src_order"] = range(len(master))
    report["unique_keys"] = int(master["_key"].nunique())

    dup_keys = master["_key"].value_counts()
    dup_keys = dup_keys[dup_keys > 1]
    report["duplicate_keys"] = sorted(dup_keys.index.tolist())

    conflict_keys = []
    for key, g in master.groupby("_key"):
        if len(g) < 2:
            continue
        # Gender: case-normalized canonical labels only.
        g_vals = {
            str(v).strip()
            for v in g["gender_allowed"].dropna().tolist()
            if str(v).strip() and str(v).strip().lower() not in {"nan", "<na>"}
        }
        if len(g_vals) > 1:
            conflict_keys.append(
                {"key": key, "column": "gender_allowed", "values": sorted(g_vals)}
            )
        # bed_status: only compare non-null snapshot values (Live vs Not-Active).
        s_vals = {
            str(v).strip().title().replace("_", "-")
            for v in g["bed_status"].dropna().tolist()
            if str(v).strip() and str(v).strip().lower() not in {"nan", "<na>"}
        }
        # Normalize Live/Available/Active → Live; Not-Active/Inactive → Not-Active
        norm_s = set()
        for v in s_vals:
            n = v.lower().replace(" ", "-")
            if n in {"live", "available", "active"}:
                norm_s.add("Live")
            elif n in {"not-active", "inactive", "disabled", "blocked"}:
                norm_s.add("Not-Active")
            else:
                norm_s.add(v)
        if len(norm_s) > 1:
            conflict_keys.append(
                {"key": key, "column": "bed_status", "values": sorted(norm_s)}
            )
        # toilet_type: case-insensitive
        t_vals = {
            str(v).strip().title()
            for v in g["toilet_type"].dropna().tolist()
            if str(v).strip() and str(v).strip().lower() not in {"nan", "<na>"}
        }
        if len(t_vals) > 1:
            conflict_keys.append(
                {"key": key, "column": "toilet_type", "values": sorted(t_vals)}
            )
    report["conflict_keys"] = conflict_keys

    # One row per apartment+bed: active Bed Map rate -> effective dates -> latest.
    picked = [
        _pick_duplicate_bed_row(g, bedmap_rate.get(key))
        for key, g in master.groupby("_key", sort=True)
    ]
    resolved = pd.DataFrame(picked).reset_index(drop=True)
    resolved = resolved.drop(columns=["_key", "_src_order"], errors="ignore")
    # Restore canonical column order from normalize.
    keep_cols = [c for c in _BEDS_MASTER_KEEP if c in resolved.columns]
    resolved = resolved[keep_cols].copy()
    report["rows_after_resolve"] = int(len(resolved))
    report["resolve_note"] = (
        "Duplicate apartment+bed rows resolve to the row matching the live Bed Map "
        "active rate; else the effective-date-active row (from_date/to_date); else "
        "the latest valid source row. Toilet-type preference is not used."
    )
    return resolved.reset_index(drop=True), report


def print_beds_master_validation(
    report: dict,
    match_stats: Optional[dict] = None,
) -> None:
    """Print beds_master load / join validation (display-only diagnostics)."""
    print("\n=== beds_master validation ===")
    print(f"Beds loaded from master     : {report.get('beds_loaded', 0)}")
    print(f"Unique apartment+bed keys   : {report.get('unique_keys', 0)}")
    print(f"Rows before resolve         : {report.get('rows_before_resolve', 0)}")
    print(f"Rows after resolve          : {report.get('rows_after_resolve', 0)}")
    dups = report.get("duplicate_keys") or []
    print(f"Duplicate join keys         : {len(dups)}")
    if dups:
        preview = ", ".join(dups[:20])
        more = f" ... (+{len(dups) - 20})" if len(dups) > 20 else ""
        print(f"  keys: {preview}{more}")
    conflicts = report.get("conflict_keys") or []
    print(f"Conflicting duplicate keys  : {len(conflicts)}")
    for c in conflicts[:15]:
        print(f"  {c['key']} | {c['column']} -> {c['values']}")
    if report.get("resolve_note"):
        print(f"Resolve note                : {report['resolve_note']}")
    if match_stats:
        print(f"Matched inventory beds      : {match_stats.get('matched_beds', 0)}")
        print(f"Unmatched inventory beds    : {match_stats.get('unmatched_beds', 0)}")
        print(f"Rooms gender changed        : {match_stats.get('gender_changed', 0)}")
        print(f"Rooms lifecycle changed     : {match_stats.get('lifecycle_changed', 0)}")
        print(
            "Inactive apartments (master): "
            f"{match_stats.get('inactive_apartments', [])}"
        )
    print("=== end beds_master validation ===\n")


def _room_master_fields(
    apartment_code,
    bed_codes: list,
    master_by_key: dict,
) -> dict:
    """Aggregate beds_master attributes for one room (bed list)."""
    apt = str(apartment_code).strip().upper() if pd.notna(apartment_code) else ""
    genders, statuses, lifecycles, toilets, rates = [], [], [], [], []
    matched = 0
    for bed in bed_codes:
        key = f"{apt}|{str(bed).strip().upper()}"
        row = master_by_key.get(key)
        if row is None:
            continue
        matched += 1
        if row.get("gender_allowed"):
            genders.append(row["gender_allowed"])
        if row.get("bed_status") is not None and not (
            isinstance(row.get("bed_status"), float) and pd.isna(row.get("bed_status"))
        ):
            statuses.append(str(row["bed_status"]).strip())
        if row.get("bed_lifecycle_status") is not None and not (
            isinstance(row.get("bed_lifecycle_status"), float)
            and pd.isna(row.get("bed_lifecycle_status"))
        ):
            lifecycles.append(str(row["bed_lifecycle_status"]).strip())
        if row.get("toilet_type") is not None and not (
            isinstance(row.get("toilet_type"), float) and pd.isna(row.get("toilet_type"))
        ):
            toilets.append(str(row["toilet_type"]).strip())
        rate = row.get("current_rate")
        if rate is not None and not (isinstance(rate, float) and pd.isna(rate)):
            try:
                rates.append(float(rate))
            except (TypeError, ValueError):
                pass

    # Gender Allowed: master only. Missing → Unknown (never infer from occupants).
    uniq_g = sorted({g for g in genders if g})
    if not uniq_g:
        gender_allowed = "Unknown"
    elif len(uniq_g) == 1:
        gender_allowed = uniq_g[0]
    else:
        gender_allowed = "Mixed"

    if any(_is_inactive_bed_status(s) for s in statuses):
        bed_status = "Not-Active"
    elif statuses:
        # Prefer canonical Live when any live-like token present.
        bed_status = statuses[0]
        for s in statuses:
            if normalize_text(s) in {"live", "available", "active"}:
                bed_status = "Live" if normalize_text(s) == "live" else s
                break
    else:
        bed_status = None

    uniq_life = sorted({x for x in lifecycles if x})
    if not uniq_life:
        bed_lifecycle_status = None
    elif len(uniq_life) == 1:
        bed_lifecycle_status = uniq_life[0]
    else:
        bed_lifecycle_status = "/".join(uniq_life)

    uniq_toilet = sorted({t for t in toilets if t})
    if not uniq_toilet:
        toilet_type = None
    elif len(uniq_toilet) == 1:
        toilet_type = uniq_toilet[0]
    else:
        toilet_type = "Mixed"

    current_rate = float(max(rates)) if rates else None

    return {
        "gender_allowed": gender_allowed,
        "bed_status": bed_status,
        "bed_lifecycle_status": bed_lifecycle_status,
        "toilet_type": toilet_type,
        "current_rate": current_rate,
        "_master_beds_matched": matched,
        "_master_beds_total": len(bed_codes),
    }


# --------------------------------------------------------------------------- #
# Public builders
# --------------------------------------------------------------------------- #


# Statuses that hold a bed right now. Production exports use ``Booked`` for
# reserved beds that must not be recommended; keep local so utils stays unchanged.
_HOLD_STATUSES = set(OCCUPIED_STATUSES) | {"booked"}
# staying_status values that mean a current_occupancy Occupied row is stale.
_DEAD_OCCUPANCY_STAY = {
    "cancelled",
    "canceled",
    "exited",
    "exit",
    "rejected",
    "no-show",
    "noshow",
}


def _optional_current_occupancy() -> Optional[pd.DataFrame]:
    """Load current_occupancy when available (recommendation occupancy source)."""
    try:
        from data_loader import DataLoader

        return DataLoader().current_occupancy()
    except Exception:  # noqa: BLE001
        return None


def _apt_bed_key(apartment_code, bed_code) -> Optional[str]:
    a = str(apartment_code or "").strip().upper()
    b = str(bed_code or "").strip().upper()
    if not a or not b or a in {"NAN", "<NA>", "NONE"}:
        return None
    return f"{a}|{b}"


def _occupied_keys_from_beds_master(master: Optional[pd.DataFrame]) -> set:
    """Beds Query (33) marks Occupied/Booked (lifecycle), keyed APT|BED."""
    if master is None or not isinstance(master, pd.DataFrame) or master.empty:
        return set()
    life_col = None
    for cand in ("bed_lifecycle_status", "occupancy"):
        if cand in master.columns:
            life_col = cand
            break
    if (
        life_col is None
        or "apartment_code" not in master.columns
        or "bed_code" not in master.columns
    ):
        return set()
    life_n = normalize_series(master[life_col])
    mask = life_n.isin({"occupied", "booked"})
    keys = set()
    for apt, bed in zip(
        master.loc[mask, "apartment_code"],
        master.loc[mask, "bed_code"],
    ):
        key = _apt_bed_key(apt, bed)
        if key:
            keys.add(key)
    return keys


def _held_bed_keys_from_occupancy(
    occ: Optional[pd.DataFrame],
    beds_master: Optional[pd.DataFrame] = None,
) -> set:
    """Return ``APT|BED`` keys currently held per the occupancy snapshot.

    A bed is held when ``occupancy_status`` is Occupied/Booked and
    ``staying_status`` is not a dead terminal status (e.g. Cancelled).

    Conflict fallback (Query 33): if the snapshot would release a bed solely
    because of a dead stay (Occupied/Booked + Cancelled/Exited/…), but
    ``beds_master`` still marks that bed Occupied/Booked, keep it held.
    Snapshot Vacant rows are never held by this fallback.
    """
    if occ is None or not isinstance(occ, pd.DataFrame) or occ.empty:
        return set()
    if "occupancy_status" not in occ.columns:
        return set()
    if "apartment_code" not in occ.columns or "bed_code" not in occ.columns:
        return set()

    status_n = normalize_series(occ["occupancy_status"])
    if "staying_status" in occ.columns:
        stay_n = occ["staying_status"].map(
            lambda v: normalize_text(v) if pd.notna(v) else ""
        )
    else:
        stay_n = pd.Series("", index=occ.index, dtype="object")

    occ_like = status_n.isin({"occupied", "booked"})
    dead_stay = stay_n.isin(_DEAD_OCCUPANCY_STAY)
    held_mask = occ_like & ~dead_stay

    keys = set()
    for apt, bed in zip(
        occ.loc[held_mask, "apartment_code"],
        occ.loc[held_mask, "bed_code"],
    ):
        key = _apt_bed_key(apt, bed)
        if key:
            keys.add(key)

    # Occupied/Booked + Cancelled (etc.): keep held if master still Occupied.
    conflict_mask = occ_like & dead_stay
    if conflict_mask.any() and beds_master is not None:
        master_occupied = _occupied_keys_from_beds_master(beds_master)
        if master_occupied:
            for apt, bed in zip(
                occ.loc[conflict_mask, "apartment_code"],
                occ.loc[conflict_mask, "bed_code"],
            ):
                key = _apt_bed_key(apt, bed)
                if key and key in master_occupied:
                    keys.add(key)

    return keys


def _seed_physical_beds(
    bookings: pd.DataFrame,
    beds_master: Optional[pd.DataFrame],
    bed_map: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """Seed the room universe with physical beds that have no booking history yet.

    Business enhancement: the room universe is the physical inventory (bed
    catalog) UNIONED with booking history — not booking history alone. Newly
    created Live apartments/beds (e.g. A33 / A34) that have never been booked
    therefore still appear in Inventory. Beds already present in ``bookings`` are
    left completely untouched, so all existing booking-based logic (occupancy,
    demand, revenue, windowed metrics) is unchanged for rooms with history.

    Seeded beds carry EMPTY history — no dates, no tenant, no rent — so they are
    never ``_active`` (occupied = 0, available = live beds, occupancy = 0%). Their
    ``current_rate`` is filled later from beds_master (Q84); a missing rate stays
    NULL (never invented). Not-Active beds are not seeded (inactive inventory does
    not create a new room). Days-vacant stays null via the bed-map layer.

    Backward compatible: when no physical bed source is supplied (beds_master and
    bed_map both absent/empty), the bookings frame is returned unchanged — the old
    bookings-only universe, i.e. legacy behaviour is preserved.
    """
    phys = None
    for src in (beds_master, bed_map):
        if (
            isinstance(src, pd.DataFrame)
            and not src.empty
            and {"apartment_code", "bed_code"} <= set(src.columns)
        ):
            phys = src.copy()
            break
    if phys is None:
        return bookings

    phys["apartment_code"] = phys["apartment_code"].astype("string").str.strip()
    phys["bed_code"] = phys["bed_code"].astype("string").str.strip()
    phys = phys[phys["apartment_code"].notna() & phys["bed_code"].notna()].copy()
    # Inactive (Not-Active) beds never seed a NEW room.
    if "bed_status" in phys.columns:
        na = (
            phys["bed_status"].astype(str).str.strip().str.lower().str.replace(" ", "-", regex=False)
            .isin({"not-active", "inactive", "disabled", "blocked"})
        )
        phys = phys[~na].copy()
    phys = phys.drop_duplicates(["apartment_code", "bed_code"], keep="first")
    if phys.empty:
        return bookings

    existing = set(
        zip(
            bookings["apartment_code"].astype(str).str.strip().str.upper(),
            bookings["bed_code"].astype(str).str.strip().str.upper(),
        )
    )
    is_new = [
        (str(a).strip().upper(), str(b).strip().upper()) not in existing
        for a, b in zip(phys["apartment_code"], phys["bed_code"])
    ]
    new_beds = phys[pd.Series(is_new, index=phys.index)]
    if new_beds.empty:
        return bookings

    bookings = bookings.copy()
    bookings["_seeded"] = False
    placeholder = pd.DataFrame(index=range(len(new_beds)), columns=bookings.columns)
    placeholder["_seeded"] = True
    placeholder["apartment_code"] = new_beds["apartment_code"].values
    placeholder["bed_code"] = new_beds["bed_code"].values
    if "bed_type" in bookings.columns and "bed_type" in new_beds.columns:
        placeholder["bed_type"] = new_beds["bed_type"].values
    # Empty history: no active stay, no tenant, no rent, no dates.
    for col in (
        "onboarding_date", "actual_exit_date", "booking_date",
        "notice_date", "estimated_exit_date",
    ):
        if col in placeholder.columns:
            placeholder[col] = pd.NaT
    if "monthly_rental" in placeholder.columns:
        placeholder["monthly_rental"] = pd.NA
    if "staying_status" in placeholder.columns:
        placeholder["staying_status"] = pd.NA

    logger.info(
        "Seeded %d never-booked physical beds into the room universe "
        "(apartments: %s).",
        len(placeholder),
        ", ".join(sorted(new_beds["apartment_code"].dropna().unique().tolist())),
    )
    return pd.concat([bookings, placeholder], ignore_index=True)


def build_room_inventory(
    bookings: pd.DataFrame,
    tenants: Optional[pd.DataFrame] = None,
    beds_master: Optional[pd.DataFrame] = None,
    current_occupancy: Optional[pd.DataFrame] = None,
    as_of: Optional[pd.Timestamp] = None,
    historical_start: Optional[pd.Timestamp] = None,
    recent_start: Optional[pd.Timestamp] = None,
    bed_map: Optional[pd.DataFrame] = None,
    apartments: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build the Room Master — the single source of truth for room identity.

    Occupancy & revenue are reported over windows (see utils config):
      * recent / business baseline  [recent_start .. as_of]  (default 2025-01-01)
        -> ``recent_occupancy_pct`` / ``recent_revenue`` (Demand inputs)
        -> ``historical_occupancy_pct`` / ``historical_revenue`` (business
           priority + tier-2 ranking — same 2025+ window; pre-2025 history is
           not representative of current operations).
    Data older than ``recent_start`` is NOT discarded — it still defines room
    capacity, room age and first-occupancy references — it is simply excluded
    from occupancy baselines used for Demand and business priority.

    ``beds_master`` (optional) supplies authoritative metadata:
    gender_allowed, bed_status, bed_lifecycle_status, toilet_type, current_rate.
    Gender Allowed is NEVER inferred from occupants.

    ``current_occupancy`` (optional, production query 34) is the live occupancy
    snapshot. When present, beds marked Occupied/Booked there are treated as
    held for available_beds (unioned with booking holds). If a snapshot row is
    Occupied/Booked but Cancelled (dead stay), Query (33) ``beds_master``
    Occupied/Booked is the fallback authority and the bed stays held.

    Parameters
    ----------
    bookings : the ``bookings`` logical table (allocation + booking history).
    tenants  : optional ``tenants`` table, used only to look up the home state
               of current occupants (``occupant_states``).
    beds_master : optional official bed inventory master.
    current_occupancy : optional live occupancy snapshot.
    as_of    : reference "today". Defaults to the latest date seen in the data,
               so results are reproducible and not tied to the wall clock.
    historical_start, recent_start : window starts (default to the utils config).

    Returns one row per ROOM with the columns documented in the module header.
    """
    historical_start = pd.Timestamp(historical_start or HISTORICAL_BASELINE_START)
    recent_start = pd.Timestamp(recent_start or RECENT_BASELINE_START)

    # Per-apartment activation date (apartments.start_date). Used to measure new
    # apartments' occupancy ONLY from start_date onward (see _windowed_metrics).
    # Existing apartments (start_date <= recent_start) are unaffected.
    activation_by_apt: dict = {}
    if apartments is not None and not apartments.empty and \
            "apartment_code" in apartments.columns and "start_date" in apartments.columns:
        _sd = pd.to_datetime(apartments["start_date"], errors="coerce", utc=True)
        try:
            _sd = _sd.dt.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        activation_by_apt = dict(zip(apartments["apartment_code"].astype(str).str.strip(), _sd))

    # Production bookings omit bed_type; fill before the hard require.
    if "bed_type" not in bookings.columns:
        bookings = bookings.copy()
        bookings["bed_type"] = "Unknown"

    require_columns(
        bookings,
        [
            "apartment_code",
            "bed_code",
            "bed_type",
            "onboarding_date",
            "actual_exit_date",
            "monthly_rental",
            "staying_status",
        ],
        table="bookings",
    )

    if current_occupancy is None:
        current_occupancy = _optional_current_occupancy()

    master_resolved, master_report = resolve_beds_master(beds_master, bed_map=bed_map)
    # Occupancy holds after master resolve so Cancelled+Occupied can defer to Q33.
    held_from_occupancy = _held_bed_keys_from_occupancy(
        current_occupancy, beds_master=master_resolved
    )

    # Bed Map = the single CURRENT-occupancy source. When present it OVERRIDES the
    # Q35∪Q50 reconstruction for every covered bed (available iff status=='vacant').
    # Beds not covered (e.g. inactive apartments absent from the bed-map) keep the
    # reconstruction as fallback. Historical windows/demand/occupant states are
    # untouched — only the current occupied/available layer moves.
    bed_map_available: Dict[str, bool] = {}
    if bed_map is not None and isinstance(bed_map, pd.DataFrame) and not bed_map.empty:
        for _, _bm in bed_map.iterrows():
            _k = _apt_bed_key(_bm.get("apartment_code"), _bm.get("bed_code"))
            if _k:
                bed_map_available[_k] = bool(_bm.get("is_available"))

    master_by_key: dict = {}
    if not master_resolved.empty:
        keyed = master_resolved.copy()
        keyed["_key"] = _beds_master_key(keyed)
        master_by_key = {
            k: {
                "gender_allowed": r["gender_allowed"],
                "bed_status": r["bed_status"],
                "bed_lifecycle_status": r["bed_lifecycle_status"],
                "toilet_type": r["toilet_type"],
                "current_rate": r["current_rate"],
            }
            for k, r in keyed.set_index("_key").to_dict(orient="index").items()
        }

    df = bookings.copy()

    # --- Seed the room universe from physical inventory ----------------------
    # Rooms come from the physical bed catalog UNIONED with booking history so
    # newly-created Live apartments/beds with no bookings yet still appear.
    # Beds already in bookings are untouched (existing logic unchanged). Legacy
    # compatible: no-op when no physical bed source is available.
    df = _seed_physical_beds(df, beds_master, bed_map)

    # --- Ensure correct dtypes (defensive; data_loader already coerces) ------
    for col in ("onboarding_date", "actual_exit_date", "booking_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    df["monthly_rental"] = pd.to_numeric(df["monthly_rental"], errors="coerce")

    # --- Keep only rows that can be attributed to a physical bed -------------
    n_raw = len(df)
    df = df[df["apartment_code"].notna() & df["bed_code"].notna()].copy()
    df["apartment_code"] = df["apartment_code"].astype("string").str.strip()
    df["bed_code"] = df["bed_code"].astype("string").str.strip()
    dropped_no_bed = n_raw - len(df)

    # --- Assign the opaque room_code (centralised identity) ------------------
    df["room_code"] = _resolve_room_code(df)
    df = df[df["room_code"].notna()].copy()

    # --- Reference date --------------------------------------------------
    if as_of is None:
        date_parts = [df["onboarding_date"], df["actual_exit_date"]]
        if "booking_date" in df.columns:
            date_parts.append(df["booking_date"])
        as_of = pd.concat(date_parts).max()
    as_of = pd.Timestamp(as_of)

    # --- Occupancy spans ------------------------------------------------------
    # A booking occupies its bed from onboarding until exit (or until `as_of`
    # if it is still ongoing). Windowed metrics (below) clip these spans to the
    # historical / recent baselines. Overlaps/no-shows contribute >=0 days.
    df["_span_end"] = df["actual_exit_date"].fillna(as_of).clip(upper=as_of)

    # --- Active (currently occupied / held) rows ------------------------------
    # Booked holds a bed even when onboarding_date is still null (use booking_date).
    status_norm = normalize_series(df["staying_status"])
    hold_start = df["onboarding_date"].copy()
    if "booking_date" in df.columns:
        hold_start = hold_start.fillna(df["booking_date"])
    df["_active"] = (
        df["actual_exit_date"].isna()
        & hold_start.notna()
        & status_norm.isin(_HOLD_STATUSES)
    )
    # Keep a usable onboarding for dedupe sort among Booked-null-onboarding rows.
    df["_hold_start"] = hold_start

    # --- Occupant attribute enrichment (state, gender, occupation, age) ------
    # Attributes are matched from the tenant table by phone first, then name.
    profile = _tenant_profile(tenants, as_of) if tenants is not None else None
    df_phone = (
        _normalize_phone(df["phone"]) if "phone" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="object")
    )
    df_name = (
        normalize_series(df["full_name"]) if "full_name" in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="object")
    )
    for attr in _OCCUPANT_ATTRS:
        phone_map, name_map = _attr_maps(profile, attr)
        mapped = df_phone.map(phone_map)
        mapped = mapped.fillna(df_name.map(name_map))
        df[f"_occ_{attr}"] = mapped

    # --- Aggregate to one row per room ---------------------------------------
    records = []
    quality: Dict[str, int] = {
        "rooms_capacity_lt_bed_type": 0,
        "rooms_overbooked_clamped": 0,
        "rooms_bed_type_inconsistent": 0,
    }

    for room_code, g in df.groupby("room_code", sort=True):
        apartment_code = g["apartment_code"].mode(dropna=True)
        apartment_code = apartment_code.iloc[0] if not apartment_code.empty else pd.NA

        bed_universe = sorted(g["bed_code"].dropna().unique().tolist())
        capacity = _resolve_capacity(g)

        # A room is "never booked" when every one of its rows is a seeded
        # physical bed (no booking history). Used only to keep demand
        # normalisation stable — display fields are unaffected.
        never_booked = bool(g["_seeded"].all()) if "_seeded" in g.columns else False

        bed_types = g["bed_type"].dropna()
        bed_type = bed_types.mode().iloc[0] if not bed_types.empty else pd.NA
        if bed_types.nunique() > 1:
            quality["rooms_bed_type_inconsistent"] += 1

        active = g[g["_active"]]
        # Dedupe active rows to one per bed (latest hold start wins).
        sort_col = "_hold_start" if "_hold_start" in active.columns else "onboarding_date"
        active_dedup = active.sort_values(sort_col).drop_duplicates(
            "bed_code", keep="last"
        )
        occupied_bed_codes = sorted(active_dedup["bed_code"].dropna().unique().tolist())

        # Union with live occupancy snapshot (recommendation source).
        if held_from_occupancy:
            apt_u = str(apartment_code).strip().upper()
            snap_beds = [
                b
                for b in bed_universe
                if f"{apt_u}|{str(b).strip().upper()}" in held_from_occupancy
            ]
            occupied_bed_codes = sorted(set(occupied_bed_codes) | set(snap_beds))

        # Bed Map override (single source of current occupancy). For beds the
        # bed-map covers, its live status wins; uncovered beds keep reconstruction.
        if bed_map_available:
            apt_u_bm = str(apartment_code).strip().upper()
            covered_set = {
                b for b in bed_universe
                if f"{apt_u_bm}|{str(b).strip().upper()}" in bed_map_available
            }
            if covered_set:
                bm_occupied = [
                    b for b in covered_set
                    if not bed_map_available[f"{apt_u_bm}|{str(b).strip().upper()}"]
                ]
                recon_uncovered = [b for b in occupied_bed_codes if b not in covered_set]
                occupied_bed_codes = sorted(set(bm_occupied) | set(recon_uncovered))

        occupied_beds = len(occupied_bed_codes)

        # Guard against data-entry overbooking (more active beds than capacity).
        if occupied_beds > capacity:
            quality["rooms_overbooked_clamped"] += 1
            capacity = occupied_beds  # keep invariants; flagged above

        available_bed_codes = [b for b in bed_universe if b not in set(occupied_bed_codes)]
        available_beds = max(capacity - occupied_beds, 0)

        current_revenue = float(active_dedup["monthly_rental"].fillna(0).sum())
        current_occupancy_pct = round(100 * occupied_beds / capacity, 2) if capacity else 0.0

        # Current asking rent = the most recent positive rent seen in the room
        # (reflects today's price; used for budget ranking and for the offered bed).
        rent_rows = g[g["monthly_rental"] > 0].sort_values("onboarding_date")
        current_rent = float(rent_rows["monthly_rental"].iloc[-1]) if not rent_rows.empty else None

        # Business-priority + Demand share the recent baseline (2025+).
        # Pre-2025 booking history is kept for capacity/age only — not these %.
        _activation = activation_by_apt.get(str(apartment_code).strip())
        recent_occupancy_pct, recent_revenue = _windowed_metrics(
            g, recent_start, as_of, capacity, activation=_activation
        )
        # Column name retained for ranking/dashboard; value is recent-window occ.
        historical_occupancy_pct, historical_revenue = (
            recent_occupancy_pct,
            recent_revenue,
        )

        # Recent demand indicators (vacancy/refill behaviour, 2025+).
        vacancy_events, refills, fill_rate, avg_vacancy = _recent_demand(
            g, recent_start, as_of
        )

        from roommate_matching import clean_occupant_states

        occupant_states = clean_occupant_states(
            active_dedup["_occ_state"].dropna().tolist()
        )
        occupant_states = sorted(occupant_states)
        # Per-occupant attribute lists (one entry per current occupant that has
        # the attribute) — consumed by the roommate compatibility score.
        occupant_genders = active_dedup["_occ_gender"].dropna().tolist()
        occupant_occupations = active_dedup["_occ_occupation"].dropna().tolist()
        occupant_ages = [round(float(a), 1) for a in active_dedup["_occ_age"].dropna().tolist()]

        master_fields = _room_master_fields(
            apartment_code, bed_universe, master_by_key
        )
        master_fields.pop("_master_beds_matched", None)
        master_fields.pop("_master_beds_total", None)

        # Prefer master current_rate for asking rent when present.
        if master_fields.get("current_rate") is not None:
            current_rent = float(master_fields["current_rate"])

        # bed_type vs derived-capacity cross check (warning only).
        hint = _BED_TYPE_HINT.get(normalize_text(bed_type)) if pd.notna(bed_type) else None
        if hint is not None and hint != len(bed_universe):
            quality["rooms_capacity_lt_bed_type"] += 1

        record = {
            "apartment_code": apartment_code,
            "room_code": room_code,
            "bed_type": bed_type,
            "capacity": capacity,
            "occupied_beds": occupied_beds,
            "available_beds": available_beds,
            "available_bed_codes": available_bed_codes,
            "current_occupancy_pct": current_occupancy_pct,
            "current_revenue": round(current_revenue, 2),
            "current_rent": round(current_rent, 2) if current_rent is not None else None,
            "recent_occupancy_pct": recent_occupancy_pct,
            "recent_revenue": recent_revenue,
            "recent_vacancy_events": vacancy_events,
            "recent_refills": refills,
            "recent_fill_rate": round(fill_rate, 3) if fill_rate is not None else None,
            "average_vacancy_days_recent": round(avg_vacancy, 1) if avg_vacancy is not None else None,
            "historical_occupancy_pct": historical_occupancy_pct,
            "historical_revenue": historical_revenue,
            "occupant_states": occupant_states,
            "occupant_genders": occupant_genders,
            "occupant_occupations": occupant_occupations,
            "occupant_ages": occupant_ages,
            "gender_allowed": master_fields["gender_allowed"],
            "bed_status": master_fields["bed_status"],
            "bed_lifecycle_status": master_fields["bed_lifecycle_status"],
            "toilet_type": master_fields["toilet_type"],
            "current_rate": master_fields["current_rate"],
            "bed_codes": bed_universe,
            "_never_booked": never_booked,
        }
        record.update(_lifecycle_attrs(g))
        # beds_master bed_status / lifecycle take priority over booking attrs.
        if master_fields["bed_status"] is not None:
            record["bed_status"] = master_fields["bed_status"]
        if master_fields["bed_lifecycle_status"] is not None:
            record["bed_lifecycle_status"] = master_fields["bed_lifecycle_status"]
        records.append(record)

    # Match stats from inventory bed universe vs resolved master keys.
    matched_beds = 0
    total_inv_beds = 0
    for rec in records:
        apt = str(rec["apartment_code"]).strip().upper()
        beds = rec.get("bed_codes") or []
        total_inv_beds += len(beds)
        for bed in beds:
            if f"{apt}|{str(bed).strip().upper()}" in master_by_key:
                matched_beds += 1
    unmatched_beds = total_inv_beds - matched_beds

    inventory = pd.DataFrame.from_records(records)
    if not inventory.empty:
        inventory = inventory.sort_values(["apartment_code", "room_code"]).reset_index(
            drop=True
        )
        # Label Active / Inactive / Unknown from beds_master.bed_status etc.
        inventory = annotate_lifecycle_status(inventory)
        # demand_score across the full inventory (Inactive stays labeled).
        # Never-booked seeded rooms are excluded from the normalisation basis so
        # rooms with history keep identical scores; drop the internal marker after.
        inventory["demand_score"] = _compute_demand_score(inventory)
        inventory = inventory.drop(columns=["_never_booked"], errors="ignore")

        inactive_apts = sorted(
            {
                str(a).strip().upper()
                for a in inventory.loc[
                    inventory["lifecycle_status"].astype(str) == "Inactive",
                    "apartment_code",
                ].unique()
            }
        )
        # Compare master gender vs former occupant-inference for change count.
        from roommate_matching import gender_allowed_label

        inferred = inventory["occupant_genders"].map(gender_allowed_label)
        master_g = inventory["gender_allowed"].astype(str).str.strip()
        gender_changed = int((inferred.astype(str).str.strip() != master_g).sum())
        life_changed = int(
            inventory["bed_lifecycle_status"].notna().sum()
            if "bed_lifecycle_status" in inventory.columns
            else 0
        )

        match_stats = {
            "matched_beds": matched_beds,
            "unmatched_beds": unmatched_beds,
            "gender_changed": gender_changed,
            "lifecycle_changed": life_changed,
            "inactive_apartments": inactive_apts,
        }
        print_beds_master_validation(master_report, match_stats)
        inventory.attrs["beds_master_report"] = master_report
        inventory.attrs["beds_master_match_stats"] = match_stats
    else:
        print_beds_master_validation(
            master_report,
            {
                "matched_beds": 0,
                "unmatched_beds": 0,
                "gender_changed": 0,
                "lifecycle_changed": 0,
                "inactive_apartments": [],
            },
        )

    _validate_room_inventory(inventory)
    logger.info(
        "Room Inventory built: %d rooms | dropped %d bookings without a bed | "
        "as_of=%s | data-quality=%s",
        len(inventory),
        dropped_no_bed,
        as_of.date() if pd.notna(as_of) else None,
        quality,
    )
    inventory.attrs["as_of"] = as_of
    inventory.attrs["historical_start"] = historical_start
    inventory.attrs["recent_start"] = recent_start
    inventory.attrs["data_quality"] = quality
    inventory.attrs["dropped_bookings_without_bed"] = dropped_no_bed
    return inventory


def _validate_room_inventory(inv: pd.DataFrame) -> None:
    """Hard invariants the Room Master must satisfy. Raises on violation."""
    if inv.empty:
        logger.warning("Room Inventory is empty.")
        return

    if inv["room_code"].duplicated().any():
        dups = inv.loc[inv["room_code"].duplicated(), "room_code"].tolist()
        raise ValueError(f"Duplicate room records detected: {dups}")

    if (inv["capacity"] < inv["occupied_beds"]).any():
        bad = inv.loc[inv["capacity"] < inv["occupied_beds"], "room_code"].tolist()
        raise ValueError(f"capacity < occupied_beds for rooms: {bad}")

    if (inv["available_beds"] < 0).any():
        bad = inv.loc[inv["available_beds"] < 0, "room_code"].tolist()
        raise ValueError(f"Negative available_beds for rooms: {bad}")

    if (inv["current_occupancy_pct"] > 100).any() or (
        inv["historical_occupancy_pct"] > 100
    ).any() or (inv["recent_occupancy_pct"] > 100).any():
        raise ValueError("Occupancy percentage exceeds 100%.")

    fr = inv["recent_fill_rate"].dropna()
    if ((fr < 0) | (fr > 1)).any():
        raise ValueError("recent_fill_rate outside [0, 1].")

    if "demand_score" in inv.columns:
        ds = inv["demand_score"].dropna()
        if ((ds < 0) | (ds > 100)).any():
            raise ValueError("demand_score outside [0, 100].")

    # Consistency: available_beds should equal capacity - occupied_beds.
    if (inv["available_beds"] != inv["capacity"] - inv["occupied_beds"]).any():
        raise ValueError("available_beds != capacity - occupied_beds.")


def _prepare_spans(
    bookings: pd.DataFrame, as_of: Optional[pd.Timestamp] = None
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Shared cleaning: attribute bookings to beds/rooms and compute span ends.

    Used by both the Room Inventory and the Occupancy History so room identity
    and span semantics stay identical everywhere.
    """
    require_columns(
        bookings,
        ["apartment_code", "bed_code", "bed_type", "onboarding_date",
         "actual_exit_date", "monthly_rental", "staying_status"],
        table="bookings",
    )
    df = bookings.copy()
    for col in ("onboarding_date", "actual_exit_date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["monthly_rental"] = pd.to_numeric(df["monthly_rental"], errors="coerce")

    df = df[df["apartment_code"].notna() & df["bed_code"].notna()].copy()
    df["apartment_code"] = df["apartment_code"].astype("string").str.strip()
    df["bed_code"] = df["bed_code"].astype("string").str.strip()
    df["room_code"] = _resolve_room_code(df)
    df = df[df["room_code"].notna()].copy()

    if as_of is None:
        as_of = pd.concat([df["onboarding_date"], df["actual_exit_date"]]).max()
    as_of = pd.Timestamp(as_of)
    df["_span_end"] = df["actual_exit_date"].fillna(as_of).clip(upper=as_of)
    return df, as_of


def _annotate_transfer_exits(
    out: pd.DataFrame, transfers: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Flag occupancy-history exits that correspond to a completed bed transfer.

    ADDITIVE and NON-BEHAVIORAL: adds ``exit_is_transfer`` (bool) and ``transfer_id``
    only. Does NOT touch stay_days / vacancy_days / refilled / exit_window, so the
    fill-time engine, blocked-room detector and revenue leakage are unaffected. The
    flag documents that a vacancy began as a same-property move (improving timeline
    accuracy) without excluding it from blocked-room detection.
    """
    out = out.copy()
    out["exit_is_transfer"] = False
    out["transfer_id"] = pd.NA
    if (
        transfers is None
        or not isinstance(transfers, pd.DataFrame)
        or transfers.empty
        or "old_apartment_code" not in transfers.columns
    ):
        return out

    completed = transfers
    if "status" in transfers.columns:
        completed = transfers[normalize_series(transfers["status"]) == "completed"]
    if completed.empty:
        return out

    # Prefer switch_date, fall back to effective_date, for the exit-date match.
    move_date = None
    for col in ("switch_date", "effective_date"):
        if col in completed.columns:
            cand = pd.to_datetime(completed[col], errors="coerce")
            move_date = cand if move_date is None else move_date.fillna(cand)
    if move_date is None:
        return out

    events = []  # (apt_upper, bed_upper, move_ts, transfer_id)
    tid = completed["id"] if "id" in completed.columns else pd.Series(pd.NA, index=completed.index)
    for apt, bed, mdate, ident in zip(
        completed["old_apartment_code"], completed["old_bed_code"], move_date, tid
    ):
        if pd.isna(apt) or pd.isna(bed) or pd.isna(mdate):
            continue
        events.append(
            (str(apt).strip().upper(), str(bed).strip().upper(), pd.Timestamp(mdate), ident)
        )
    if not events:
        return out

    apt_u = out["apartment_code"].astype(str).str.strip().str.upper()
    bed_u = out["bed_code"].astype(str).str.strip().str.upper()
    for i in out.index:
        exit_d = out.at[i, "actual_exit_date"]
        if pd.isna(exit_d):
            continue
        a, b = apt_u.at[i], bed_u.at[i]
        for ev_a, ev_b, ev_d, ev_id in events:
            # Same bed, exit within a week of the recorded switch date.
            if a == ev_a and b == ev_b and abs((pd.Timestamp(exit_d) - ev_d).days) <= 7:
                out.at[i, "exit_is_transfer"] = True
                out.at[i, "transfer_id"] = ev_id
                break
    return out


def build_occupancy_history(
    bookings: pd.DataFrame,
    as_of: Optional[pd.Timestamp] = None,
    historical_start: Optional[pd.Timestamp] = None,
    recent_start: Optional[pd.Timestamp] = None,
    transfers: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """PHASE 2 STEP 2 — the canonical per-bed occupancy timeline.

    One row per real tenancy span (onboarding -> exit) per bed, ordered in time,
    enriched with the vacancy gap to the next tenant. This is the single input
    the fill-time engine (Step 3) and blocked-room detector will aggregate.

    Reconstruction rules (same as the demand indicators):
      * keep only rows with an onboarding date (drop Cancelled/Booked);
      * drop same-day "no-show" spans (stay <= 1 day) as data noise;
      * sort spans per (apartment_code, bed_code) by onboarding_date;
      * vacancy_days = next onboarding - this exit, clamped >= 0 (overlaps -> 0).

    Returned columns:
        apartment_code, room_code, bed_code, bed_type, full_name, phone,
        onboarding_date, actual_exit_date, stay_days, is_active,
        next_onboarding_date, vacancy_days, refilled, exit_window, monthly_rental

    ``exit_window`` classifies each completed tenancy's vacancy event by its exit
    date: 'recent' (>= recent_start), 'historical' (>= historical_start), 'pre'
    (older, kept as reference), or 'ongoing' (still occupied).
    """
    historical_start = pd.Timestamp(historical_start or HISTORICAL_BASELINE_START)
    recent_start = pd.Timestamp(recent_start or RECENT_BASELINE_START)

    df, as_of = _prepare_spans(bookings, as_of)

    # Keep real tenancy spans only.
    df = df[df["onboarding_date"].notna()].copy()
    df["stay_days"] = (df["_span_end"] - df["onboarding_date"]).dt.days.clip(lower=0)
    df["is_active"] = df["actual_exit_date"].isna()
    df = df[df["is_active"] | (df["stay_days"] > 1)].copy()

    # Sequence per bed and derive the vacancy gap to the next tenant.
    df = df.sort_values(["apartment_code", "bed_code", "onboarding_date"])
    grp = df.groupby(["apartment_code", "bed_code"], sort=False)
    df["next_onboarding_date"] = grp["onboarding_date"].shift(-1)
    df["vacancy_days"] = (
        (df["next_onboarding_date"] - df["actual_exit_date"]).dt.days.clip(lower=0)
    )
    df["refilled"] = df["next_onboarding_date"].notna() & df["actual_exit_date"].notna()

    def _classify(exit_date) -> str:
        if pd.isna(exit_date):
            return "ongoing"
        if exit_date >= recent_start:
            return "recent"
        if exit_date >= historical_start:
            return "historical"
        return "pre"

    df["exit_window"] = df["actual_exit_date"].map(_classify)

    columns = [
        "apartment_code", "room_code", "bed_code", "bed_type", "full_name",
        "phone", "onboarding_date", "actual_exit_date", "stay_days", "is_active",
        "next_onboarding_date", "vacancy_days", "refilled", "exit_window",
        "monthly_rental",
    ]
    # Carry explicit lifecycle/status columns when the source provides them.
    columns.extend(c for c in ACTIVE_STATUS_COLUMNS if c in df.columns and c not in columns)
    out = df[[c for c in columns if c in df.columns]].reset_index(drop=True)
    # Label Status; keep Inactive visible for operational reference.
    out = annotate_lifecycle_status(out)
    # Optional transfer annotation (additive columns only; default None = unchanged).
    if transfers is not None:
        out = _annotate_transfer_exits(out, transfers)
    out.attrs["as_of"] = as_of
    out.attrs["historical_start"] = historical_start
    out.attrs["recent_start"] = recent_start
    logger.info("Occupancy History built: %d tenancy spans across %d beds.",
                len(out),
                out.groupby(["apartment_code", "bed_code"]).ngroups if not out.empty else 0)
    return out


def build_current_allocation(bookings: pd.DataFrame, tenants: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Phase 2 — later step.")


def build_bed_inventory(bookings: pd.DataFrame) -> pd.DataFrame:
    raise NotImplementedError("Phase 2 — later step.")


# --------------------------------------------------------------------------- #
# Script entry point: load via data_loader, build, and save for inspection.
# --------------------------------------------------------------------------- #


def _stringify_lists(inv: pd.DataFrame) -> pd.DataFrame:
    """Make list columns CSV-friendly for the saved artifact."""
    out = inv.copy()
    for col in ("available_bed_codes", "occupant_states", "occupant_genders",
                "occupant_occupations", "occupant_ages", "bed_codes"):
        if col in out.columns:
            out[col] = out[col].map(lambda xs: "; ".join(map(str, xs)) if isinstance(xs, list) else xs)
    return out


def generate_and_save(output_name: str = "room_inventory.csv") -> pd.DataFrame:
    """Load data through data_loader, build the Room Inventory, save to outputs/."""
    from pathlib import Path

    from data_loader import DataLoader

    loader = DataLoader()
    bookings = loader.bookings()
    tenants = None
    try:
        tenants = loader.tenants()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tenants table unavailable; occupant_states will be empty: %s", exc)

    beds_master = None
    try:
        beds_master = loader.beds_master()
    except Exception as exc:  # noqa: BLE001
        logger.warning("beds_master unavailable; gender/status from master skipped: %s", exc)

    current_occupancy = None
    try:
        current_occupancy = loader.current_occupancy()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "current_occupancy unavailable; using bookings holds only: %s", exc
        )

    transfers = None
    try:
        transfers = loader.transfers()
    except Exception as exc:  # noqa: BLE001
        logger.warning("transfers unavailable; occupancy history not annotated: %s", exc)

    bed_map = None
    try:
        bed_map = loader.bed_map()
    except Exception as exc:  # noqa: BLE001
        logger.warning("bed-map unavailable; occupancy falls back to reconstruction: %s", exc)

    inventory = build_room_inventory(
        bookings,
        tenants=tenants,
        beds_master=beds_master,
        current_occupancy=current_occupancy,
        bed_map=bed_map,
    )
    history = build_occupancy_history(bookings, transfers=transfers)

    outputs_dir = Path(__file__).resolve().parents[1] / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _stringify_lists(inventory).to_csv(outputs_dir / output_name, index=False)
    history.to_csv(outputs_dir / "occupancy_history.csv", index=False)
    logger.info("Saved Room Inventory -> %s", outputs_dir / output_name)
    logger.info("Saved Occupancy History -> %s", outputs_dir / "occupancy_history.csv")
    return inventory


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    inv = generate_and_save()
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print(f"\nTotal rooms: {len(inv)}")
    print(f"as_of: {inv.attrs.get('as_of')}")
    print(f"data-quality: {inv.attrs.get('data_quality')}")
    print(f"historical window: {inv.attrs.get('historical_start')}  |  "
          f"recent window: {inv.attrs.get('recent_start')}")
    cols = [
        "apartment_code", "room_code", "bed_type", "capacity", "occupied_beds",
        "available_beds", "current_occupancy_pct",
        "recent_occupancy_pct", "recent_fill_rate", "average_vacancy_days_recent",
        "recent_revenue", "historical_occupancy_pct", "demand_score",
    ]
    print("\n=== Sample (rooms with an available bed, by demand_score) ===")
    avail = inv[inv["available_beds"] > 0].sort_values("demand_score", ascending=False)
    print(_stringify_lists(avail)[cols].head(12).to_string(index=False))
    print("\n=== Sample (fully occupied rooms) ===")
    full = inv[inv["available_beds"] == 0]
    print(_stringify_lists(full)[cols].head(8).to_string(index=False))
