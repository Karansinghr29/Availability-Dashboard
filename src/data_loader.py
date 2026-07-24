"""
data_loader.py
==============

SINGLE SOURCE OF TRUTH for *where the data comes from*.

This is the ONLY module in the whole project that knows about files, paths,
databases or connection strings. Every other module (preprocessing,
recommendation_engine, analytics, dashboard, ...) receives clean pandas
DataFrames and must never open a file or a DB connection by itself.

Design goals
------------
1. Future-proof: today the data lives in CSV exports. Tomorrow it will live in
   PostgreSQL / Supabase. When that happens, ONLY this file changes. The public
   API (`DataLoader.tenants()`, `.bookings()`, ...) stays identical and returns
   the same DataFrame schema, so downstream code keeps working untouched.

2. Fully dynamic: nothing is hardcoded. Files are NOT matched by their (ugly,
   auto-generated) names such as "Supabase Snippet Untitled query (23).csv".
   Instead each file is classified into a *logical table* by matching its
   column signature. Add a new export, rename a file, or add a column -> the
   loader still finds the right table. When the DB arrives, the same logical
   names map to real SQL tables/views.

3. Clean data out: 'null' strings become real NaN, dates are parsed, numeric
   columns are coerced. When multiple CSVs match one logical table, ONE
   primary Vishful export is used (Q35–Q39); older exports only backfill
   missing columns — rows are never mixed.

Logical tables exposed
----------------------
    tenants               Tenant profiles (identity, home city/state, current bed)
    bookings              Booking history (Vishful Q35 primary; Q31 column fallback)
    maintenance           Maintenance / service tickets
    invoices              Invoice ledger (Vishful Q38)
    payments              Payment receipts (collected revenue)
    electricity_bills     Electricity bill payments per apartment
    electricity_readings  Electricity meter readings per apartment
    beds_master           Bed inventory (Vishful Q37 primary; Q33 column fallback)
    current_occupancy     Live Occupied/Vacant (Vishful Q36 primary; Q34 column fallback)
    billing_snapshot      Per-tenant billing snapshot (Vishful Q39)

Non-core tables are recognised but excluded from the business scope:
    assets                Physical asset inventory (ignored)
    expenses              Expense ledger (optional; not required by Phase-1 scope)
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# String tokens that CSV exports use to represent a missing value. Normalised
# to real NaN on load so downstream code never has to special-case them.
NULL_TOKENS = {"null", "NULL", "None", "none", "NaN", "nan", "", "NA", "N/A"}


@dataclass(frozen=True)
class TableSpec:
    """Describes one logical table and how to recognise / clean it.

    signature       : columns that (together) uniquely identify this table.
    min_match       : fraction of `signature` columns that must be present for a
                      file to be classified as this table.
    primary_key     : column used to de-duplicate when several files map here.
    date_columns    : columns to parse as datetimes.
    numeric_columns : columns to coerce to numeric.
    required        : whether this table is required by the business scope.
    """

    name: str
    signature: frozenset
    primary_key: Optional[str] = None
    date_columns: tuple = ()
    numeric_columns: tuple = ()
    min_match: float = 0.7
    required: bool = True
    description: str = ""


# The registry is the contract between raw data and the rest of the codebase.
# Downstream code only ever refers to these logical names.
# Raw-header renames applied before cleaning so downstream sees stable names.
_COLUMN_ALIASES: Dict[str, Dict[str, str]] = {
    "bookings": {
        "tenant_name": "full_name",
        # Vishful allotment export (query 35) uses exit_date.
        "exit_date": "actual_exit_date",
    },
    "current_occupancy": {
        "tenant_name": "full_name",
        "current_tenant": "full_name",
        "tenant_phone": "phone",
        "move_in_date": "onboarding_date",
    },
    "beds_master": {
        # Production export uses Occupied/Vacant in `occupancy`.
        "occupancy": "bed_lifecycle_status",
    },
}

# When several CSVs classify to the same logical table, pick ONE primary source.
# Prefer Vishful production exports (Q35–Q39). Older Q31/Q33/Q34 are fallback
# only for missing *columns* — never concatenated as extra rows.
_PRIMARY_CSV_MARKERS: Dict[str, frozenset] = {
    # Query (50)/(36): live Occupied/Vacant + current stay (Vishful occupancy).
    # tenant_id is the discriminator that makes the fuller Q50 snapshot win over Q36.
    "current_occupancy": frozenset({"occupancy_status", "current_tenant", "tenant_id"}),
    # Query (35): allotment / booking history (exit_date + tenant_id).
    "bookings": frozenset({"tenant_id", "exit_date"}),
    # Query (37): bed master with Active/Not Active bed_status.
    "beds_master": frozenset({"bed_status", "toilet_type", "current_rate"}),
    # Query (38): invoice ledger.
    "invoices": frozenset({"allotment_id", "organization_id", "rent_amount"}),
}

# Tables that may use an older CSV only to fill columns absent on the primary.
# Bookings (Q35) are complete for the pipeline — do NOT backfill from Q31 by
# apartment+bed (that would stamp one legacy booking_id onto every stay).
_COLUMN_FALLBACK_TABLES = frozenset({"current_occupancy", "beds_master"})


TABLE_REGISTRY: Dict[str, TableSpec] = {
    # Production tenants (query 32): thinner profile; no id / DOB / rating.
    "tenants": TableSpec(
        name="tenants",
        signature=frozenset(
            {
                "full_name",
                "phone",
                "gender",
                "city",
                "state",
                "property_name",
                "apartment_code",
                "bed_code",
                "staying_status",
                "monthly_rental",
            }
        ),
        primary_key=None,
        date_columns=(
            "date_of_birth",
            "onboarding_date",
            "notice_date",
            "actual_exit_date",
            "created_at",
            "date_of_joining",
        ),
        numeric_columns=("age", "tenant_rating", "pincode", "monthly_rental"),
        min_match=0.7,
        description="Tenant profiles: identity, home city/state, current allocation.",
    ),
    # Production bookings: prefer Vishful Q35 (tenant_id + exit_date).
    # Legacy Q31 (booking_id + actual_exit_date) is column-fallback only.
    "bookings": TableSpec(
        name="bookings",
        signature=frozenset(
            {
                "tenant_name",
                "booking_date",
                "apartment_code",
                "bed_code",
                "onboarding_date",
                "notice_date",
                "monthly_rental",
                "staying_status",
            }
        ),
        primary_key=None,  # never silent-dedupe; history is an event log
        date_columns=(
            "booking_date",
            "onboarding_date",
            "estimated_exit_date",
            "actual_exit_date",
            "exit_date",
            "notice_date",
            "expected_payment_date",
            "created_at",
        ),
        numeric_columns=(
            "monthly_rental",
            "deposit_paid",
            "discount",
            "total_due",
            "paid_amount",
            "balance_due",
            "onboarding_charges",
            "prorated_rent",
            "expected_stay_days",
        ),
        min_match=0.7,
        description=(
            "Allocation + booking history (Vishful Q35 primary; Q31 fallback "
            "for missing columns only)."
        ),
    ),
    "maintenance": TableSpec(
        name="maintenance",
        signature=frozenset(
            {
                "ticket_number",
                "issue_type",
                "issue_sub_type",
                "priority",
                "status",
                "sla_deadline",
                "resolved_at",
                "apartment_code",
            }
        ),
        primary_key="ticket_number",
        date_columns=("created_at", "sla_deadline", "resolved_at", "closed_at"),
        required=False,
        description="Maintenance / service tickets raised per apartment/bed.",
    ),
    "invoices": TableSpec(
        name="invoices",
        signature=frozenset(
            {
                "invoice_number",
                "invoice_date",
                "due_date",
                "rent_amount",
                "electricity_amount",
                "total_amount",
                "billing_month",
                "allotment_id",
                "tenant_id",
            }
        ),
        primary_key="id",
        date_columns=("invoice_date", "due_date", "created_at"),
        numeric_columns=(
            "rent_amount",
            "electricity_amount",
            "other_charges",
            "total_amount",
            "late_fee",
            "estimated_eb",
            "amount_paid",
            "balance",
        ),
        required=False,
        description="Invoice ledger = accrued revenue (rent + electricity + charges).",
    ),
    "payments": TableSpec(
        name="payments",
        signature=frozenset(
            {
                "payment_date",
                "payment_mode",
                "amount_paid",
                "receipt_number",
                "tenant_allotment_id",
                "tenant_id",
                "receipt_type",
            }
        ),
        primary_key="id",
        date_columns=("payment_date", "created_at"),
        numeric_columns=("amount_paid", "base_amount", "processing_fee"),
        required=False,
        description="Payment receipts = collected revenue.",
    ),
    "electricity_bills": TableSpec(
        name="electricity_bills",
        signature=frozenset(
            {
                "apartment_code",
                "bill_date",
                "bill_amount",
                "billing_period_start",
                "billing_period_end",
                "reference_number",
            }
        ),
        primary_key=None,
        date_columns=(
            "bill_date",
            "payment_date",
            "billing_period_start",
            "billing_period_end",
        ),
        numeric_columns=("bill_amount",),
        required=False,
        description="Electricity bill payments per apartment.",
    ),
    "electricity_readings": TableSpec(
        name="electricity_readings",
        signature=frozenset(
            {
                "eb_meter_number",
                "reading_date",
                "start_reading",
                "current_reading",
                "units_consumed",
                "eb_amount",
                "is_danger",
            }
        ),
        primary_key=None,
        date_columns=("reading_date",),
        numeric_columns=(
            "start_reading",
            "current_reading",
            "units_consumed",
            "eb_amount",
        ),
        required=False,
        description="Electricity meter readings per apartment.",
    ),
    # Production beds master: prefer Vishful Q37 (bed_status Active/Not Active).
    # Legacy Q33 supplies occupancy Occupied/Vacant as column fallback only.
    "beds_master": TableSpec(
        name="beds_master",
        signature=frozenset(
            {
                "apartment_code",
                "bed_code",
                "bed_type",
                "gender_allowed",
                "toilet_type",
                "current_rate",
                # bed_status keeps the real catalog (Q37) at 1.0 while pushing the
                # full occupancy snapshot (Q50, no bed_status) below current_occupancy
                # so Q50 is classified as occupancy, not a bed master.
                "bed_status",
            }
        ),
        primary_key=None,  # never silent-dedupe; preprocessing validates keys
        date_columns=("onboarding_date", "notice_date", "created_at", "updated_at"),
        numeric_columns=("current_rate", "monthly_rate"),
        min_match=0.85,
        required=False,
        description=(
            "Official bed inventory master (Vishful Q37 primary; Q33 fallback "
            "for missing columns such as occupancy)."
        ),
    ),
    # Production live occupancy: prefer Vishful Q36 (current_tenant).
    # Legacy Q34 is column-fallback only — never mixed as extra bed rows.
    "current_occupancy": TableSpec(
        name="current_occupancy",
        signature=frozenset(
            {
                "apartment_code",
                "bed_code",
                "bed_type",
                "occupancy_status",
                "staying_status",
                "move_in_date",
            }
        ),
        primary_key=None,
        date_columns=(
            "move_in_date",
            "onboarding_date",
            "notice_date",
            "estimated_exit_date",
            "created_at",
        ),
        numeric_columns=("monthly_rental", "balance_due", "current_rate"),
        min_match=0.75,
        required=False,
        description=(
            "Live Occupied/Vacant snapshot (Vishful Q36 primary; Q34 column "
            "fallback only)."
        ),
    ),
    # Vishful billing snapshot (query 39) — recognised for future revenue views.
    "billing_snapshot": TableSpec(
        name="billing_snapshot",
        signature=frozenset(
            {
                "tenant_name",
                "apartment_code",
                "bed_code",
                "monthly_rental",
                "billing_month",
                "total_due",
                "paid_amount",
                "balance_due",
                "payment_status",
            }
        ),
        primary_key=None,
        date_columns=(),
        numeric_columns=(
            "monthly_rental",
            "electricity",
            "total_due",
            "paid_amount",
            "balance_due",
        ),
        min_match=0.75,
        required=False,
        description="Per-tenant billing snapshot (Vishful Q39).",
    ),
    # Vishful full tenant master (query 42): authoritative identity with a stable
    # tenant id. Home city/state are sparse here and are backfilled from `tenants`
    # (Q32) inside DataLoader.tenants(). Kept as its own logical table so the merge
    # is explicit and the CSV->DB swap stays a one-file change.
    "tenant_master": TableSpec(
        name="tenant_master",
        signature=frozenset(
            {
                "id",
                "user_id",
                "email",
                "date_of_birth",
                "kyc_completed",
                "full_name",
                "phone",
                "city",
                "state",
                "age",
                "profession",
            }
        ),
        primary_key="id",
        date_columns=("date_of_birth", "created_at"),
        numeric_columns=("age", "pincode"),
        min_match=0.7,
        required=False,
        description="Full tenant master (Vishful Q42): identity + demographics + tenant id.",
    ),
    # Vishful apartment master (query 45): the apartment_id <-> apartment_code bridge
    # plus apartment-level gender policy. Used to resolve UUID bed rows (Q43) and the
    # transfer history (Q56) back to text apartment codes.
    "apartment_master": TableSpec(
        name="apartment_master",
        signature=frozenset(
            {
                "id",
                "property_id",
                "apartment_code",
                "floor_number",
                "apartment_type",
                "gender_allowed",
                "eb_meter_number",
                "size_sqft",
            }
        ),
        primary_key="id",
        date_columns=("created_at", "start_date", "end_date"),
        numeric_columns=("floor_number", "size_sqft"),
        min_match=0.7,
        required=False,
        description="Apartment master (Vishful Q45): apartment_id <-> apartment_code bridge.",
    ),
    # Vishful bed master keyed by apartment_id (query 43). Carries bed_lifecycle_status
    # and the bed id used by the transfer history. Resolved to text keys via the
    # apartment master. NOT the primary bed catalog (Q37 keeps that role).
    "beds_master_uuid": TableSpec(
        name="beds_master_uuid",
        signature=frozenset(
            {
                "id",
                "organization_id",
                "apartment_id",
                "bed_code",
                "bed_type",
                "toilet_type",
                "status",
                "bed_lifecycle_status",
            }
        ),
        primary_key="id",
        date_columns=("created_at",),
        min_match=0.75,
        required=False,
        description="Bed master keyed by apartment_id (Vishful Q43): bed id + lifecycle.",
    ),
    # Vishful bed/room transfer history (query 56). old_bed_id -> new_bed_id switches;
    # used ONLY to keep the occupancy timeline accurate (avoid duplicate occupancy
    # events). Does not change blocked-room / leakage math.
    "transfers": TableSpec(
        name="transfers",
        signature=frozenset(
            {
                "id",
                "tenant_id",
                "allotment_id",
                "old_bed_id",
                "new_bed_id",
                "switch_type",
                "switch_date",
                "old_apartment_id",
                "new_apartment_id",
            }
        ),
        primary_key="id",
        date_columns=(
            "switch_date",
            "effective_date",
            "created_at",
            "completed_at",
            "cancelled_at",
        ),
        numeric_columns=("old_rent", "new_rent", "rent_difference", "deposit_difference"),
        min_match=0.8,
        required=False,
        description="Bed/room transfer history (Vishful Q56): old_bed_id -> new_bed_id.",
    ),
    # ---- recognised but NOT part of the Phase-1 business scope ----
    "assets": TableSpec(
        name="assets",
        signature=frozenset(
            {
                "asset_code",
                "serial_number",
                "asset_type",
                "warranty_months",
                "warranty_expiry",
                "condition",
            }
        ),
        primary_key="asset_code",
        required=False,
        description="Physical asset inventory. Ignored by the availability system.",
    ),
    "expenses": TableSpec(
        name="expenses",
        signature=frozenset(
            {
                "expense_date",
                "amount",
                "category_id",
                "related_asset_id",
                "ticket_resolution_id",
                "billing_month",
            }
        ),
        primary_key="id",
        date_columns=("expense_date", "created_at"),
        numeric_columns=("amount",),
        required=False,
        description="Expense ledger (maintenance/asset costs). Optional / not in scope.",
    ),
}


def _default_data_dir() -> Path:
    """Resolve the directory that actually holds the source data.

    Search order (first match with recognisable data wins):
      1. AVAILABILITY_DATA_DIR environment variable
      2. <project>/data
      3. the project's parent directory (the exports currently live one level up)
      4. the project directory itself
    """
    project_root = Path(__file__).resolve().parents[1]  # .../Availability_AI
    candidates: List[Path] = []

    env_dir = os.environ.get("AVAILABILITY_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.extend(
        [
            project_root / "data",
            project_root.parent,  # workspace root where the CSVs currently sit
            project_root,
        ]
    )

    for cand in candidates:
        if cand.is_dir() and any(cand.glob("*.csv")):
            return cand
    # Fall back to the conventional data folder even if empty.
    return project_root / "data"


# --------------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------------- #


@dataclass
class DataLoader:
    """Load logical business tables regardless of the underlying source.

    Parameters
    ----------
    source : {"csv"} (future: "postgres", "supabase")
        Where the data comes from. Only this class cares about it.
    data_dir : str | Path, optional
        Directory containing CSV exports. Defaults to an auto-discovered folder.
    connection : optional
        Reserved for the future DB connection object / SQLAlchemy engine.
    """

    source: str = "csv"
    data_dir: Optional[Path] = None
    connection: object = None
    _cache: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)
    _file_map: Dict[str, List[Path]] = field(default_factory=dict, repr=False)
    _maps: Dict[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.source == "csv":
            self.data_dir = Path(self.data_dir) if self.data_dir else _default_data_dir()
            logger.info("DataLoader(csv) using data_dir=%s", self.data_dir)

    # ------------------------------------------------------------------ #
    # Public API — the stable contract used by the rest of the project
    # ------------------------------------------------------------------ #

    def load(self, table: str, refresh: bool = False) -> pd.DataFrame:
        """Return a cleaned DataFrame for a logical table name."""
        if table not in TABLE_REGISTRY:
            raise KeyError(
                f"Unknown logical table '{table}'. "
                f"Known: {sorted(TABLE_REGISTRY)}"
            )
        if refresh or table not in self._cache:
            self._cache[table] = self._load_table(table)
        return self._cache[table]

    # Convenience accessors (stable across the CSV -> DB migration).
    def tenants(self) -> pd.DataFrame:
        """Authoritative tenant table: Q42 identity + Q32 location backfill.

        ``tenant_master`` (Vishful Q42) is the identity source (stable ``tenant_id`` +
        demographics). The legacy ``tenants`` export (Q32) backfills home ``city`` /
        ``state`` and current allocation columns ONLY where Q42 is missing. Falls back
        to whichever source exists, so behaviour is unchanged when only Q32 is present
        (backward compatible). Result is a column superset of the old ``tenants`` frame.
        """
        master = None
        try:
            master = self.load("tenant_master")
        except FileNotFoundError:
            master = None
        legacy = None
        try:
            legacy = self.load("tenants")
        except FileNotFoundError:
            legacy = None

        if master is None or master.empty:
            return legacy if legacy is not None else pd.DataFrame()

        base = master.copy()
        if "tenant_id" not in base.columns and "id" in base.columns:
            base["tenant_id"] = base["id"]
        if legacy is None or legacy.empty:
            return base
        return self._merge_tenant_sources(base, legacy)

    @staticmethod
    def _phone_key(series: pd.Series) -> pd.Series:
        """Reduce phone numbers to their last 10 digits for robust joining."""

        def clean(value):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            text = str(value).strip()
            if not text:
                return None
            if "e" in text.lower():  # Excel scientific notation
                try:
                    text = str(int(float(text)))
                except (ValueError, OverflowError):
                    pass
            digits = "".join(ch for ch in text if ch.isdigit())
            if not digits:
                return None
            return digits[-10:] if len(digits) >= 10 else digits

        return series.map(clean)

    def _merge_tenant_sources(self, base: pd.DataFrame, legacy: pd.DataFrame) -> pd.DataFrame:
        """Fill Q42 nulls with Q32 location/allocation values, joined by phone."""
        if "phone" not in base.columns or "phone" not in legacy.columns:
            return base  # cannot join safely; return identity-only master

        backfill_cols = [
            "city", "state", "property_name", "apartment_code", "bed_code",
            "monthly_rental", "onboarding_date", "notice_date", "actual_exit_date",
        ]
        cols_present = [c for c in backfill_cols if c in legacy.columns]
        if not cols_present:
            return base

        legacy_keyed = legacy.copy()
        legacy_keyed["_pk"] = self._phone_key(legacy_keyed["phone"])
        legacy_keyed = legacy_keyed[legacy_keyed["_pk"].notna()]
        legacy_keyed = legacy_keyed.drop_duplicates("_pk", keep="last").set_index("_pk")

        out = base.copy()
        out["_pk"] = self._phone_key(out["phone"])
        for col in cols_present:
            src = out["_pk"].map(legacy_keyed[col])
            if col in out.columns:
                out[col] = out[col].where(out[col].notna(), src)
            else:
                out[col] = src
        return out.drop(columns=["_pk"], errors="ignore")

    def bookings(self) -> pd.DataFrame:
        """Return bookings with production-schema gaps filled for the pipeline.

        Primary source: Vishful Q35 (``exit_date`` → ``actual_exit_date``).
        Legacy Q31 is used only to backfill columns absent on Q35 — never as
        extra history rows. ``tenant_name`` is aliased to ``full_name``.
        ``bed_type`` is attached from beds_master / current_occupancy when needed.
        """
        df = self.load("bookings")
        return self._ensure_bookings_pipeline_schema(df)

    def maintenance(self) -> pd.DataFrame:
        return self.load("maintenance")

    def invoices(self) -> pd.DataFrame:
        return self.load("invoices")

    def payments(self) -> pd.DataFrame:
        return self.load("payments")

    def electricity_bills(self) -> pd.DataFrame:
        return self.load("electricity_bills")

    def electricity_readings(self) -> pd.DataFrame:
        return self.load("electricity_readings")

    def beds_master(self) -> pd.DataFrame:
        """Official bed master (gender policy, status, toilet, rate).

        Primary source: Vishful Q37 (``bed_status`` Active / Not Active).
        Legacy Q33 backfills missing columns only (e.g. Occupied/Vacant
        ``occupancy`` → ``bed_lifecycle_status``). If ``bed_status`` is still
        missing after that, Not-Active values may be filled from Backup_old
        query (8).
        """
        return self._apply_bed_status_fallback(self.load("beds_master"))

    def current_occupancy(self) -> pd.DataFrame:
        """Live bed occupancy snapshot (recommendation occupancy source).

        Primary source: Vishful Q36 (``occupancy_status`` + ``current_tenant``).
        Legacy Q34 backfills missing columns only — occupancy *rows* always
        come from Q36 so AI matches the Vishful application.
        """
        return self.load("current_occupancy")

    def billing_snapshot(self) -> pd.DataFrame:
        """Per-tenant billing snapshot (Vishful Q39), when present."""
        return self.load("billing_snapshot")

    def bed_map(self) -> pd.DataFrame:
        """THE single current-occupancy source: live per-bed status from Vishful.

        Reads the newest ``bed-map-*.csv`` export (production bed roster). One row
        per bed with the app's live ``Status``. This is the authoritative CURRENT
        occupancy layer — every page's occupancy is derived from it (via
        ``build_room_inventory`` / ``detect_blocked_rooms``); historical analytics
        stay on Q35/Q42/Q56/revenue.

        Columns
        -------
        apartment_code, bed_code, bed_status (raw: vacant/occupied/booked/notice/…),
        is_available (True only when status == 'vacant'), days_vacant (numeric or NaN),
        current_tenant, bed_rate (numeric or NaN).

        Returns an empty stable-schema frame when no bed-map file exists, so callers
        fall back to the Q35∪Q50 reconstruction (backward compatible).
        """
        cols = [
            "apartment_code", "bed_code", "bed_status", "is_available",
            "days_vacant", "current_tenant", "bed_rate",
        ]
        cache_key = "_bed_map"
        if cache_key in self._cache:
            return self._cache[cache_key]

        data_dir = Path(self.data_dir) if self.data_dir else _default_data_dir()
        files = sorted(data_dir.glob("bed-map-*.csv"))
        if not files:
            logger.info("No bed-map-*.csv found under %s; occupancy falls back to reconstruction.", data_dir)
            empty = pd.DataFrame(columns=cols)
            self._cache[cache_key] = empty
            return empty
        # Newest export wins (date in filename, then mtime).
        fp = max(files, key=lambda p: (p.name, p.stat().st_mtime))

        raw = pd.read_csv(fp, dtype=str, keep_default_na=False)
        raw.columns = [c.strip() for c in raw.columns]

        def _get(row, *names) -> str:
            for n in names:
                if n in row:
                    v = str(row[n]).strip()
                    if v and v not in NULL_TOKENS:
                        return v
            return ""

        rows = []
        for _, r in raw.iterrows():
            apt = _get(r, "Apartment", "apartment_code")
            bed = _get(r, "Bed", "bed_code")
            if not apt or not bed:
                continue
            status = _get(r, "Status", "bed_status").lower()
            dv = _get(r, "Days Vacant", "days_vacant")
            rate = _get(r, "Rate", "current_rate", "bed_rate")
            rows.append(
                {
                    "apartment_code": apt,
                    "bed_code": bed,
                    "bed_status": status or pd.NA,
                    "is_available": status == "vacant",
                    "days_vacant": pd.to_numeric(dv, errors="coerce"),
                    "current_tenant": _get(r, "Tenant", "current_tenant") or pd.NA,
                    "bed_rate": pd.to_numeric(rate, errors="coerce"),
                }
            )
        out = pd.DataFrame(rows, columns=cols)
        logger.info("Loaded bed-map source %s (%d beds; %d vacant).",
                    fp.name, len(out), int(out["is_available"].sum()) if not out.empty else 0)
        self._cache[cache_key] = out
        return out

    def tenant_master(self) -> pd.DataFrame:
        """Full tenant master (Vishful Q42): identity + demographics + tenant id."""
        return self.load("tenant_master")

    def apartment_master(self) -> pd.DataFrame:
        """Apartment master (Vishful Q45): apartment_id <-> apartment_code bridge."""
        return self.load("apartment_master")

    def beds_master_uuid(self) -> pd.DataFrame:
        """Bed master keyed by apartment_id (Vishful Q43): bed id + lifecycle."""
        return self.load("beds_master_uuid")

    def _bed_id_to_text_key(self) -> Dict[str, tuple]:
        """Map bed UUID -> (apartment_code, bed_code) via Q43 beds + Q45 apartments.

        Cached. Empty when the bridge exports are absent (transfers then unresolved).
        """
        if "bed_id_text" in self._maps:
            return self._maps["bed_id_text"]

        out: Dict[str, tuple] = {}
        try:
            beds = self.load("beds_master_uuid")
        except FileNotFoundError:
            beds = None
        try:
            apts = self.load("apartment_master")
        except FileNotFoundError:
            apts = None

        if beds is None or beds.empty:
            self._maps["bed_id_text"] = out
            return out

        apt_map: Dict[str, str] = {}
        if (
            apts is not None
            and not apts.empty
            and "id" in apts.columns
            and "apartment_code" in apts.columns
        ):
            for aid, code in zip(apts["id"], apts["apartment_code"]):
                if pd.notna(aid) and pd.notna(code):
                    apt_map[str(aid).strip()] = str(code).strip()

        for bid, aid, bcode in zip(
            beds.get("id", []),
            beds.get("apartment_id", []),
            beds.get("bed_code", []),
        ):
            if bid is None or (isinstance(bid, float) and pd.isna(bid)):
                continue
            apt_code = apt_map.get(str(aid).strip()) if pd.notna(aid) else None
            bed_code = str(bcode).strip() if pd.notna(bcode) else None
            out[str(bid).strip()] = (apt_code, bed_code)

        self._maps["bed_id_text"] = out
        return out

    def transfers(self) -> pd.DataFrame:
        """Bed/room transfer history (Vishful Q56) resolved to text keys.

        Adds ``old_apartment_code`` / ``old_bed_code`` / ``new_apartment_code`` /
        ``new_bed_code`` via the Q43+Q45 bridge. Returns an empty (stable-schema) frame
        when the transfer/bridge exports are absent, so callers treat transfers as
        optional and behaviour is unchanged without them.
        """
        stable_cols = [
            "old_apartment_code", "old_bed_code", "new_apartment_code", "new_bed_code",
        ]
        try:
            tr = self.load("transfers")
        except FileNotFoundError:
            return pd.DataFrame(columns=stable_cols)
        if tr is None or tr.empty:
            return pd.DataFrame(columns=stable_cols)

        bed_map = self._bed_id_to_text_key()

        def _lookup(bid, idx):
            if bid is None or (isinstance(bid, float) and pd.isna(bid)):
                return None
            pair = bed_map.get(str(bid).strip())
            return pair[idx] if pair else None

        out = tr.copy()
        old_ids = out["old_bed_id"] if "old_bed_id" in out.columns else pd.Series(index=out.index, dtype=object)
        new_ids = out["new_bed_id"] if "new_bed_id" in out.columns else pd.Series(index=out.index, dtype=object)
        out["old_apartment_code"] = old_ids.map(lambda b: _lookup(b, 0))
        out["old_bed_code"] = old_ids.map(lambda b: _lookup(b, 1))
        out["new_apartment_code"] = new_ids.map(lambda b: _lookup(b, 0))
        out["new_bed_code"] = new_ids.map(lambda b: _lookup(b, 1))
        return out

    def _bed_status_fallback_dirs(self) -> List[Path]:
        """Directories that may hold the legacy bed_status snapshot (query 8)."""
        root = Path(self.data_dir) if self.data_dir else _default_data_dir()
        project_root = Path(__file__).resolve().parents[1]
        candidates = [
            root / "Backup_old",
            root / "Backup_Old",
            root.parent / "Backup_old",
            root.parent / "Backup_Old",
            project_root.parent / "Backup_old",
            project_root.parent / "Backup_Old",
        ]
        seen: set = set()
        out: List[Path] = []
        for path in candidates:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            if path.is_dir():
                out.append(path)
        return out

    def _find_bed_status_fallback_file(self) -> Optional[Path]:
        """Locate the legacy beds snapshot that carries ``bed_status``."""
        required = {"apartment_code", "bed_code", "bed_status"}
        for folder in self._bed_status_fallback_dirs():
            for fp in sorted(folder.glob("*.csv")):
                try:
                    header = set(pd.read_csv(fp, nrows=0).columns)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Could not read bed_status fallback header %s: %s",
                        fp.name,
                        exc,
                    )
                    continue
                if required.issubset(header):
                    return fp
        return None

    def _load_bed_status_fallback(self) -> pd.DataFrame:
        """Load apartment_code + bed_code + bed_status only from legacy query (8)."""
        cache_key = "_bed_status_fallback"
        if cache_key in self._cache:
            return self._cache[cache_key]

        empty = pd.DataFrame(columns=["apartment_code", "bed_code", "bed_status"])
        fp = self._find_bed_status_fallback_file()
        if fp is None:
            logger.info(
                "No legacy bed_status fallback CSV found under Backup_old; "
                "production beds_master bed_status left as-is."
            )
            self._cache[cache_key] = empty
            return empty

        raw = pd.read_csv(fp, dtype=str, keep_default_na=False)
        raw.columns = [c.strip() for c in raw.columns]
        for col in raw.columns:
            if raw[col].dtype == object:
                raw[col] = raw[col].map(
                    lambda v: pd.NA
                    if (isinstance(v, str) and v.strip() in NULL_TOKENS)
                    else (v.strip() if isinstance(v, str) else v)
                )
        keep = raw[["apartment_code", "bed_code", "bed_status"]].copy()
        keep["apartment_code"] = keep["apartment_code"].astype("string").str.strip()
        keep["bed_code"] = keep["bed_code"].astype("string").str.strip()
        keep = keep[
            keep["apartment_code"].notna()
            & keep["bed_code"].notna()
            & keep["bed_status"].notna()
        ].copy()
        # One status per bed; last row wins if duplicates.
        keep = keep.drop_duplicates(
            subset=["apartment_code", "bed_code"], keep="last"
        ).reset_index(drop=True)
        logger.info(
            "Loaded bed_status fallback from %s (%d beds).",
            fp.name,
            len(keep),
        )
        self._cache[cache_key] = keep
        return keep

    @staticmethod
    def _is_missing_bed_status(value) -> bool:
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return True
        try:
            if pd.isna(value):
                return True
        except (TypeError, ValueError):
            pass
        text = str(value).strip()
        return text == "" or text.lower() in {"nan", "<na>", "none", "null", "na", "n/a"}

    @staticmethod
    def _is_not_active_bed_status(value) -> bool:
        """True only for inactive tokens (Not-Active / Inactive / …)."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return False
        n = " ".join(str(value).strip().lower().split()).replace(" ", "-")
        if not n:
            return False
        return n in {
            "not-active",
            "inactive",
            "disabled",
            "blocked",
        } or n.startswith("not-active") or n.startswith("inactive")

    def _apply_bed_status_fallback(self, master: pd.DataFrame) -> pd.DataFrame:
        """Fill missing production bed_status with Not-Active from legacy query (8).

        Production gender / rent / occupancy / toilet are never overwritten.
        Live (and other active) values from the legacy file are ignored — only
        Not-Active restores Inactive until production exports bed_status again.
        """
        if master is None or not isinstance(master, pd.DataFrame) or master.empty:
            return master

        fallback = self._load_bed_status_fallback()
        if fallback.empty:
            return master

        lookup: Dict[tuple, str] = {}
        for _, row in fallback.iterrows():
            if not self._is_not_active_bed_status(row["bed_status"]):
                continue
            apt = str(row["apartment_code"] or "").strip().upper()
            bed = str(row["bed_code"] or "").strip().upper()
            if apt and bed:
                lookup[(apt, bed)] = "Not-Active"
        if not lookup:
            return master

        df = master.copy()
        if "bed_status" not in df.columns:
            df["bed_status"] = pd.NA

        filled = 0
        new_status = []
        for apt, bed, status in zip(
            df.get("apartment_code", []),
            df.get("bed_code", []),
            df["bed_status"],
        ):
            if not self._is_missing_bed_status(status):
                new_status.append(status)
                continue
            key = (
                str(apt or "").strip().upper(),
                str(bed or "").strip().upper(),
            )
            fb = lookup.get(key)
            if fb is not None:
                new_status.append(fb)
                filled += 1
            else:
                new_status.append(status)
        df["bed_status"] = new_status
        if filled:
            logger.info(
                "Applied Not-Active bed_status fallback to %d production bed rows.",
                filled,
            )
        return df

    def _ensure_bookings_pipeline_schema(self, bookings: pd.DataFrame) -> pd.DataFrame:
        """Fill columns the rest of the pipeline still requires."""
        df = bookings.copy()
        if "full_name" not in df.columns and "tenant_name" in df.columns:
            df["full_name"] = df["tenant_name"]
        if "actual_exit_date" not in df.columns:
            df["actual_exit_date"] = pd.NaT

        needs_bed_type = (
            "bed_type" not in df.columns
            or df["bed_type"].isna().all()
            or (
                df["bed_type"].astype(str).str.strip().isin({"", "nan", "<NA>", "None"}).all()
            )
        )
        if needs_bed_type:
            df = self._attach_bed_type(df)
        return df

    def _attach_bed_type(self, bookings: pd.DataFrame) -> pd.DataFrame:
        """Join bed_type from beds_master / current_occupancy by apartment+bed."""
        df = bookings.copy()
        lookup: Dict[tuple, str] = {}
        for table in ("beds_master", "current_occupancy"):
            try:
                src = self.load(table)
            except FileNotFoundError:
                continue
            if src is None or src.empty or "bed_type" not in src.columns:
                continue
            for _, row in src.iterrows():
                apt = str(row.get("apartment_code", "") or "").strip().upper()
                bed = str(row.get("bed_code", "") or "").strip().upper()
                bt = str(row.get("bed_type", "") or "").strip()
                if not apt or not bed or not bt or bt.lower() in {"nan", "<na>", "none"}:
                    continue
                key = (apt, bed)
                if key not in lookup:
                    lookup[key] = bt

        def _lookup_bt(apt: object, bed: object) -> str:
            key = (
                str(apt or "").strip().upper(),
                str(bed or "").strip().upper(),
            )
            return lookup.get(key, "Unknown")

        df["bed_type"] = [
            _lookup_bt(a, b)
            for a, b in zip(df.get("apartment_code", []), df.get("bed_code", []))
        ]
        return df

    def load_all(self, required_only: bool = True) -> Dict[str, pd.DataFrame]:
        """Load every (required) logical table into a dict of DataFrames."""
        names = [
            n for n, s in TABLE_REGISTRY.items() if (s.required or not required_only)
        ]
        out: Dict[str, pd.DataFrame] = {}
        for name in names:
            try:
                out[name] = self.load(name)
            except FileNotFoundError:
                logger.warning("No source found for logical table '%s'.", name)
        return out

    # ------------------------------------------------------------------ #
    # Source-specific loading. FUTURE DB WORK GOES HERE AND NOWHERE ELSE.
    # ------------------------------------------------------------------ #

    def _load_table(self, table: str) -> pd.DataFrame:
        if self.source == "csv":
            return self._load_table_csv(table)
        if self.source in {"postgres", "supabase"}:
            return self._load_table_db(table)
        raise ValueError(f"Unsupported source '{self.source}'.")

    def _load_table_db(self, table: str) -> pd.DataFrame:
        """Future integration point for PostgreSQL / Supabase.

        Implementation plan (Phase 2):
            query = f"SELECT * FROM {LOGICAL_TO_SQL[table]}"
            df = pd.read_sql(query, self.connection)
            return self._clean(df, TABLE_REGISTRY[table])

        The returned DataFrame MUST match the CSV schema so that downstream
        modules require zero changes.
        """
        raise NotImplementedError(
            "Database source not implemented yet. Phase 2 will implement this "
            "method only; the public API and returned schemas stay identical."
        )

    # ------------------------------------------------------------------ #
    # CSV specifics
    # ------------------------------------------------------------------ #

    def _discover(self) -> Dict[str, List[Path]]:
        """Classify every CSV in `data_dir` into logical tables by signature."""
        if self._file_map:
            return self._file_map

        files = sorted(self.data_dir.glob("*.csv"))
        mapping: Dict[str, List[Path]] = {}
        for fp in files:
            try:
                header = pd.read_csv(fp, nrows=0)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not read header of %s: %s", fp.name, exc)
                continue
            cols = set(header.columns)
            best_name, best_score = None, 0.0
            for name, spec in TABLE_REGISTRY.items():
                overlap = len(spec.signature & cols) / len(spec.signature)
                if overlap > best_score:
                    best_name, best_score = name, overlap
            if best_name and best_score >= TABLE_REGISTRY[best_name].min_match:
                mapping.setdefault(best_name, []).append(fp)
                logger.info("Classified %s -> %s (%.0f%%)", fp.name, best_name, best_score * 100)
            else:
                logger.info("Unclassified file skipped: %s", fp.name)

        self._file_map = mapping
        return mapping

    @staticmethod
    def _csv_header_columns(path: Path) -> set:
        try:
            return set(pd.read_csv(path, nrows=0).columns)
        except Exception:  # noqa: BLE001
            return set()

    def _select_primary_csv_files(self, table: str, files: List[Path]) -> List[Path]:
        """Choose ONE primary CSV; optional secondary for missing-column backfill.

        Vishful production exports (Q35–Q39) win via ``_PRIMARY_CSV_MARKERS``.
        Older snapshots are never concatenated as extra rows — they may only
        fill columns that the primary file lacks.
        """
        if len(files) <= 1:
            return list(files)

        markers = _PRIMARY_CSV_MARKERS.get(table)
        if not markers:
            newest = max(files, key=lambda p: p.stat().st_mtime)
            logger.info(
                "Multiple CSVs for '%s'; using newest primary %s",
                table,
                newest.name,
            )
            return [newest]

        scored = []
        for fp in files:
            cols = self._csv_header_columns(fp)
            scored.append((len(markers & cols), fp.stat().st_mtime, fp))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        best_score, _, primary = scored[0]
        if best_score == 0:
            primary = max(files, key=lambda p: p.stat().st_mtime)
            logger.info(
                "Multiple CSVs for '%s'; no marker match; using newest %s",
                table,
                primary.name,
            )
            return [primary]

        selected = [primary]
        logger.info(
            "Multiple CSVs for '%s'; primary=%s (markers %d/%d)",
            table,
            primary.name,
            best_score,
            len(markers),
        )

        if table in _COLUMN_FALLBACK_TABLES:
            for _score, _, fp in scored[1:]:
                if fp.resolve() == primary.resolve():
                    continue
                selected.append(fp)
                logger.info(
                    "%s secondary (column backfill only): %s",
                    table,
                    fp.name,
                )
                break
        return selected

    def _backfill_missing_columns(
        self,
        table: str,
        primary: pd.DataFrame,
        secondary: pd.DataFrame,
    ) -> pd.DataFrame:
        """Left-join missing columns from secondary onto primary rows only.

        Never appends secondary rows (no mixed occupancy / history records).
        """
        if secondary is None or secondary.empty or primary.empty:
            return primary
        key_cols = ["apartment_code", "bed_code"]
        if any(c not in primary.columns or c not in secondary.columns for c in key_cols):
            return primary
        missing = [c for c in secondary.columns if c not in primary.columns]
        if not missing:
            return primary

        left = primary.copy()
        right = secondary[key_cols + missing].copy()
        left["_k"] = (
            left["apartment_code"].astype(str).str.strip().str.upper()
            + "|"
            + left["bed_code"].astype(str).str.strip().str.upper()
        )
        right["_k"] = (
            right["apartment_code"].astype(str).str.strip().str.upper()
            + "|"
            + right["bed_code"].astype(str).str.strip().str.upper()
        )
        add = right[["_k"] + missing].drop_duplicates("_k", keep="last")
        out = left.merge(add, on="_k", how="left").drop(columns=["_k"])
        logger.info(
            "Backfilled %s columns from legacy CSV (rows unchanged=%d): %s",
            table,
            len(out),
            missing,
        )
        return out

    def _load_table_csv(self, table: str) -> pd.DataFrame:
        spec = TABLE_REGISTRY[table]
        files = self._discover().get(table, [])
        if not files:
            raise FileNotFoundError(
                f"No CSV file in {self.data_dir} matches logical table '{table}'."
            )
        selected = self._select_primary_csv_files(table, files)
        aliases = _COLUMN_ALIASES.get(table, {})

        def _read(path: Path) -> pd.DataFrame:
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            if aliases:
                frame = frame.rename(
                    columns={k: v for k, v in aliases.items() if k in frame.columns}
                )
            return frame

        primary = _read(selected[0])
        if len(selected) > 1 and table in _COLUMN_FALLBACK_TABLES:
            secondary = _read(selected[1])
            primary = self._backfill_missing_columns(table, primary, secondary)
        elif len(selected) > 1:
            # Should not happen for marked tables; never concat unlike exports.
            logger.warning(
                "Ignoring extra CSVs for '%s' beyond primary %s",
                table,
                selected[0].name,
            )

        return self._clean(primary, spec)

    # ------------------------------------------------------------------ #
    # Cleaning (shared by every source)
    # ------------------------------------------------------------------ #

    @staticmethod
    def _clean(df: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
        df = df.copy()
        df.columns = [c.strip() for c in df.columns]

        # 1. Normalise null-like tokens -> real NaN, trim strings.
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].map(
                    lambda v: pd.NA
                    if (isinstance(v, str) and v.strip() in NULL_TOKENS)
                    else (v.strip() if isinstance(v, str) else v)
                )

        # 2. Type coercion.
        for col in spec.date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce", utc=False)
        for col in spec.numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # 3. De-duplicate on primary key when there is one.
        if spec.primary_key and spec.primary_key in df.columns:
            df = df.drop_duplicates(subset=[spec.primary_key], keep="last")

        return df.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Introspection helper used by the Phase-1 verification report.
    # ------------------------------------------------------------------ #

    def profile(self) -> pd.DataFrame:
        """Return a small summary (table, files, rows, columns) for validation."""
        rows = []
        mapping = self._discover() if self.source == "csv" else {}
        for name, spec in TABLE_REGISTRY.items():
            try:
                df = self.load(name)
                rows.append(
                    {
                        "logical_table": name,
                        "required": spec.required,
                        "source_files": len(mapping.get(name, [])),
                        "rows": len(df),
                        "columns": df.shape[1],
                    }
                )
            except FileNotFoundError:
                rows.append(
                    {
                        "logical_table": name,
                        "required": spec.required,
                        "source_files": 0,
                        "rows": 0,
                        "columns": 0,
                    }
                )
        return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    loader = DataLoader()
    print(f"\nData directory: {loader.data_dir}\n")
    print("=== Discovered logical tables ===")
    print(loader.profile().to_string(index=False))
