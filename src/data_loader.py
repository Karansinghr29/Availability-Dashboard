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
        # Q74 catalog uses `status` (Live / Not-Active) instead of bed_status.
        "status": "bed_status",
    },
}

# When several CSVs classify to the same logical table, pick ONE primary source.
# Prefer current Vishful production exports (Q73/Q74/Q68/…). Older snapshots are
# fallback only for missing *columns* — never concatenated as extra rows.
_PRIMARY_CSV_MARKERS: Dict[str, frozenset] = {
    # Q68 (and Q50): live Occupied/Vacant + current stay.
    "current_occupancy": frozenset({"occupancy_status", "current_tenant", "tenant_id"}),
    # Q73 bookings: text keys + actual_exit_date (legacy Q35 used exit_date).
    "bookings": frozenset({"tenant_id", "actual_exit_date", "apartment_code", "bed_type"}),
    # Q74 beds catalog: rate + from_date (legacy Q37 used bed_status).
    "beds_master": frozenset({"current_rate", "toilet_type", "from_date", "apartment_code"}),
    # Q76 tenant allocation backfill (merged into tenant_master by phone).
    "tenants": frozenset({"tenant_id", "allotment_created_at", "apartment_code"}),
    # Q67 UUID bed bridge (floor_number distinguishes from Q74 rate catalog).
    "beds_master_uuid": frozenset({"apartment_id", "bed_lifecycle_status", "id", "floor_number"}),
    # Q71 apartment master: prefer pure apartment rows (no bed_code mega-joins).
    "apartment_master": frozenset({"eb_meter_number", "apartment_code", "floor_number", "size_sqft"}),
    # Query (38): invoice ledger.
    "invoices": frozenset({"allotment_id", "organization_id", "rent_amount"}),
}

# Tables that may use an older CSV only to fill columns absent on the primary.
# Bookings (Q35) are complete for the pipeline — do NOT backfill from Q31 by
# apartment+bed (that would stamp one legacy booking_id onto every stay).
_COLUMN_FALLBACK_TABLES = frozenset({"current_occupancy", "beds_master"})

# Friendly labels for the "<label> -> <file>" selection log (business names).
_LOG_LABEL: Dict[str, str] = {
    "allotments": "Bookings",
    "beds_master_uuid": "Beds",
    "apartment_master": "Apartments",
    "property_master": "Properties",
    "tenant_master": "Tenants",
    "bed_rates": "Bed rates",
    "asset_master": "Assets",
    "maintenance_tickets": "Maintenance tickets",
}


TABLE_REGISTRY: Dict[str, TableSpec] = {
    # Production tenants allocation backfill (Q76 primary; legacy Q32 fallback).
    # Demographics (city/gender/state) live on tenant_master (Q66) and are merged
    # by phone inside DataLoader.tenants().
    "tenants": TableSpec(
        name="tenants",
        signature=frozenset(
            {
                "full_name",
                "phone",
                "property_name",
                "apartment_code",
                "bed_code",
                "monthly_rental",
                "onboarding_date",
                "notice_date",
                "actual_exit_date",
            }
        ),
        primary_key=None,
        date_columns=(
            "date_of_birth",
            "onboarding_date",
            "notice_date",
            "actual_exit_date",
            "created_at",
            "allotment_created_at",
            "date_of_joining",
        ),
        numeric_columns=("age", "tenant_rating", "pincode", "monthly_rental"),
        min_match=0.7,
        description="Tenant allocation backfill (Q76); city/state come from tenant_master.",
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
    # Production beds master: prefer Vishful Q74 (status Live/Not-Active + rates).
    # Legacy Q37/Q33 supply column fallback only when Q74 lacks a field.
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
                # Q37 used bed_status; Q74 uses status (aliased to bed_status on load).
                # Scoring treats either token as a match in _discover.
                "bed_status",
            }
        ),
        primary_key=None,  # never silent-dedupe; preprocessing validates keys
        date_columns=(
            "onboarding_date",
            "notice_date",
            "created_at",
            "updated_at",
            "from_date",
            "to_date",
        ),
        numeric_columns=("current_rate", "monthly_rate"),
        min_match=0.85,
        required=False,
        description=(
            "Official bed inventory master (Vishful Q74 primary; Q37/Q33 column "
            "fallback only)."
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
    # ================================================================= #
    # Normalized DB export (Q81-Q86). These UUID-keyed relational tables
    # are JOINED inside DataLoader to reconstruct the flattened logical
    # tables (bookings / beds_master / current_occupancy). Downstream code
    # never sees them directly — it keeps consuming the same accessors.
    # ================================================================= #
    # Q81 allotments: the booking/stay event log keyed by UUIDs. Replaces the
    # flattened bookings export (legacy Q35/Q73). No text codes / names here —
    # apartment_code, bed_code, bed_type and tenant name are joined in.
    "allotments": TableSpec(
        name="allotments",
        signature=frozenset(
            {
                "tenant_id",
                "apartment_id",
                "bed_id",
                "booking_date",
                "onboarding_date",
                "estimated_exit_date",
                "actual_exit_date",
                "staying_status",
                "monthly_rental",
                "deposit_paid",
            }
        ),
        primary_key="id",  # allotment id is unique; safe to dedupe defensively
        date_columns=(
            "booking_date",
            "onboarding_date",
            "estimated_exit_date",
            "actual_exit_date",
            "notice_date",
            "expected_payment_date",
            "created_at",
        ),
        numeric_columns=(
            "monthly_rental",
            "deposit_paid",
            "discount",
            "onboarding_charges",
            "prorated_rent",
            "total_due",
            "paid_amount",
            "balance_due",
            "processing_fee",
            "expected_stay_days",
        ),
        min_match=0.8,
        required=False,
        description="Allotments (Vishful Q81): UUID booking/stay event log; source of reconstructed bookings.",
    ),
    # Q84 effective-dated bed rate card, keyed by (property_id, bed_type, toilet_type)
    # with from_date/to_date windows. Supplies current_rate for reconstructed beds_master.
    "bed_rates": TableSpec(
        name="bed_rates",
        signature=frozenset(
            {
                "id",
                "property_id",
                "bed_type",
                "toilet_type",
                "monthly_rate",
                "from_date",
                "to_date",
            }
        ),
        primary_key="id",
        date_columns=("from_date", "to_date", "created_at"),
        numeric_columns=("monthly_rate",),
        min_match=0.85,
        required=False,
        description="Effective-dated bed rate card (Vishful Q84): active rate per bed_type+toilet_type.",
    ),
    # Q86 property master (one row per property). Supplies property_name for the
    # reconstructed bookings frame.
    "property_master": TableSpec(
        name="property_master",
        signature=frozenset(
            {
                "id",
                "property_name",
                "address",
                "city",
                "state",
                "code",
            }
        ),
        primary_key="id",
        date_columns=("created_at", "start_date"),
        min_match=0.8,
        required=False,
        description="Property master (Vishful Q86): property_id -> property_name/address.",
    ),
    # ================================================================= #
    # Maintenance / Asset / Ticket / Vendor domain (new Supabase export).
    # Signature-based like everything else — NO filenames hardcoded. These
    # feed the maintenance data model (src/maintenance.py); the existing
    # availability pages never touch them, so backward compatibility holds.
    # ================================================================= #
    # Physical asset master (purchase/warranty/condition + type + supplier).
    "asset_master": TableSpec(
        name="asset_master",
        signature=frozenset(
            {
                "asset_type_id", "asset_code", "serial_number", "purchase_date",
                "purchase_price", "supplier_id", "warranty_expiry", "condition", "status",
            }
        ),
        primary_key="id",
        date_columns=("purchase_date", "warranty_expiry", "invoice_date", "created_at"),
        numeric_columns=("purchase_price", "warranty_months", "capacity_value"),
        min_match=0.8,
        required=False,
        description="Asset master (physical assets: purchase/warranty/condition/type/supplier).",
    ),
    # Asset type catalog (expected life, maintenance cycle, depreciation).
    "asset_types": TableSpec(
        name="asset_types",
        signature=frozenset(
            {
                "category_id", "name", "expected_life_months", "depreciation_method",
                "maintenance_cycle_months", "replacement_cost_estimate",
            }
        ),
        primary_key="id",
        date_columns=("created_at",),
        numeric_columns=(
            "expected_life_months", "depreciation_years",
            "maintenance_cycle_months", "replacement_cost_estimate",
        ),
        min_match=0.8,
        required=False,
        description="Asset type catalog (expected life + maintenance cycle + depreciation).",
    ),
    # Asset -> room placement log.
    "asset_allocations": TableSpec(
        name="asset_allocations",
        signature=frozenset(
            {"asset_id", "allocation_type", "allocated_date", "allocated_by", "apartment_id", "bed_id"}
        ),
        primary_key="id",
        date_columns=("allocated_date", "created_at"),
        min_match=0.8,
        required=False,
        description="Asset allocation log (asset -> apartment/bed placement).",
    ),
    # Maintenance item / spare-part master.
    "maintenance_items": TableSpec(
        name="maintenance_items",
        signature=frozenset(
            {"item_name", "item_code", "unit", "minimum_stock_level", "issue_type_id", "default_cost"}
        ),
        primary_key="id",
        date_columns=("created_at", "updated_at"),
        numeric_columns=("default_cost", "default_unit_cost", "minimum_stock_level"),
        min_match=0.8,
        required=False,
        description="Maintenance item / spare-part master.",
    ),
    # Maintenance item purchases / stock receipts.
    "maintenance_item_purchases": TableSpec(
        name="maintenance_item_purchases",
        signature=frozenset(
            {
                "maintenance_item_id", "quantity_received", "quantity_available",
                "purchase_cost_per_unit", "purchased_on", "invoice_number",
            }
        ),
        primary_key="id",
        date_columns=("purchased_on", "created_at"),
        numeric_columns=("quantity_received", "quantity_available", "purchase_cost_per_unit"),
        min_match=0.8,
        required=False,
        description="Maintenance item purchases (stock receipts per item).",
    ),
    # Vendor / supplier master.
    "vendors": TableSpec(
        name="vendors",
        signature=frozenset(
            {"vendor_name", "contact_person", "vendor_rating", "gst_number", "pan_number", "bank_ifsc", "id_proof_url"}
        ),
        primary_key="id",
        date_columns=("created_at",),
        numeric_columns=("vendor_rating",),
        min_match=0.8,
        required=False,
        description="Vendor / service-provider master (rating + KYC + bank).",
    ),
    # Issue type / complaint category (priority + SLA).
    "issue_types": TableSpec(
        name="issue_types",
        signature=frozenset({"name", "icon", "priority", "sla_hours"}),
        primary_key="id",
        date_columns=("created_at",),
        numeric_columns=("sla_hours",),
        min_match=0.8,
        required=False,
        description="Issue type catalog (priority + SLA hours).",
    ),
    # Ticket status-transition log (lifecycle audit trail).
    "ticket_logs": TableSpec(
        name="ticket_logs",
        signature=frozenset({"ticket_id", "action", "old_status", "new_status", "created_by"}),
        primary_key="id",
        date_columns=("created_at",),
        min_match=0.8,
        required=False,
        description="Ticket status-transition log (reconstructs ticket lifecycle).",
    ),
    # Ticket resolutions (resolved_at + cost breakdown + closure).
    "ticket_resolutions": TableSpec(
        name="ticket_resolutions",
        signature=frozenset(
            {
                "ticket_id", "resolution_type", "service_type", "resolved_at",
                "total_parts_cost", "total_labour_cost", "total_cost", "closure_summary",
            }
        ),
        primary_key="id",
        date_columns=("resolved_at", "payment_date", "created_at", "updated_at"),
        numeric_columns=(
            "total_parts_cost", "total_labour_cost", "total_cost",
            "actual_total_cost", "diagnostic_estimated_cost",
        ),
        min_match=0.8,
        required=False,
        description="Ticket resolutions (resolved_at + parts/labour/total cost).",
    ),
    # Ticket cost estimates / approval lines (+ repeat-job recurrence flags).
    "ticket_cost_estimates": TableSpec(
        name="ticket_cost_estimates",
        signature=frozenset(
            {"ticket_id", "cost_type", "approved_by", "decline_reason", "repeat_job_alert", "repeat_job_previous_ticket_id"}
        ),
        primary_key="id",
        date_columns=("approved_at", "created_at", "updated_at"),
        numeric_columns=("quantity", "unit_price", "total", "price"),
        min_match=0.8,
        required=False,
        description="Ticket cost estimates / approvals (with repeat-job flags).",
    ),
    # Per-ticket purchases (estimate -> actual, vendor).
    "ticket_purchases": TableSpec(
        name="ticket_purchases",
        signature=frozenset(
            {"ticket_id", "cost_estimate_id", "estimated_cost", "actual_cost", "vendor_id", "purchased_by"}
        ),
        primary_key="id",
        date_columns=("purchase_date", "created_at", "updated_at"),
        numeric_columns=("estimated_cost", "actual_cost", "quantity"),
        min_match=0.8,
        required=False,
        description="Per-ticket purchases (estimate vs actual cost, vendor).",
    ),
    # Maintenance cost lines (per-ticket cost distribution to beds/tenants).
    "maintenance_cost_lines": TableSpec(
        name="maintenance_cost_lines",
        signature=frozenset(
            {"ticket_id", "purchase_id", "maintenance_type", "parts_details", "diagnosis_summary", "distributed_beds", "cost_scope"}
        ),
        primary_key="id",
        date_columns=("created_at",),
        numeric_columns=("quantity", "unit_price", "actual_cost", "distributed_amount"),
        min_match=0.8,
        required=False,
        description="Maintenance cost lines (per-ticket cost + distribution to beds).",
    ),
    # Maintenance ticket header (asset-level history). Carries asset_id so tickets
    # can be attributed to a specific asset (no room approximation). Source of the
    # leakage-safe predictive-maintenance dataset.
    "maintenance_tickets": TableSpec(
        name="maintenance_tickets",
        signature=frozenset(
            {"asset_id", "ticket_number", "closure_cost", "issue_type_id", "priority", "status", "resolved_at", "sla_deadline"}
        ),
        primary_key="id",
        date_columns=("created_at", "resolved_at", "closed_at", "sla_deadline", "updated_at"),
        numeric_columns=("closure_cost",),
        min_match=0.8,
        required=False,
        description="Maintenance ticket header with asset_id (asset-level ticket history).",
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
      2. the project's parent directory when it holds current Vishful exports
         (Q66–Q76 live one level up under ``Data\\``)
      3. <project>/data (packaged / older snapshot)
      4. the project directory itself
    """
    project_root = Path(__file__).resolve().parents[1]  # .../Availability_AI
    candidates: List[Path] = []

    env_dir = os.environ.get("AVAILABILITY_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    parent = project_root.parent
    packaged = project_root / "data"

    # Prefer the live export folder — detected by COLUMN SIGNATURE, not by a
    # hardcoded filename. The folder is the current normalized export if any CSV
    # header matches the allotments (Q81) signature. Survives filename/export-
    # number changes and legacy-file removal (the old query(73) marker is gone
    # once legacy exports are cleaned out).
    def _has_current_exports(path: Path) -> bool:
        if not path.is_dir():
            return False
        spec = TABLE_REGISTRY.get("allotments")
        if spec is None:
            return False
        for fp in path.glob("*.csv"):
            try:
                cols = set(pd.read_csv(fp, nrows=0).columns)
            except Exception:  # noqa: BLE001
                continue
            if len(spec.signature & cols) / len(spec.signature) >= spec.min_match:
                return True
        return False

    if _has_current_exports(parent):
        candidates.append(parent)
    candidates.extend(
        [
            packaged,
            parent,
            project_root,
        ]
    )

    # Deduplicate while preserving order.
    seen = set()
    ordered: List[Path] = []
    for cand in candidates:
        key = str(cand.resolve()) if cand.exists() else str(cand)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cand)

    for cand in ordered:
        if cand.is_dir() and any(cand.glob("*.csv")):
            return cand
    # Fall back to the conventional data folder even if empty.
    return packaged


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
        """Authoritative tenant table: Q66 identity + Q76 allocation backfill.

        ``tenant_master`` (Vishful Q66) is the identity source (stable ``tenant_id`` +
        demographics). The allocation export (Q76; legacy Q32) backfills home ``city`` /
        ``state`` and current allocation columns ONLY where the master is missing.
        Falls back to whichever source exists. Result is a column superset of the
        old ``tenants`` frame.
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

        Primary source: Vishful Q73 (``actual_exit_date``; legacy Q35 ``exit_date``
        is aliased to ``actual_exit_date``). Older snapshots are column-fallback
        only — never extra history rows. ``tenant_name`` → ``full_name``.
        ``bed_type`` is attached from beds_master / current_occupancy when needed.
        """
        if self._db_export_present():
            df = self._reconstruct_bookings()
        else:
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

        Primary source: Vishful Q74 (``status`` Live / Not-Active → ``bed_status``).
        Legacy Q37/Q33 backfill missing columns only. If ``bed_status`` is still
        missing after that, Not-Active values may be filled from Backup_old
        query (8).
        """
        if self._db_export_present():
            # Q83 status is complete (Live / Not-Active / In-Progress) — no legacy
            # Backup_old bed_status fallback needed.
            return self._reconstruct_beds_master()
        return self._apply_bed_status_fallback(self.load("beds_master"))

    def current_occupancy(self) -> pd.DataFrame:
        """Live bed occupancy snapshot (recommendation occupancy source).

        Primary source: Vishful Q68 (``occupancy_status`` + ``current_tenant``).
        Older occupancy snapshots backfill missing columns only — occupancy
        *rows* always come from the primary so AI matches the Vishful application.
        """
        if self._db_export_present():
            return self._reconstruct_current_occupancy()
        return self.load("current_occupancy")

    def billing_snapshot(self) -> pd.DataFrame:
        """Per-tenant billing snapshot (Vishful Q39), when present."""
        return self.load("billing_snapshot")

    def bed_map(self) -> pd.DataFrame:
        """THE single current-occupancy source: live per-bed status.

        When the normalized DB export (Q81-Q86) is present it is generated
        entirely from the database (``_reconstruct_bed_map``) — no external
        bed-map CSV is read, matching the production application which derives
        availability from the normalized tables. Only when the DB export is
        absent does this fall back to the newest ``bed-map-*.csv`` snapshot.

        One row per bed with the app's live ``Status``. This is the authoritative
        CURRENT occupancy layer — every page's occupancy is derived from it (via
        ``build_room_inventory`` / ``detect_blocked_rooms``); historical analytics
        stay on bookings / tenants / transfers / revenue.

        Columns
        -------
        apartment_code, bed_code, bed_status (raw: vacant/occupied/booked/notice/…),
        is_available (True only when status == 'vacant'), days_vacant (numeric or NaN),
        current_tenant, bed_rate (numeric or NaN), bed_type.

        Returns an empty stable-schema frame when no source exists, so callers
        fall back to the bookings∪occupancy reconstruction (backward compatible).
        """
        cols = [
            "apartment_code", "bed_code", "bed_status", "is_available",
            "days_vacant", "current_tenant", "bed_rate", "bed_type",
        ]
        cache_key = "_bed_map"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Production path: derive the bed-map from the normalized DB export.
        if self._db_export_present():
            out = self._reconstruct_bed_map()
            self._cache[cache_key] = out
            return out

        data_dir = Path(self.data_dir) if self.data_dir else _default_data_dir()
        candidates: List[Path] = list(data_dir.glob("bed-map-*.csv"))
        # Q75-style Supabase exports are not named bed-map-*.csv.
        bed_map_markers = {"Apartment", "Bed", "Status", "Days Vacant"}
        for fp in data_dir.glob("*.csv"):
            if fp in candidates:
                continue
            header_cols = self._csv_header_columns(fp)
            if bed_map_markers.issubset(header_cols):
                candidates.append(fp)
        if not candidates:
            logger.info(
                "No bed-map source found under %s; occupancy falls back to reconstruction.",
                data_dir,
            )
            empty = pd.DataFrame(columns=cols)
            self._cache[cache_key] = empty
            return empty
        # Newest export wins (mtime, then name).
        fp = max(candidates, key=lambda p: (p.stat().st_mtime, p.name))

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
        logger.info(
            "Loaded bed-map source %s (%d beds; %d vacant).",
            fp.name,
            len(out),
            int(out["is_available"].sum()) if not out.empty else 0,
        )
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

    # ------------------------------------------------------------------ #
    # Normalized DB export (Q81-Q86) — raw accessors + JOIN reconstruction
    # ------------------------------------------------------------------ #

    def allotments(self) -> pd.DataFrame:
        """Raw allotments (Vishful Q81): UUID booking/stay event log."""
        return self.load("allotments")

    def bed_rates(self) -> pd.DataFrame:
        """Raw effective-dated bed rate card (Vishful Q84)."""
        return self.load("bed_rates")

    def property_master(self) -> pd.DataFrame:
        """Raw property master (Vishful Q86)."""
        return self.load("property_master")

    # -- Maintenance / asset / ticket / vendor raw accessors -------------- #
    # Each returns an empty frame (not an error) when its export is absent,
    # so the maintenance model degrades gracefully and existing pages are
    # never affected.
    def _load_optional(self, table: str) -> pd.DataFrame:
        try:
            return self.load(table)
        except FileNotFoundError:
            return pd.DataFrame()

    def asset_master(self) -> pd.DataFrame:
        """Physical asset master (purchase/warranty/condition/type/supplier)."""
        return self._load_optional("asset_master")

    def asset_types(self) -> pd.DataFrame:
        """Asset type catalog (expected life, maintenance cycle)."""
        return self._load_optional("asset_types")

    def asset_allocations(self) -> pd.DataFrame:
        """Asset allocation log (asset -> apartment/bed)."""
        return self._load_optional("asset_allocations")

    def maintenance_items(self) -> pd.DataFrame:
        """Maintenance item / spare-part master."""
        return self._load_optional("maintenance_items")

    def maintenance_item_purchases(self) -> pd.DataFrame:
        """Maintenance item purchases (stock receipts)."""
        return self._load_optional("maintenance_item_purchases")

    def vendors(self) -> pd.DataFrame:
        """Vendor / service-provider master."""
        return self._load_optional("vendors")

    def issue_types(self) -> pd.DataFrame:
        """Issue type catalog (priority + SLA)."""
        return self._load_optional("issue_types")

    def ticket_logs(self) -> pd.DataFrame:
        """Ticket status-transition log."""
        return self._load_optional("ticket_logs")

    def ticket_resolutions(self) -> pd.DataFrame:
        """Ticket resolutions (resolved_at + cost)."""
        return self._load_optional("ticket_resolutions")

    def ticket_cost_estimates(self) -> pd.DataFrame:
        """Ticket cost estimates / approvals."""
        return self._load_optional("ticket_cost_estimates")

    def ticket_purchases(self) -> pd.DataFrame:
        """Per-ticket purchases."""
        return self._load_optional("ticket_purchases")

    def maintenance_cost_lines(self) -> pd.DataFrame:
        """Maintenance cost lines (per-ticket distribution)."""
        return self._load_optional("maintenance_cost_lines")

    def maintenance_tickets(self) -> pd.DataFrame:
        """Maintenance ticket header with asset_id (asset-level ticket history)."""
        return self._load_optional("maintenance_tickets")

    def _db_export_present(self) -> bool:
        """True when the normalized DB export (Q81-Q86) is the live source.

        Reconstruction takes over the flattened accessors (bookings /
        beds_master / current_occupancy) only when the three structural
        exports are all discoverable: allotments (Q81), the UUID bed catalog
        (Q83) and the apartment master (Q85). Otherwise the loader falls back
        to the legacy flattened CSV path — behaviour is unchanged without them.
        """
        if self.source != "csv":
            return False
        if "_db_export_present" in self._maps:
            return bool(self._maps["_db_export_present"])
        mapping = self._discover()
        present = all(
            mapping.get(name) for name in ("allotments", "beds_master_uuid", "apartment_master")
        )
        self._maps["_db_export_present"] = present
        logger.info("Normalized DB export (Q81-Q86) present = %s", present)
        return present

    def _uuid_bridges(self) -> Dict[str, pd.DataFrame]:
        """Cached lookup frames that translate UUID keys to text/business fields.

        Returns a dict with keys ``apartments`` (apartment_id -> apartment_code /
        gender_allowed / property_id / status), ``beds`` (bed_id -> apartment_id /
        bed_code / bed_type / toilet_type / status / bed_lifecycle_status),
        ``tenants`` (tenant_id -> full_name / phone) and ``properties``
        (property_id -> property_name). Each is empty-safe.
        """
        if "_uuid_bridges" in self._maps:
            return self._maps["_uuid_bridges"]

        def _safe(name: str) -> pd.DataFrame:
            try:
                df = self.load(name)
            except FileNotFoundError:
                return pd.DataFrame()
            return df if df is not None else pd.DataFrame()

        apts = _safe("apartment_master")
        beds = _safe("beds_master_uuid")
        tens = _safe("tenant_master")
        props = _safe("property_master")

        def _cols(df, cols):
            keep = [c for c in cols if c in df.columns]
            return df[keep].copy() if keep and not df.empty else pd.DataFrame(columns=cols)

        bridges = {
            "apartments": _cols(
                apts, ["id", "apartment_code", "gender_allowed", "property_id", "status"]
            ),
            "beds": _cols(
                beds,
                [
                    "id",
                    "apartment_id",
                    "bed_code",
                    "bed_type",
                    "toilet_type",
                    "status",
                    "bed_lifecycle_status",
                ],
            ),
            "tenants": _cols(tens, ["id", "full_name", "phone"]),
            "properties": _cols(props, ["id", "property_name"]),
        }
        self._maps["_uuid_bridges"] = bridges
        return bridges

    def _active_bed_rate_map(self, as_of: Optional[pd.Timestamp] = None) -> Dict[tuple, float]:
        """Active monthly rate per (property_id, bed_type, toilet_type) from Q84.

        Picks the rate row whose ``from_date`` <= as_of <= ``to_date`` (nulls are
        open-ended). When several qualify the latest ``from_date`` wins; when none
        qualify the latest ``from_date`` overall is used so a rate is always found.
        Keys are lower-cased on bed_type/toilet_type for a case-robust join.
        """
        if "_bed_rate_map" in self._maps:
            return self._maps["_bed_rate_map"]

        out: Dict[tuple, float] = {}
        try:
            rates = self.load("bed_rates")
        except FileNotFoundError:
            rates = None
        if rates is None or rates.empty:
            self._maps["_bed_rate_map"] = out
            return out

        as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()

        def _norm(v) -> str:
            return str(v).strip().lower() if v is not None and not (isinstance(v, float) and pd.isna(v)) else ""

        r = rates.copy()
        r["_pid"] = r.get("property_id", pd.Series(index=r.index, dtype=object)).map(
            lambda v: str(v).strip() if pd.notna(v) else ""
        )
        r["_bt"] = r.get("bed_type", pd.Series(index=r.index, dtype=object)).map(_norm)
        r["_tt"] = r.get("toilet_type", pd.Series(index=r.index, dtype=object)).map(_norm)
        fd = pd.to_datetime(r.get("from_date"), errors="coerce")
        td = pd.to_datetime(r.get("to_date"), errors="coerce")

        for key, g in r.groupby(["_pid", "_bt", "_tt"], sort=False):
            gi = g.index
            g_fd = fd.loc[gi]
            g_td = td.loc[gi]
            active = g.loc[(g_fd.isna() | (g_fd <= as_of)) & (g_td.isna() | (g_td >= as_of))]
            pool = active if not active.empty else g
            # latest from_date within the chosen pool
            pool_fd = fd.loc[pool.index]
            pick_idx = pool_fd.idxmax() if pool_fd.notna().any() else pool.index[-1]
            rate = pd.to_numeric(r.loc[pick_idx, "monthly_rate"], errors="coerce")
            if pd.notna(rate):
                out[key] = float(rate)
        self._maps["_bed_rate_map"] = out
        return out

    def _reconstruct_bookings(self) -> pd.DataFrame:
        """Flattened bookings frame (legacy Q35 schema) from Q81 joins.

        Q81 allotments ⋈ Q85 (apartment_code) ⋈ Q83 (bed_code, bed_type)
        ⋈ Q82 (full_name, phone) ⋈ Q86 (property_name). Preserves the
        allotment event log 1:1 — no rows added or dropped.
        """
        al = self.load("allotments").copy()
        if al.empty:
            return al
        b = self._uuid_bridges()

        def _left(df, bridge, on_left, on_right, renames):
            if bridge.empty or on_left not in df.columns or on_right not in bridge.columns:
                for tgt in renames.values():
                    if tgt not in df.columns:
                        df[tgt] = pd.NA
                return df
            add = bridge.rename(columns={on_right: on_left, **renames})
            add = add[[on_left] + list(renames.values())].drop_duplicates(on_left, keep="last")
            return df.merge(add, on=on_left, how="left")

        al = _left(al, b["apartments"], "apartment_id", "id", {"apartment_code": "apartment_code"})
        al = _left(al, b["beds"], "bed_id", "id", {"bed_code": "bed_code", "bed_type": "bed_type"})
        al = _left(al, b["tenants"], "tenant_id", "id", {"full_name": "full_name", "phone": "phone"})
        al = _left(al, b["properties"], "property_id", "id", {"property_name": "property_name"})
        # tenant_name mirror kept for any legacy consumer / alias parity.
        al["tenant_name"] = al["full_name"]
        return al.reset_index(drop=True)

    def _reconstruct_beds_master(self) -> pd.DataFrame:
        """Flattened beds_master frame (legacy Q37 schema) from Q83 joins.

        Q83 beds ⋈ Q85 (apartment_code, gender_allowed) ⋈ Q84 (active rate ->
        current_rate). One row per physical bed. ``bed_status`` = the bed's own
        Q83 status, forced to Not-Active when its apartment is Not-Active so
        inactive apartments (e.g. A22) stay Inactive downstream exactly as the
        legacy Backup_old bed_status fallback did.
        """
        beds = self.load("beds_master_uuid").copy()
        if beds.empty:
            return beds
        b = self._uuid_bridges()
        apts = b["apartments"]

        # The beds_master_uuid logical table may resolve to a pre-enriched bed
        # view that already carries apartment_code / gender_allowed / property_id.
        # Drop those so the apartment master (Q85) is the single authority for
        # them and the join never produces _x/_y suffix collisions.
        beds = beds.drop(
            columns=[c for c in ("apartment_code", "gender_allowed", "property_id") if c in beds.columns],
            errors="ignore",
        )
        if not apts.empty and "apartment_id" in beds.columns:
            add = apts.rename(columns={"id": "apartment_id"})
            keep = [c for c in ["apartment_id", "apartment_code", "gender_allowed", "property_id", "status"] if c in add.columns]
            add = add[keep].rename(columns={"status": "_apt_status"}).drop_duplicates("apartment_id", keep="last")
            beds = beds.merge(add, on="apartment_id", how="left")
        for col in ("apartment_code", "gender_allowed", "property_id", "_apt_status"):
            if col not in beds.columns:
                beds[col] = pd.NA

        # bed_status: bed's own Q83 status, downgraded to Not-Active for
        # Not-Active apartments (all their beds are inactive).
        def _is_not_active(v) -> bool:
            return str(v).strip().lower().replace(" ", "-") in {"not-active", "inactive", "disabled", "blocked"}

        bed_status = beds.get("status", pd.Series(index=beds.index, dtype=object))
        beds["bed_status"] = [
            "Not-Active" if _is_not_active(aps) else (str(bs).strip() if pd.notna(bs) else pd.NA)
            for bs, aps in zip(bed_status, beds["_apt_status"])
        ]

        # current_rate from the active Q84 rate card by property+bed_type+toilet.
        rate_map = self._active_bed_rate_map()

        def _rate(pid, bt, tt):
            key = (
                str(pid).strip() if pd.notna(pid) else "",
                str(bt).strip().lower() if pd.notna(bt) else "",
                str(tt).strip().lower() if pd.notna(tt) else "",
            )
            return rate_map.get(key, pd.NA)

        beds["current_rate"] = [
            _rate(p, bt, tt)
            for p, bt, tt in zip(beds["property_id"], beds.get("bed_type"), beds.get("toilet_type"))
        ]
        beds["current_rate"] = pd.to_numeric(beds["current_rate"], errors="coerce")
        return beds.reset_index(drop=True)

    # Stay statuses that mean the bed is currently HELD (occupied/reserved).
    _OPEN_STAY_STATUSES = frozenset({"staying", "on-notice", "notice", "booked"})

    def _reconstruct_current_occupancy(self) -> pd.DataFrame:
        """Flattened live-occupancy snapshot (legacy Q50 schema) from Q81.

        An open allotment (``actual_exit_date`` is null AND ``staying_status`` in
        Staying / On-Notice / Booked) marks its bed Occupied; every other Q83 bed
        is Vacant. This is the FALLBACK occupancy layer — the live bed-map
        (``bed_map()``) still overrides it inside ``build_room_inventory`` when
        present. Not part of Q81-Q86, so with no bed-map this Q81 derivation is
        what every page consumes.
        """
        try:
            al = self.load("allotments").copy()
        except FileNotFoundError:
            al = pd.DataFrame()
        beds = self._reconstruct_beds_master()
        if beds.empty:
            return pd.DataFrame(
                columns=[
                    "apartment_code", "bed_code", "bed_type", "occupancy_status",
                    "staying_status", "current_tenant", "monthly_rental", "move_in_date",
                ]
            )

        # Held bed_ids from open allotments.
        held_ids: Dict[str, dict] = {}
        if not al.empty:
            b = self._uuid_bridges()
            tmap = {}
            if not b["tenants"].empty:
                tmap = dict(zip(b["tenants"]["id"].astype(str), b["tenants"]["full_name"]))
            exit_null = al["actual_exit_date"].isna() if "actual_exit_date" in al.columns else pd.Series(True, index=al.index)
            stay = al.get("staying_status", pd.Series(index=al.index, dtype=object)).map(
                lambda v: str(v).strip().lower() if pd.notna(v) else ""
            )
            open_mask = exit_null & stay.isin(self._OPEN_STAY_STATUSES)
            for _, row in al.loc[open_mask].iterrows():
                bid = str(row.get("bed_id")).strip() if pd.notna(row.get("bed_id")) else ""
                if not bid:
                    continue
                held_ids[bid] = {
                    "staying_status": row.get("staying_status"),
                    "current_tenant": tmap.get(str(row.get("tenant_id")).strip()),
                    "monthly_rental": row.get("monthly_rental"),
                    "move_in_date": row.get("onboarding_date"),
                }

        beds_uuid = self.load("beds_master_uuid")
        # Rebuild bed_id alongside apartment_code/bed_code to test held membership.
        bu = beds_uuid[["id", "apartment_id", "bed_code"]].rename(columns={"id": "bed_id"})
        apts = self._uuid_bridges()["apartments"]
        if not apts.empty:
            bu = bu.merge(
                apts.rename(columns={"id": "apartment_id"})[["apartment_id", "apartment_code"]],
                on="apartment_id",
                how="left",
            )
        rows = []
        for _, r in bu.iterrows():
            bid = str(r.get("bed_id")).strip() if pd.notna(r.get("bed_id")) else ""
            info = held_ids.get(bid)
            rows.append(
                {
                    "apartment_code": r.get("apartment_code"),
                    "bed_code": r.get("bed_code"),
                    "occupancy_status": "Occupied" if info else "Vacant",
                    "staying_status": info.get("staying_status") if info else pd.NA,
                    "current_tenant": info.get("current_tenant") if info else pd.NA,
                    "monthly_rental": info.get("monthly_rental") if info else pd.NA,
                    "move_in_date": info.get("move_in_date") if info else pd.NaT,
                }
            )
        occ = pd.DataFrame(rows)
        # attach bed_type from the reconstructed beds master. Dedupe on the
        # bed key first so a duplicate source bed code (e.g. A34|TS2) can't
        # fan the snapshot out to extra rows.
        bt = beds[["apartment_code", "bed_code", "bed_type"]].drop_duplicates(
            ["apartment_code", "bed_code"], keep="first"
        )
        occ = occ.merge(bt, on=["apartment_code", "bed_code"], how="left")
        return occ.reset_index(drop=True)

    def _reconstruct_bed_map(self, as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Generate the live bed-map ENTIRELY from the normalized DB export.

        One row per Q83 bed. No external CSV is read. Fields:

        * ``bed_status`` = Q83 ``bed_lifecycle_status`` (occupied / vacant /
          notice / booked) — the live per-bed status the production app derives.
        * ``is_available`` = True only when status == 'vacant'.
        * ``current_tenant`` = full_name (Q82) of the bed's current open Q81
          allotment (actual_exit_date null AND staying_status in Staying /
          On-Notice / Booked); null when the bed is vacant.
        * ``bed_rate`` = active Q84 rate for (property_id, bed_type, toilet_type).
        * ``days_vacant`` = (as_of - latest actual_exit_date among the bed's Q81
          allotments) for vacant beds; NaN for held beds or beds never occupied.
        * ``apartment_code`` (Q85), ``bed_code`` / ``bed_type`` (Q83).

        ``as_of`` defaults to today (live). Passing a date reproduces the map as
        of that day (used by validation to align with a dated snapshot).
        """
        cols = [
            "apartment_code", "bed_code", "bed_status", "is_available",
            "days_vacant", "current_tenant", "bed_rate", "bed_type",
        ]
        try:
            beds = self.load("beds_master_uuid").copy()
        except FileNotFoundError:
            beds = pd.DataFrame()
        if beds is None or beds.empty:
            return pd.DataFrame(columns=cols)

        as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
        bridges = self._uuid_bridges()
        apts = bridges["apartments"]
        tmap = {}
        if not bridges["tenants"].empty:
            tmap = dict(zip(bridges["tenants"]["id"].astype(str), bridges["tenants"]["full_name"]))
        rate_map = self._active_bed_rate_map(as_of)

        # apartment_code + property_id per bed (authoritative from Q85).
        beds = beds.drop(
            columns=[c for c in ("apartment_code", "property_id") if c in beds.columns],
            errors="ignore",
        )
        if not apts.empty and "apartment_id" in beds.columns:
            add = apts.rename(columns={"id": "apartment_id"})
            keep = [c for c in ["apartment_id", "apartment_code", "property_id"] if c in add.columns]
            beds = beds.merge(add[keep].drop_duplicates("apartment_id", keep="last"), on="apartment_id", how="left")
        for col in ("apartment_code", "property_id"):
            if col not in beds.columns:
                beds[col] = pd.NA

        # Per bed_id from Q81: open (live) stay statuses, current occupant, last exit.
        # Availability is derived from Q81 open allotments — the same authority as
        # _reconstruct_current_occupancy — NOT from Q83 bed_lifecycle_status, which
        # disagrees with Q81/live for ~15 beds. Dead stays (Cancelled/Exited) never
        # hold a bed.
        open_status: Dict[str, set] = {}
        current_tenant: Dict[str, object] = {}
        last_exit: Dict[str, pd.Timestamp] = {}
        # Occupant precedence: the person physically there wins over an incoming
        # booking (Staying > On-Notice > Booked).
        _rank = {"staying": 3, "on-notice": 2, "notice": 2, "booked": 1}
        _best_rank: Dict[str, tuple] = {}
        try:
            al = self.load("allotments")
        except FileNotFoundError:
            al = pd.DataFrame()
        if al is not None and not al.empty and "bed_id" in al.columns:
            stay = al.get("staying_status", pd.Series(index=al.index, dtype=object)).map(
                lambda v: str(v).strip().lower() if pd.notna(v) else ""
            )
            exitd = al["actual_exit_date"] if "actual_exit_date" in al.columns else pd.Series(pd.NaT, index=al.index)
            onboard = al["onboarding_date"] if "onboarding_date" in al.columns else pd.Series(pd.NaT, index=al.index)
            open_mask = exitd.isna() & stay.isin(self._OPEN_STAY_STATUSES)
            for idx in al.index[open_mask]:
                bid = str(al.at[idx, "bed_id"]).strip() if pd.notna(al.at[idx, "bed_id"]) else ""
                if not bid:
                    continue
                st = stay.at[idx]
                open_status.setdefault(bid, set()).add(st)
                ob = onboard.at[idx]
                ob_key = ob if pd.notna(ob) else pd.Timestamp.min
                cand = (_rank.get(st, 0), ob_key)
                if bid not in _best_rank or cand > _best_rank[bid]:
                    _best_rank[bid] = cand
                    current_tenant[bid] = tmap.get(str(al.at[idx, "tenant_id"]).strip())
            # latest exit per bed (days_vacant of vacant beds).
            ex_idx = al.index[exitd.notna()]
            if len(ex_idx):
                ex = pd.DataFrame({"bid": al.loc[ex_idx, "bed_id"].astype(str).values,
                                   "ex": exitd.loc[ex_idx].values})
                last_exit = ex.groupby("bid")["ex"].max().to_dict()

        def _norm(v) -> str:
            return str(v).strip().lower() if pd.notna(v) else ""

        def _status_from_open(s: set) -> str:
            if "on-notice" in s and "booked" in s:
                return "notice-booked"
            if "on-notice" in s or "notice" in s:
                return "notice"
            if "booked" in s:
                return "booked"
            if "staying" in s:
                return "occupied"
            return "vacant"

        rows = []
        for _, r in beds.iterrows():
            bid = str(r.get("id")).strip() if pd.notna(r.get("id")) else ""
            status = _status_from_open(open_status.get(bid, set()))
            is_vac = status == "vacant"
            dv = pd.NA
            if is_vac:
                le = last_exit.get(bid)
                if le is not None and pd.notna(le):
                    dv = (as_of - pd.Timestamp(le).normalize()).days
            pid = str(r.get("property_id")).strip() if pd.notna(r.get("property_id")) else ""
            rate = rate_map.get((pid, _norm(r.get("bed_type")), _norm(r.get("toilet_type"))), pd.NA)
            rows.append(
                {
                    "apartment_code": r.get("apartment_code"),
                    "bed_code": r.get("bed_code"),
                    "bed_status": status,
                    "is_available": is_vac,
                    "days_vacant": pd.to_numeric(dv, errors="coerce"),
                    "current_tenant": (pd.NA if is_vac else current_tenant.get(bid)),
                    "bed_rate": pd.to_numeric(rate, errors="coerce"),
                    "bed_type": r.get("bed_type"),
                }
            )
        out = pd.DataFrame(rows, columns=cols)
        out = out[out["apartment_code"].notna() & out["bed_code"].notna()].reset_index(drop=True)
        logger.info(
            "Generated bed-map from Q81-Q86 (%d beds; %d vacant).",
            len(out),
            int(out["is_available"].sum()) if not out.empty else 0,
        )
        return out

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
                scored_cols = set(cols)
                # Q74 ships `status` (Live/Not-Active); treat as bed_status for match.
                if name == "beds_master" and "status" in scored_cols:
                    scored_cols = scored_cols | {"bed_status"}
                overlap = len(spec.signature & scored_cols) / len(spec.signature)
                if overlap > best_score:
                    best_name, best_score = name, overlap
            # Text-key bed catalog (apartment_code + current_rate) must not be
            # swallowed by beds_master_uuid (which also matches Q74 strongly).
            if (
                best_name == "beds_master_uuid"
                and {"apartment_code", "current_rate", "toilet_type"}.issubset(cols)
            ):
                bm_cols = set(cols) | ({"bed_status"} if "status" in cols else set())
                bm_spec = TABLE_REGISTRY["beds_master"]
                bm_score = len(bm_spec.signature & bm_cols) / len(bm_spec.signature)
                if bm_score >= bm_spec.min_match:
                    best_name, best_score = "beds_master", bm_score
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

        Deployment fix: several generations of the same logical export can coexist
        in ``data_dir`` — e.g. ``query (81..86)``, ``(87)``, ``(89)``, ``(90)`` …
        The primary is ALWAYS the newest file by modification time (never filename
        order, never a hardcoded query number). Older snapshots are never
        concatenated as extra rows — they may only fill columns the primary lacks.
        """
        if len(files) <= 1:
            # Single match -> unchanged behaviour.
            return list(files)

        # Requirement: collect matches, sort DESC by Path.stat().st_mtime, take [0].
        matches = sorted(files, key=lambda p: p.stat().st_mtime, reverse=True)
        primary = matches[0]
        logger.info("%s -> %s", _LOG_LABEL.get(table, table), primary.name)

        selected = [primary]
        if table in _COLUMN_FALLBACK_TABLES:
            for fp in matches[1:]:
                if fp.resolve() == primary.resolve():
                    continue
                # Skip mega-join dumps (Q69) that share bed columns but are not
                # a beds catalog — they would stamp allotment fields onto rows.
                if table == "beds_master":
                    cols = self._csv_header_columns(fp)
                    if {"booking_date", "staying_status", "tenant_id"}.issubset(cols):
                        continue
                selected.append(fp)
                logger.info(
                    "%s secondary (column backfill only): %s",
                    _LOG_LABEL.get(table, table),
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
