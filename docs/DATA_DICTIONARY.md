# Data Dictionary — Availability AI Recommendation System

This dictionary describes the datasets found in the `data/` folder. Files are
Supabase CSV exports with auto-generated names (`Supabase Snippet Untitled query
(NN).csv`). Because those names are not stable, `data_loader.py` classifies each
file into a **logical table** by matching its column *signature*, not its
filename. The logical table is the name every other module uses.

| Source file (current export) | Logical table | Rows (approx) | Business area | In scope |
|---|---|---:|---|:--:|
| query (23).csv | `tenants` | ~1,055 | Tenant Details | ✅ |
| query (21).csv | `tenants` (extra snapshot, merged) | ~102 | Tenant Details | ✅ |
| query (22).csv | `bookings` | ~1,163 | Room Allocation + Booking History + Notice | ✅ |
| query (28).csv | `invoices` | ~6,694 | Revenue / Invoice | ✅ |
| query (29).csv | `payments` | ~8,314 | Revenue (collections) | ✅ |
| query (24).csv | `maintenance` | ~1,523 | Maintenance | ✅ |
| query (25).csv | `electricity_bills` | ~67 | Electricity | ✅ |
| query (26).csv | `electricity_readings` | ~32 | Electricity | ✅ |
| query (27).csv | `assets` | ~1,697 | Asset inventory | ❌ ignored |
| query (30).csv | `expenses` | ~503 | Expense ledger | ❌ optional |

> Files `(21)` and `(23)` share the exact same schema (tenant profiles); the
> loader concatenates them and de-duplicates on `id`.

---

## 1. `tenants` — Tenant Details  *(files 21 & 23)*

Tenant identity, KYC, **home city/state**, and **current bed allocation**. The
current-allocation columns (`property_name`, `apartment_code`, `bed_code`) are
populated only for tenants who currently hold a bed (~207 rows); they are null
for exited tenants.

| Column | Type | Description / Notes |
|---|---|---|
| `id` | uuid | **Primary key** (= `tenant_id` in invoices/payments). |
| `organization_id` | uuid | Owning organisation (single org in this export). |
| `user_id` | uuid | Auth user id. |
| `full_name`, `first_name`, `last_name` | text | Tenant name. |
| `phone`, `email` | text | Contact. |
| `date_of_birth`, `age`, `gender` | date/int/text | Demographics. |
| `city`, `pincode`, `state` | text | **Home location — drives same-state matching.** |
| `profession`, `course`, `company_name`, `company_city`, `company_state`, `designation` | text | Work/study details. |
| `staying_status` | text | `staying`, `on-notice`, `exited`, … (case varies). |
| `food_preference` | text | Veg / Non-veg (optional). |
| `tenant_rating`, `rating_last_computed` | num/ts | Behaviour rating. |
| `kyc_completed` | bool | KYC flag. |
| id/bank/aadhar/pan fields | text | KYC & payout details (mostly out of scope). |
| `created_at`, `date_of_joining` | timestamp | Record + joining dates. |
| `property_id` | uuid | Property FK (UUID). |
| `property_name` | text | **Current property (text key).** |
| `apartment_code` | text | **Current apartment (text key), e.g. `B43`.** |
| `bed_code` | text | **Current bed (text key), e.g. `A1`, `C2`.** |

## 2. `bookings` — Allocation + Booking History + Notice  *(file 22)*

The operational backbone. One row per **allocation event** (a tenant occupying a
bed for a period). Multiple rows per tenant (history of moves). Contains rent and
notice fields, so it also serves the Notice requirement.

| Column | Type | Description / Notes |
|---|---|---|
| `full_name`, `phone` | text | Tenant (join key to `tenants`). |
| `property_name` | text | Property (all `Vista Heights` in this export). |
| `apartment_code` | text | Apartment/unit code. |
| `bed_code` | text | Bed code within the apartment. |
| `bed_type` | text | `Single` / `Double` / `Triple` → **room capacity 1/2/3**. |
| `onboarding_date` | date | Move-in date. |
| `estimated_exit_date` | date | Planned exit. |
| `actual_exit_date` | date | **Null ⇒ booking is currently active/occupied.** |
| `notice_date` | date | Date notice was given (Notice requirement). |
| `monthly_rental` | numeric | **Bed rent — revenue & revenue-loss driver.** |
| `deposit_paid`, `discount`, `onboarding_charges`, `prorated_rent` | numeric | Money fields. |
| `staying_status` | text | `Staying`, `On-Notice`, `Exited`, … |
| `total_due`, `paid_amount`, `balance_due`, `payment_status` | num/text | Dues snapshot. |
| `expected_payment_date`, `expected_stay_days` | date/num | Mostly null. |

## 3. `invoices` — Revenue / Invoice ledger  *(file 28)*

Accrued revenue. UUID-keyed (Supabase native table).

| Column | Type | Description / Notes |
|---|---|---|
| `id` | uuid | **Primary key.** |
| `tenant_id` | uuid | FK → `tenants.id` (**bridge to text world**). |
| `property_id`, `apartment_id`, `bed_id` | uuid | UUID location keys (no text-code map in data — see relationships). |
| `allotment_id` | uuid | Allocation reference (= `payments.tenant_allotment_id`). |
| `invoice_number` | text | Human invoice no. |
| `invoice_date`, `due_date`, `created_at` | date/ts | Dates. |
| `billing_month` | text | e.g. `2026-07`. |
| `rent_amount`, `electricity_amount`, `other_charges`, `total_amount` | numeric | Charge breakdown. |
| `amount_paid`, `balance`, `late_fee`, `estimated_eb` | numeric | Settlement. |
| `status` | text | `paid`, `pending`, … |
| `invoice_type`, `reference_type`, `reference_id` | text/uuid | Classification. |
| `locked`, `is_deleted` | bool | Flags. |

## 4. `payments` — Collected revenue  *(file 29)*

| Column | Type | Description / Notes |
|---|---|---|
| `id` | uuid | **Primary key.** |
| `tenant_id` | uuid | FK → `tenants.id`. |
| `tenant_allotment_id` | uuid | Allocation reference (= `invoices.allotment_id`). |
| `payment_date`, `created_at` | date/ts | Dates. |
| `payment_mode` | text | `upi`, `cash`, … (case varies). |
| `amount_paid`, `base_amount`, `processing_fee` | numeric | Amounts. |
| `receipt_number`, `reference_number` | text | Receipt refs. |
| `receipt_type` | text | `payment`, … |
| `bank_account_id` | uuid | Deposit account. |

## 5. `maintenance` — Tickets  *(file 24)*

| Column | Type | Description / Notes |
|---|---|---|
| `ticket_number` | text | **Primary key.** |
| `property_name`, `apartment_code`, `bed_code` | text | Location (text keys). |
| `issue_type`, `issue_sub_type`, `description` | text | Issue detail. |
| `priority`, `status` | text | `low/medium/high`, `assigned/resolved/…`. |
| `tenant_name`, `tenant_phone` | text | Reporter. |
| `assigned_to` | text | Assignee. |
| `created_at`, `sla_deadline`, `resolved_at`, `closed_at` | ts | Lifecycle. |
| `tenant_approved` | bool | Closure approval. |

## 6. `electricity_bills` — Bill payments  *(file 25)*

| Column | Type | Description |
|---|---|---|
| `property_name`, `apartment_code` | text | Location. |
| `bill_date`, `payment_date` | date | Dates. |
| `bill_amount` | numeric | Billed amount. |
| `payment_mode`, `reference_number` | text | Payment info. |
| `billing_period_start`, `billing_period_end` | date | Period. |
| `notes` | text | Free text. |

## 7. `electricity_readings` — Meter readings  *(file 26)*

| Column | Type | Description |
|---|---|---|
| `property_name`, `apartment_code` | text | Location. |
| `eb_meter_number` | text | Meter id. |
| `reading_date` | date | Reading date. |
| `start_reading`, `current_reading`, `units_consumed` | numeric | Consumption. |
| `eb_amount` | numeric | Cost. |
| `is_danger` | bool | High-consumption flag. |

## 8. `assets` — Inventory *(file 27, IGNORED)*

Physical assets (ACs, mattresses, …) keyed by `asset_code` + `apartment_code`.
Not required by the availability/recommendation scope.

## 9. `expenses` — Expense ledger *(file 30, OPTIONAL)*

UUID-keyed expense records (often linked to maintenance resolutions/assets).
Not required for Phase-1 business goals; available if profitability analysis is
added later.
