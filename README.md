# Availability AI Recommendation System

An intelligent availability system for a PG / co-living business, with two
modules:

1. **Customer Room Recommendation** — suggest the best **rooms** to a new enquiry.
2. **Management Analytics** — occupancy, blocked-room detection, revenue loss.

### Recommendations are always ROOM-level

The customer is always shown a **room with a specific available bed**, never an
apartment. Apartment occupancy/revenue are **internal ranking signals only**.

Location hierarchy: `Property → Apartment (internal ranking) → Room (shown) →
Bed (the offered bed)`. **Room identity lives in exactly one place** —
`preprocessing.build_room_inventory()` — which emits an opaque `room_code`.
Every other module consumes the **Room Inventory** and never derives rooms from
`bed_code`, so the future DB (with its own `room_id` / `room_code` / `capacity`)
requires no downstream change.

**Flow:** city → resolve state → find rooms (≥1 free bed) whose current
occupants share the customer's state → rank by the ladder below.

**Priority ladder (room-level) — ACTIVE:**
1. **Priority 1** — same-state rooms inside the **lowest-occupancy apartments**
   (fill those apartments first).
2. **Priority 2** — highest **demand score**.
3. **Priority 3** — **budget** match (final tie-breaker only).

If no same-state room exists anywhere, fall back to any available room ranked by
lowest occupancy → demand. **Revenue is never used for recommendation ranking**
(it is reserved for pricing / leakage / management analytics). The engine
optimises occupancy and business fill strategy. Bed-type preference
(Triple→Double→Single) is a filter with automatic nearest-type fallback.

**Roommate compatibility score (0–100) — DISPLAY ONLY (for now).** Computed per
room from whatever tenant data exists (same-state, gender, student/working, age
similarity, current occupancy), averaging only the *available* factors so it
strengthens automatically as more tenant data (e.g. from PostgreSQL) arrives.
It is returned as an extra output column but does **not** influence ranking. It
becomes an active ranking signal only when the business enables roommate
preference matching (`recommend_rooms(..., use_compatibility_in_ranking=True)`)
— no architectural change required.

**Each of the Top-3 recommendations shows:** Apartment Code · Room Code · Bed
Code · Bed Type · Monthly Rent · Current Occupancy % · Available Beds ·
Roommates' States · Compatibility Score (info) · Demand Score · Reason
(e.g. *"Same-state roommates + Lowest-occupancy apartment + High demand score +
Budget match"*).

### Room Master (single source of truth)

`build_room_inventory()` produces one row per room, consumed by every module:

| apartment_code | room_code | bed_type | capacity | occupied_beds | available_beds | available_bed_codes | current_occupancy_pct | current_revenue | current_rent | recent_occupancy_pct | recent_revenue | recent_vacancy_events | recent_refills | recent_fill_rate | average_vacancy_days_recent | historical_occupancy_pct | historical_revenue | occupant_states | occupant_genders | occupant_occupations | occupant_ages | bed_codes | demand_score |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|

**Room demand indicators** (recent window):
- `recent_fill_rate` = refills ÷ vacancy events (2025+).
- `average_vacancy_days_recent` = mean exit→refill gap (2025+).
- `demand_score` (0–100) = composite of recent occupancy, recent fill rate, low
  average vacancy, and revenue performance — each signal **min-max normalised
  across the current room set** (no hardcoded thresholds; weights in
  `utils.DEMAND_SCORE_WEIGHTS`). Reused by recommendation, pricing, dashboard and
  revenue-leakage analytics.

**Two analytics windows** (configured in `src/utils.py`):
- **Historical baseline — 2023+** → `historical_*` columns, long-term reporting.
- **Recent operational baseline — 2025+** → `recent_*` columns, used for
  recommendations and blocked-room/anomaly detection (operations stabilised in
  2025). Blocked-room detection compares current vacancy **primarily** against
  the recent average fill time, and reports the historical average too.
- Pre-2023 data is **never discarded** — it still defines room capacity and room
  age; it is just excluded from the baseline occupancy/revenue/fill-time metrics.

Everything is **dynamic** — no hardcoded apartment names, tenants, cities,
states, revenues or occupancies. The system adapts automatically as data grows.

> **Status: all phases complete.** Data loader, Room Inventory, Occupancy
> History, Fill-Time engine, Recommendation Engine (with display-only roommate
> compatibility), Blocked-Room Detection, Portfolio Revenue-Leakage analytics
> and the Streamlit dashboard are implemented. See the roadmap at the end and
> `PRODUCTION_VALIDATION_REPORT.md` for the validation pass.

---

## Future-proofing (the key design rule)

Today data comes from CSV. Later it will come from **PostgreSQL / Supabase**.

Therefore **only `src/data_loader.py` knows where data comes from.** Every other
module works purely with pandas DataFrames. When the DB is connected, only
`data_loader.py` changes (implement `_load_table_db`); the recommendation
engine, analytics and dashboard keep working unchanged because the **logical
table names and DataFrame schemas stay identical**.

```
raw source (CSV today / Postgres tomorrow)
        │
        ▼
  data_loader.py   ← the ONLY source-aware module
        │  returns clean pandas DataFrames (stable schema)
        ▼
 preprocessing → occupancy / revenue / roommate / blocked-room → ranking
        │
        ▼
   recommendation_engine  +  management analytics
        │
        ▼
        dashboard/app.py
```

## Project structure

```
Availability_AI/
├── data/                     # (optional) local copy of source data
├── outputs/                  # generated reports/artifacts (see below)
├── docs/
│   ├── DATA_DICTIONARY.md    # every dataset & column
│   └── RELATIONSHIPS.md      # how the tables connect (ER + join keys)
├── src/
│   ├── data_loader.py        # source of truth for data ingress (CSV → DB later)
│   ├── utils.py              # generic helpers + analytics config (windows/weights)
│   ├── preprocessing.py      # build_room_inventory + build_occupancy_history
│   ├── occupancy_analysis.py # fill-time engine + vacancy_report
│   ├── roommate_matching.py  # city→state + compatibility_scores
│   ├── ranking.py            # recommendation ranking ladder
│   ├── recommendation_engine.py  # orchestrates customer recommendations
│   ├── blocked_room_detector.py  # blocked rooms + per-bed revenue leakage
│   └── revenue_analytics.py  # portfolio leakage aggregation (no recalculation)
├── dashboard/
│   └── app.py                # Streamlit dashboard (display only)
├── requirements.txt
└── README.md
```

### Output files (generated by the pipeline)

| File | Produced by | Contents |
|---|---|---|
| `outputs/room_inventory.csv` | `preprocessing.py` | one row per room: capacity, occupancy (current/recent/historical), rent, demand, occupants |
| `outputs/occupancy_history.csv` | `preprocessing.py` | per-bed tenancy timeline + vacancy gaps |
| `outputs/room_fill_time.csv` | `occupancy_analysis.py` | per-room recent/historical average fill days + expected fill days |
| `outputs/vacancy_report.csv` | `occupancy_analysis.py` | per vacant bed: vacant days, expected fill, status |
| `outputs/blocked_rooms.csv` | `blocked_room_detector.py` | per vacant bed: status + `estimated_revenue_loss` + reason |
| `outputs/revenue_summary.csv` | `revenue_analytics.py` | portfolio KPIs (metric, value) |
| `outputs/revenue_leakage_by_apartment.csv` | `revenue_analytics.py` | per-apartment leakage + status counts |
| `outputs/recommendations_demo.csv` | `recommendation_engine.py` | 5 demo customer recommendations |

## The datasets

`data_loader.py` classifies each CSV into a **logical table** by its column
signature (filenames like `Supabase Snippet Untitled query (23).csv` are
ignored). Required tables:

| Logical table | Business area | Source file |
|---|---|---|
| `tenants` | Tenant Details (+ current allocation, home city/state) | (21) + (23) |
| `bookings` | Room Allocation + Booking History + Notice + bed rent | (22) |
| `invoices` | Revenue / Invoice (accrued) | (28) |
| `payments` | Revenue (collected) | (29) |
| `maintenance` | Maintenance tickets | (24) |
| `electricity_bills` | Electricity bill payments | (25) |
| `electricity_readings` | Electricity meter readings | (26) |

Recognised but **out of scope**: `assets` (27, ignored), `expenses` (30, optional).

See **[docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md)** and
**[docs/RELATIONSHIPS.md](docs/RELATIONSHIPS.md)** for full detail.

## Quick start

```bash
pip install -r requirements.txt

# Verify data discovery + row counts (Phase-1 sanity check)
python src/data_loader.py
```

By default the loader auto-discovers the data folder. To point it elsewhere:

```bash
# Windows PowerShell
$env:AVAILABILITY_DATA_DIR = "D:\path\to\csvs"
```

```python
from src.data_loader import DataLoader

loader = DataLoader()            # source="csv" (default)
tenants  = loader.tenants()      # clean DataFrame
bookings = loader.bookings()
print(loader.profile())          # table / rows / columns summary
```

When the database is ready (Phase 2):

```python
loader = DataLoader(source="postgres", connection=engine)  # same API, same schemas
```

---

## Requirement feasibility with the current data

| Business requirement | Feasible? | Data used |
|---|:--:|---|
| Identify tenant details | ✅ | `tenants` |
| Bed details (type, rent) | ⚠️ derived | `bookings.bed_type`, `monthly_rental` (no bed master) |
| Room allocation (who is where now) | ✅ | `bookings` (open) / `tenants` current cols |
| Booking history | ✅ | `bookings` (multi-row per tenant) |
| Revenue / invoice | ✅ | `invoices`, `payments`, `bookings.monthly_rental` |
| Notice tracking | ✅ | `bookings.notice_date`, status `On-Notice` |
| Maintenance | ✅ | `maintenance` |
| Electricity | ✅ | `electricity_bills`, `electricity_readings` |
| City → State mapping | ⚠️ partial | derived from `tenants(city,state)` + fallback needed |
| Same-state room preference | ✅ | occupied beds → `tenants.state` |
| Room-level unit (apartment+bed-letter) + available beds | ✅ | derived room = `apartment_code`+`bed_code[0]`, capacity from `bed_type` |
| Lowest occupancy apartment (internal ranking) | ✅ | derived occupancy per apartment |
| Lowest revenue apartment (internal ranking) | ✅ | `bookings.monthly_rental` (or invoices via bridge) |
| Bed-type preference (Triple→Double→Single) | ✅ | `bookings.bed_type` |
| Budget filter / pricing | ✅ | `bookings.monthly_rental` |
| Historical average fill time | ✅ | `bookings` onboarding vs prior exit gaps |
| Current vacant duration | ✅ | last `actual_exit_date` → today |
| Estimated revenue loss | ✅ | bed rent × vacant days |
| Blocked-room (Triple run as Double) detection | ✅ | occupancy + fill-time comparison |

**Conclusion: every Phase-1 business requirement is implementable** with the
available data, subject to the derived assumptions and gaps below.

## ⚠️ Gap report — missing data / columns to confirm before Phase 2

1. **No bed / room master table.** There is no dataset listing every bed and its
   room capacity. We must *derive* bed inventory from `bookings` (distinct
   `apartment_code`+`bed_code`) and room capacity from `bed_type`
   (Single=1, Double=2, Triple=3). **Impact:** a bed that was never booked is
   invisible; occupancy denominators rely on this inference.
   **Recommendation:** provide a beds/rooms master (apartment → beds → type,
   base rent) if available.

2. **No `apartment_id` ↔ `apartment_code` mapping.** `invoices`/`payments` use
   UUID `apartment_id`/`bed_id`; every other table uses text codes. Apartment-level
   *collected* revenue must be routed through `tenants.id = tenant_id`, which
   reflects a tenant's **current** apartment and can misattribute historical
   invoices. **Recommendation:** expose an apartment/bed dimension table mapping
   UUIDs to codes. Until then, apartment revenue is computed from
   `bookings.monthly_rental` (current run-rate), which needs no bridge.

3. **City → State mapping is only as complete as the tenant data.** We can learn
   `city → state` from `tenants`, but a customer city not present among tenants
   (e.g. **Coimbatore** if no tenant is from there) cannot be resolved.
   **Recommendation:** add a static India city→state reference (layered on top of
   the learned map) to guarantee coverage.

4. **No property location (`city`/`state`) column.** Properties can't be
   geo-filtered by the customer's city; the customer's city is currently used
   only to derive their state for roommate matching. With a single property in
   the data this is acceptable, but multi-city expansion needs a property
   location column. **Recommendation:** add `property.city` / `property.state`.

5. **No dedicated Notice table.** Notice is inferred from `bookings.notice_date`
   and `staying_status='On-Notice'`. Sufficient for now; confirm this is the
   authoritative source.

6. **Status label inconsistency.** `staying_status` mixes case
   (`staying`/`Staying`, `on-notice`/`On-Notice`, `exited`/`Exited`). Handled by
   case-insensitive normalisation in `utils.py`; flagged so it is not treated as
   distinct categories.

None of these block Phase 1. Items **1–3** should be confirmed with the business
before the recommendation/analytics logic is built in Phase 2.

---

## Phase 2 implementation order

Everything is fully dynamic — no apartment/room/bed codes, cities, states,
revenue, occupancy, capacity or bed types are ever hardcoded. The same code
works when new CSVs are dropped in today and when PostgreSQL/Supabase replaces
the CSV source (only `data_loader.py` changes).

1. ✅ **Build Room Inventory** — `preprocessing.build_room_inventory()` (single
   source of truth: room identity, capacity, two-window occupancy/revenue, and
   demand indicators). Saved to `outputs/room_inventory.csv`.
2. ✅ **Build Occupancy History** — `preprocessing.build_occupancy_history()`
   (canonical per-bed tenancy timeline + vacancy gaps). Saved to
   `outputs/occupancy_history.csv`.
3. ✅ **Build Fill-Time engine** — `occupancy_analysis.fill_time()` +
   `current_vacant_days()` + `vacancy_report()` (recent 2025+ primary, historical
   2023+ reporting/fallback; dynamic Normal/Delayed/Critical status). Saved to
   `outputs/vacancy_report.csv` and `outputs/room_fill_time.csv`.
4. ✅ **Build Customer Recommendation engine** — `recommendation_engine.recommend_rooms()`
   composing `roommate_matching` (city→state: tenant-learned + static India
   fallback; `compatibility_scores` — display-only) + `ranking` (lowest-occupancy
   apartment + same-state → demand → budget; bed-type filter with nearest
   fallback; revenue excluded from ranking). Room-level Top-3 with reasons.
5. ✅ **Build Blocked-Room detection & Revenue Leakage** —
   `blocked_room_detector.detect_blocked_rooms()`. Consumes only existing outputs
   (Room Inventory + Occupancy History + Fill-Time `vacancy_report`); per vacant
   bed it classifies Normal/Delayed/Critical/Unknown against the room's OWN
   expected fill days and computes `estimated_revenue_loss = current_rent ×
   (current_vacant_days / 30.44)` (room/bed level only — no apartment/invoice
   revenue). Sorted Critical → highest loss → longest vacancy. Saved to
   `outputs/blocked_rooms.csv`.
6. ✅ **Build Portfolio Revenue-Leakage analytics** —
   `revenue_analytics.summarize()` / `generate_and_save()`. **Aggregation only**
   over `blocked_rooms` (the single source of truth for leakage — no
   recalculation): portfolio KPIs (total leakage, Critical/Delayed/Unknown room
   counts, average vacancy days, highest-loss room & apartment) plus a
   per-apartment breakdown. Saved to `outputs/revenue_summary.csv` and
   `outputs/revenue_leakage_by_apartment.csv`.
7. ✅ **Build Dashboard** — `dashboard/app.py`. **Display only** — reads the
   generated `outputs/*.csv` and calls the existing Recommendation Engine; it
   never computes occupancy, fill time, blocked rooms, leakage, demand or
   recommendations. Five pages: Customer Recommendation, Blocked Rooms, Revenue
   Leakage, Occupancy Analytics, Room Search. Rerunning the pipeline refreshes
   every page with no code change.

### Running the dashboard

Generate the outputs, then launch Streamlit:

```bash
python src/preprocessing.py          # room_inventory.csv, occupancy_history.csv
python src/occupancy_analysis.py     # room_fill_time.csv, vacancy_report.csv
python src/blocked_room_detector.py  # blocked_rooms.csv
python src/revenue_analytics.py      # revenue_summary.csv, revenue_leakage_by_apartment.csv
streamlit run dashboard/app.py
```

Use the sidebar **Refresh data** button after re-running the pipeline to clear
the dashboard's cache.
