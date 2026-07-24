# Production Validation Report — Availability AI

**Phase 8 — Validation & Production Readiness**
Scope: validation, testing, bug-fixing, cleanup and documentation only. No new
features, no business-logic changes, no changes to recommendation ranking,
blocked-room detection or revenue analytics.

> **Execution environment note:** the IDE shell in this workspace does not return
> process output (commands do not execute), so the pipeline could **not be run
> here**. This report therefore documents a thorough **static** validation
> (code review, import/reference resolution, dead-code and hardcoding scans,
> data-flow tracing) plus a **runtime checklist to run on your machine**. Every
> item below marked _static_ was verified by inspection; items marked _pending_
> require a local run.

---

## 1. Modules tested (static review)

| Module | Verified (static) |
|---|---|
| `src/data_loader.py` | Exposes `tenants()`, `bookings()`, `invoices()`; schema-signature discovery; DataFrame contract stable for DB swap. |
| `src/utils.py` | Pure helpers + config (windows, demand weights). Unused helpers removed. `Optional`/`Iterable` still required. |
| `src/preprocessing.py` | `build_room_inventory` (adds `current_rent`, occupant `state/gender/occupation/age`), `build_occupancy_history`. All `from utils` imports used. Invariants enforced in `_validate_room_inventory`. |
| `src/occupancy_analysis.py` | `fill_time`, `current_vacant_days`, `vacancy_report`. Unchanged (Step 3). |
| `src/roommate_matching.py` | `resolve_state` (tenant-learned → static fallback), `same_state_rooms`, `compatibility_scores` (0–100, dynamic factors). |
| `src/ranking.py` | Active ladder: occupancy → demand → budget. Revenue excluded. Compatibility computed but **display-only** unless `use_compatibility_in_ranking=True`. |
| `src/recommendation_engine.py` | `recommend_rooms` orchestration + demo. `_apartment_metrics` trimmed to occupancy-only (revenue was unused). |
| `src/blocked_room_detector.py` | Consumes existing outputs only; per-bed leakage = `current_rent × (vacant_days / 30.44)`; Critical→loss→vacancy sort. |
| `src/revenue_analytics.py` | Aggregation only over `blocked_rooms` (single source of truth); portfolio KPIs + per-apartment breakdown. |
| `dashboard/app.py` | Display only; reads `outputs/*.csv`, delegates recommendations to the engine; 5 pages; filters; cache-refresh button. |

**Import graph:** no circular imports (`utils` ← everything; `roommate_matching` ←
`ranking` ← `recommendation_engine`; analytics modules independent). All
module-level imports resolve to used symbols.

---

## 2. Outputs generated (produced by running the pipeline)

| File | Producing command |
|---|---|
| `room_inventory.csv`, `occupancy_history.csv` | `python src/preprocessing.py` |
| `room_fill_time.csv`, `vacancy_report.csv` | `python src/occupancy_analysis.py` |
| `blocked_rooms.csv` | `python src/blocked_room_detector.py` |
| `revenue_summary.csv`, `revenue_leakage_by_apartment.csv` | `python src/revenue_analytics.py` |
| `recommendations_demo.csv` | `python src/recommendation_engine.py` |

---

## 3. Runtime validation checklist (run locally)

```powershell
cd "D:\data science\Availability  Analysis\Data\Availability_AI"
pip install -r requirements.txt
python src/preprocessing.py
python src/occupancy_analysis.py
python src/blocked_room_detector.py
python src/revenue_analytics.py
python src/recommendation_engine.py
streamlit run dashboard/app.py
```

- [ ] _pending_ — each script exits without a traceback.
- [ ] _pending_ — all 7 output files exist and are non-empty.
- [ ] _pending_ — `room_inventory`: no duplicate `room_code`; `available_beds =
      capacity − occupied_beds`; occupancy ≤ 100% (enforced by
      `_validate_room_inventory`, which raises on violation).
- [ ] _pending_ — `blocked_rooms`: `vacancy_status ∈ {Normal, Delayed, Critical,
      Unknown}`; `estimated_revenue_loss` present where `current_rent` exists.
- [ ] _pending_ — dashboard: all 5 pages load; Apartment/Status/Room-type/Occupancy
      filters work; recommendation page returns Top-3 with detected state.
- [ ] _pending_ — dynamic test: edit a few source rows → rerun the 5 scripts →
      click **Refresh data** → verify recommendations/blocked/leakage/occupancy
      change with **no code edits**.

### Data-integrity expectations (by construction)
- **No duplicate rows:** `room_inventory` de-duplicates on `room_code`;
  `revenue_*` are groupby aggregations; blocked rows are one-per-vacant-bed.
- **Dates:** `data_loader` coerces date columns; `preprocessing` re-coerces
  defensively (`errors="coerce"`), so invalid dates become `NaT` rather than
  crashing.
- **NaN where a value should exist:** required identity columns (`apartment_code`,
  `room_code`, `bed_code`) are filtered to non-null upstream. Legitimately-empty
  fields remain blank by design: `current_rent`/`estimated_revenue_loss` when a
  room has never had a positive rent; `expected_fill_days` for rooms with no
  refill history (→ `vacancy_status = Unknown`); occupant `gender/occupation/age`
  when the tenant record lacks them (compatibility simply averages fewer factors).

---

## 4. Code quality actions taken

- **Removed** obsolete stub `src/revenue_analysis.py` (fully superseded by
  `revenue_analytics.py` + `blocked_room_detector.py`; no references remained).
- **Removed** unused helpers from `utils.py`: `is_occupied`, `coalesce`,
  `days_between`, and the unused `VACATED_STATUSES` constant.
- **Trimmed** `recommendation_engine._apartment_metrics` to occupancy-only
  (the previously-computed `apartment_revenue` was unused; revenue is excluded
  from ranking by design).
- **Updated** `README.md`: project structure, output-file table, execution order,
  dashboard launch steps, PostgreSQL migration note, roadmap marked complete.
- No unused module-level imports remain (verified per file).
- Retained intentional stubs that are future extension points (not dead):
  `preprocessing.build_current_allocation`, `preprocessing.build_bed_inventory`,
  `occupancy_analysis.apartment_occupancy`.

---

## 5. Final "no hardcoding" verification (static)

| Concern | Result |
|---|---|
| Apartment codes | None in logic (only demo/placeholder strings). |
| Room / bed codes | None; `room_code` is derived once in `preprocessing`. |
| City → State | Only the **documented static fallback dictionary** in `roommate_matching.py`; primary map is learned from tenant data. |
| Occupancy thresholds | None. Vacancy status uses each room's own `expected_fill_days`. |
| Revenue thresholds | None. Leakage is `rent × days/30.44`; excluded from ranking. |
| Fill days | None. Recent/historical averages are computed per room. |
| Demand thresholds | None. Signals are min-max normalised across the current room set. |

**Non-business constants (acceptable):** `30.44` (days/month unit conversion),
`_DEFAULT_CRITICAL_MULTIPLIER = 2.0` (the business rule "Critical ≥ 2× expected",
exposed as an overridable parameter), `_DEFAULT_RECENT_MIN_EVENTS = 1` (event
count, not a day value), demand blend weights (relative, sum to 1), and an age
sanity bound `10–100` (data-quality guard for DOB-derived age, used only by the
display-only compatibility score).

---

## 6. Warnings / caveats

- **Runtime not executed here** — the workspace shell returns no output; all
  runtime checklist items are _pending_ a local run.
- **UUID ↔ text-key gap** — `invoices`/`payments` (UUID) are not directly joined
  to `bookings`/rooms (text keys); revenue for recommendations/leakage uses
  `bookings.monthly_rental` (room/bed level), per design. See
  `docs/RELATIONSHIPS.md`.
- **City→State fallback is not exhaustive** — unknown cities resolve to `None`
  (state shown as "Unknown"); extend `roommate_matching.STATIC_CITY_STATE`.
- **Data-quality flags** — `build_room_inventory` records counts for overbooked
  rooms (capacity clamped to occupied), bed-type/capacity mismatches, and bookings
  dropped for missing bed — surfaced via `inventory.attrs["data_quality"]`.
- **Compatibility is display-only** — computed and shown, but not part of ranking
  until `use_compatibility_in_ranking=True`.

---

## 7. Known assumptions

- A room is identified once, centrally, in `build_room_inventory`; capacity is
  **derived** from observed beds (no bed/room master table exists yet).
- Occupancy = **active bookings** (no exit date + occupied status).
- `current_rent` = the most recent **positive** `monthly_rental` seen in the room.
- Two analytics windows: historical **2023+**, recent **2025+**; pre-2023 data is
  kept for capacity/age only.
- `expected_fill_days` = recent average if enough recent refills, else historical.
- Revenue leakage prorates monthly rent by `30.44` days/month.

---

## 8. Future improvements (not in scope for this phase)

- Implement the DB path in `data_loader.py` (`_load_table_db`) for PostgreSQL/Supabase.
- Add an automated test suite (pytest) covering inventory invariants, vacancy
  classification, and leakage aggregation on a small fixture dataset.
- Resolve the UUID↔text-key bridge to reconcile billed/collected revenue with rooms.
- Expand the static city→state fallback (or plug a geocoding lookup).
- Optional: enable roommate compatibility in ranking once the business wants
  preference matching (`use_compatibility_in_ranking=True`).
- Persist a run timestamp/manifest alongside outputs for dashboard freshness display.
```
