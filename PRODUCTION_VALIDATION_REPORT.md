# Production Validation Report — Availability AI

**Scope:** validation of the migration to the normalized database export (Q81–Q86)
as the production data source, plus the physical-inventory room-universe
enhancement. No business logic, recommendation ranking, blocked-room detection or
revenue-analytics rules were changed. This report documents validation that was
**executed** (pipeline run + all six dashboard pages exercised), not a static-only
review.

---

## 1. Data source — normalized DB export (Q81–Q86)

The pipeline now reconstructs its logical tables from the normalized relational
export, joined inside `data_loader.py` (the only source-aware module):

| Logical table | Reconstructed from |
|---|---|
| `bookings` | Q81 allotments ⋈ Q85 (apartment_code) ⋈ Q83 (bed_code, bed_type) ⋈ Q82 (full_name, phone) ⋈ Q86 (property_name) |
| `beds_master` | Q83 beds ⋈ Q85 (apartment_code, gender_allowed) ⋈ Q84 (active effective-dated rate → current_rate) |
| `current_occupancy` | Q81 open allotments (actual_exit_date null AND staying_status ∈ Staying/On-Notice/Booked) |
| `bed_map` | **Generated entirely from Q81–Q86** — no external CSV. Status/tenant from Q81 open allotments; rate from Q84; days-vacant from last exit date |
| `tenants` | Q82 tenant master (identity + city/state) |

Reconstruction is gated by `_db_export_present()`. When Q81/Q83/Q85 are absent the
loader falls back to the legacy flattened CSV path — **backward compatible**.

---

## 2. Room universe — physical inventory ∪ bookings

`build_room_inventory` seeds the room universe from the physical bed catalog
UNIONED with booking history (`_seed_physical_beds`), so newly-created Live
apartments with no bookings still appear. Seeded beds carry empty history
(never `_active`) → occupied 0, available = live beds, occupancy 0%, tenant/days
null, `current_rate` from Q84 (NULL when the source has no rate). Existing
booking-driven rooms are untouched.

---

## 3. Pipeline execution (run, exit 0)

```
python src/preprocessing.py            OK
python src/occupancy_analysis.py       OK
python src/blocked_room_detector.py    OK
python src/revenue_analytics.py        OK
python src/recommendation_engine.py    OK
```

Only benign `pd.to_datetime` "could not infer format" UserWarnings; no tracebacks.
All 8 `outputs/*.csv` regenerated non-empty.

---

## 4. Dashboard validation — all six pages (executed)

Built via `dashboard/app.get_live_analytics()` (the single shared bundle every
page consumes):

| Page | Result |
|---|---|
| **1. Inventory** | 115 rooms · 38 apartments · capacity 203 · occupied 177 · available 26. Never-booked Live rooms **A33** (2) and **A34** (1) now present. |
| **2. Room Search** | 21 active vacant beds (includes A33/A34 beds). |
| **3. Customer Recommendation** | Chennai/Male → detected state Tamil Nadu, Top-5; A33-B, A33-C, A34-TS in candidate pool. |
| **4. Occupancy Analytics** | Portfolio occupancy 87.19% (177/203); occupancy history 1101 rows. |
| **5. Blocked Rooms** | 6 blocked (partially-occupied) rooms; vacancy_status classification unchanged. |
| **6. Revenue Leakage** | Total estimated leakage ₹38,731.93 over 6 vacant beds; highest-loss room C34 / C34-C / C1. |

Every page loads and computes without error.

---

## 5. Schema compatibility

Reconstructed frames preserve the pre-migration accessor schemas. `bookings`
carries all `require_columns` fields (apartment_code, bed_code, bed_type,
onboarding_date, actual_exit_date [datetime], monthly_rental, staying_status);
`beds_master` carries the `_BEDS_MASTER_KEEP` set (apartment_code, bed_code,
gender_allowed, bed_status, bed_lifecycle_status, toilet_type, current_rate);
`current_occupancy` carries occupancy_status + staying_status + keys. Downstream
modules run unchanged.

---

## 6. Functional parity (DB export vs previous production run)

Comparison of the DB-export bundle against the legacy flattened bundle on the same
snapshot:

- **Identical**: room counts, per-room capacity / occupied / available / occupancy%,
  recent & historical occupancy, demand scores, occupancy history, blocked-room set
  and classification, and revenue-leakage KPIs — apart from the intentionally added
  never-booked rooms.
- **Bed-map vs prior Vishful export** (bed-map-2026-07-23.csv): rate 100% match;
  days-vacant exact on every commonly-vacant bed; status/tenant reconciled to Q81
  open allotments (the more accurate source).
- **Expected differences** (not regressions): portfolio occupancy diluted from the
  added physical capacity; a small set of status/tenant beds reflect source
  divergence (see §7).

---

## 7. Remaining data-quality issues — source database, not code

1. **Q84 rate-card gap** — no `(Triple, Common)` rate row → 4 beds have
   `current_rate = NULL` (A33-C1/C2/C3, A34-TS2). NULL is preserved by design; no
   value is invented. Fix at source (add the rate to Q84).
2. **Q81 vs live-app divergence** — ~7 status + 4 tenant beds where the allotment
   state differs from the live application, consistent with un-exported bed
   transfers (Q56 is not part of Q81–Q86). Reconcile at source.
3. **Future-dated bookings** — a `Booked` allotment with a future onboarding date
   is treated as held; the live app shows such beds as vacant until move-in. This
   is a business-rule choice, unchanged here.

None originate in the loader or pipeline code.

---

## 8. No-hardcoding verification

| Concern | Result |
|---|---|
| Apartment / room / bed codes | None in logic; `room_code` derived once in `preprocessing`. |
| Rates | From the Q84 effective-dated rate card only; NULL when absent (never defaulted). |
| City → State | Learned from tenant data first; documented static fallback in `roommate_matching`. |
| Occupancy / vacancy thresholds | None; per-room `expected_fill_days`. |
| Demand thresholds | None; signals min-max normalised across booked rooms. |

Non-business constants unchanged: `30.44` days/month, Critical ≥ 2× expected
(overridable), demand blend weights, age sanity bound 10–100.

---

## 9. Production readiness

**READY.** Migration to Q81–Q86 is the production baseline: pipeline runs clean,
all six dashboard pages verified, schemas preserved, functional parity confirmed
(new rooms aside), backward-compatible legacy path retained. Two source-side data
items (§7.1 Q84 Triple/Common rate, §7.2 Q56 transfers) are owner actions at the
database, not code defects.

---

## 10. Repository cleanup (this pass)

Removed diagnostic/debug scripts and generated audit artifacts:
`src/_audit_datasources.py`, `src/_debug_recommendation_pipeline.py`,
`outputs/_AUDIT_STATUS.md`, `outputs/_dataloader_path_audit.json`,
`outputs/_dataset_audit.json`, `outputs/_dup_bed_audit.json`,
`outputs/_dup_bed_audit_compact.json`,
`outputs/_recommendation_validation_audit.json`, `outputs/_val_stderr.txt`, and all
`__pycache__/` folders. Production source, documentation, config, `requirements.txt`,
`outputs/.gitkeep` and the `outputs/*.csv` demo files are retained.
