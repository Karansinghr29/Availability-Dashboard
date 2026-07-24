# Availability AI — Project Handover

**Prepared for:** Owner / Business
**Date:** 23 July 2026
**Status:** Feature-complete and validated on live Vishful data (CSV). Ready for
day-to-day use; direct production-database connection is the next step.

---

## 1. What the system does (implemented features)

| Feature | What it delivers | Status |
|---|---|---|
| **Smart availability recommendation** | For any customer enquiry it returns the best available rooms (a room + a specific vacant bed), each with a plain-English reason. | ✅ Live |
| **Same-state tenant matching** | Prioritises rooms where existing occupants are from the customer's home state, so new tenants are placed with compatible roommates. | ✅ Live |
| **Occupancy-based ranking** | When no same-state match exists, it fills the emptiest apartments/rooms first — good for business occupancy. | ✅ Live |
| **Bed-type fallback** | If the requested bed type (e.g. Triple) is unavailable, it automatically offers the nearest available type (e.g. Double) and says so. | ✅ Live |
| **Blocked-room detection** | Flags beds staying vacant far longer than that room's own historical refill time — a "possible occupancy block" signal (Normal / Delayed / Critical). | ✅ Live |
| **Revenue-leakage identification** | Estimates rupees being lost to each vacant bed and rolls it up to apartment and portfolio level. | ✅ Live |

The customer is always shown a **real room and bed with the correct rent** —
apartment-level occupancy/revenue are used only internally to decide ranking.

---

## 2. New Vishful datasets integrated

Three newer production exports were integrated to improve accuracy. **Nothing was
blindly replaced** — older files remain available as fallbacks.

| Dataset | Role | Why it matters |
|---|---|---|
| **Q50 — Current Occupancy** | Now the authoritative "who is in which bed right now" source (replaces the older Q36). | Fuller, consistent live occupancy; removed contradictory "occupied-yet-vacant" rows. |
| **Q42 + Q32 — Tenant information** | Q42 is the master tenant identity (stable tenant ID + demographics); Q32 backfills home city/state where Q42 is blank. | Reliable identity **and** complete location data → better state matching. |
| **Q56 — Transfer history** | Records bed-to-bed transfers; used **only** to keep the occupancy timeline accurate. | Improves data quality without changing any blocked-room or leakage rule. |

Deliberately **left out** (not needed for these features): payments, invoice line
items, electricity, and settlement tables.

---

## 3. How data flows (architecture)

```
   Vishful data (CSV today → database later)
                │
                ▼
        DataLoader            ← the ONLY part that knows where data comes from
   (picks the right file per table; merges Q42+Q32; bridges Q56 IDs)
                │  clean, standard tables
                ▼
        Preprocessing         ← builds the "Room Master" (single source of truth)
                │                and the per-bed occupancy timeline
        ┌───────┴────────┐
        ▼                ▼
 Recommendation      Blocked-room + Revenue-leakage
   engine               analytics
        └───────┬────────┘
                ▼
           Dashboard          ← rebuilds everything live from the data;
   (Recommendation, Blocked Rooms, Revenue, Occupancy, Room Search pages)
```

**Key design principle:** only the DataLoader knows the data source. When the
Vishful database replaces the CSV files, **only that one component changes** — the
recommendation, analytics and dashboard keep working unchanged.

---

## 4. Validation results

An independent validation was run against the live data covering all requested
features. **Result: 34 / 34 checks passed.**

| Area | Checks | Outcome |
|---|---|---|
| Smart recommendation | city→state (9 cities), same-state priority, ranking order, bed-type fallback, bed codes, rent values | ✅ All pass |
| Blocked room / leakage | one-bed-remaining flag, refill-time vs vacancy, long-unfilled detection, leakage formula, Q56 does not change rules | ✅ All pass |
| Data sources | Q50 occupancy, Q42+Q32 tenants, Q56 bridge (14/14 transfers resolved), no old file overriding | ✅ All pass |
| Regression | recommendation & blocked-room behaviour preserved, dashboard runs live | ✅ All pass |

Highlights: city→state correct for all tested cities; same-state rooms always
ranked ahead of others; rent shown on every card matches the official bed rate;
revenue-leakage maths unchanged; the Q50 upgrade removed a false vacancy that had
previously inflated leakage figures.

---

## 5. Current limitations

1. **CSV-based loader.** The system currently reads Vishful CSV exports. It is
   built to switch to the live database with a single-component change, but that
   switch is **not yet implemented** (the database connector is a pending stub).
2. **Database connector pending.** Direct, always-live connection to
   Postgres/Supabase is the next engineering step.
3. **Remaining data-quality items (in the source data, not the software):**
   - One bed (A44-C2) is marked occupied in the live occupancy file but exited in
     the booking history — a source disagreement needing a business call on which
     record wins.
   - One tenant appears in two beds at once (an unclosed old booking in the source).
   - A few older transfers can't be matched to a booking exit date (cosmetic only;
     no effect on numbers).
   These are Vishful data-entry matters; the software handles them safely today.
4. **No automated test suite yet** and **no forecasting** (forecasting was never in
   scope for this phase).

---

## 6. Production readiness

**Ready to use now (CSV-fed):**
- Customer recommendation, same-state matching, occupancy ranking, bed-type fallback.
- Blocked-room detection and revenue-leakage reporting.
- The live dashboard (all five pages) rebuilding from the latest Vishful exports.

**Pending before a direct production-database go-live:**
- Implement the database connector (the prepared single-component swap).
- Add an automated test suite so future data/code changes are checked automatically.
- Business decision on the occupancy vs booking-history source-of-truth (the A44
  case above).

**Bottom line:** every feature you requested is working and independently validated
on real data. The system can go live today on the CSV exports, and is intentionally
built so that connecting it straight to the Vishful production database is a small,
well-contained next step rather than a rewrite.

---

*Supporting documents in this project: `README.md` (design), `docs/DATA_DICTIONARY.md`
and `docs/RELATIONSHIPS.md` (data), `PRODUCTION_VALIDATION_REPORT.md` (technical
validation), and the regenerated audit files under `outputs/` with
`outputs/_AUDIT_STATUS.md` listing which are current.*
