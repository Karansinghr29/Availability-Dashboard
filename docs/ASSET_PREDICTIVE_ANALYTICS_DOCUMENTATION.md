# Asset Predictive Analytics — Technical & Business Documentation

**Scope:** the "Asset Predictive Analytics" page (`dashboard/app.py → page_asset_predictive`) and its engine (`src/asset_engine.py`), plus the presentation-layer room view built inside `app.py`.
**Audience:** company owner, developers, future data-science team.
**Status:** describes *current behaviour only*. No improvements suggested. Statistical/rule-based engine — **not** machine learning. Read-only — no database records are written.

---

## PART 0 — FOUNDATIONS (shared by every section)

Every section below reuses one cached build. Reading these foundations once makes each section shorter.

### 0.1 Source tables (loaded via `DataLoader`)

| Table (loader method) | Columns used | Why |
|---|---|---|
| `maintenance_tickets` (q18) | `id`, `ticket_number`, `apartment_id`, `bed_id`, `issue_type_id`, `asset_id`, `created_at`, `resolved_at`, `sla_deadline`, `status`, `assigned_to`, `closure_cost` | The event log. Every prediction starts from these tickets. |
| `asset_master` | `id`, `asset_code`, `asset_type_id`, `brand`, `model`, `purchase_date`, `warranty_expiry` | Physical assets, their identity, brand, and age anchor (purchase date). |
| `asset_types` | `id`, `name`, `expected_life_months`, `maintenance_cycle_months` | Per-type lifecycle & service-cycle constants used in age-ratio and maintenance-due maths. |
| `asset_allocations` | `asset_id`, `apartment_id`, `bed_id`, `allocated_date` | Where each asset physically sits; earliest allocation date is the fallback age anchor. |
| `beds_master_uuid` | `id`, `apartment_id`, `bed_code` | Resolves a ticket's `bed_id` → human room label (`bed_code`) and the bed's apartment. |
| `apartment_master` | `id`, `apartment_code`, `floor_number` | Human apartment codes; floor for hotspot density. |
| `issue_types` | `id`, `name` | Human issue names; the issue→asset-type bridge. |
| `maintenance_items` | `issue_type_id`, `asset_type_id` | The bridge that says "issue X can be caused by asset-type Y". |
| `ticket_resolutions` | `total_cost` | Fallback average ticket cost when `closure_cost` is null. |

### 0.2 Ticket → asset mapping (`_map_tickets`) and confidence

For every ticket, exactly one confidence label is assigned:

- **Verified** — the ticket already carries a real `asset_id`. Used as-is.
- Otherwise the engine builds a *candidate pool* = assets allocated to the ticket's `bed_id` (else its `apartment_id`), filtered to asset-types that the ticket's issue type can map to (`maintenance_items`):
  - exactly **1** candidate → **High**
  - **2–3** candidates → **Medium** (first candidate used)
  - **0 or >3** candidates → **Low** (no asset assigned)

**Exact rule (from code):**
```
if asset_id present:           confidence = Verified
elif 1 candidate:              confidence = High
elif 2..3 candidates:          confidence = Medium
else:                          confidence = Low  (asset_id = "")
```
Live counts: Verified 266, High 213, Medium 69, Low 957 (of 1,505 tickets).

### 0.3 Which tickets are scored into assets (`build`)

Asset-level scoring uses **only Verified + High** tickets (479 tickets), grouped by `asset_id` → **251 unique assets**. Medium and Low tickets never enter per-asset scoring. Room-level and asset-type-level views (Sections 4, 6, 12) use **all** tickets that carry an apartment or a mappable issue.

### 0.4 Age anchor (per asset)
`purchase_date` if present → age source **"Purchase Date"**; else earliest `allocated_date` → **"Allocation Date"**; else none → **"Failure History"**. `age_months = age_days / 30.44`.

### 0.5 Core per-asset formulas (`_score`) — plain English

- **ticket_count** = number of Verified/High tickets pinned to that asset.
- **recent_30d** = of those, how many in the last 30 days.
- **repeat_count** = number of issue types that occurred ≥2 times for that asset.
- **age_ratio** = age_months ÷ expected_life_months (blank if either missing).
- **Health score (0–100)** = `100 − penalty`, penalty capped per factor:
  - age: `min(35 × age_ratio, 35)`
  - tickets: `min(8 × ticket_count, 32)`
  - recent: `min(12 × recent_30d, 24)`
  - overdue service: `+15` if past maintenance cycle
  - repeats: `min(6 × repeat_count, 18)`
- **Risk level**: Critical if health < 35 **or** (repeat ≥ 3 and recent ≥ 2); else High if < 55; Medium if < 75; else Low.
- **Failure probability (30 days)** = Poisson: `exposure = max(days since first ticket, 30)`; `λ = min(ticket_count / exposure, 0.2)`; `prob = 100 × (1 − e^(−λ×30))`. The `max(.,30)` floor and `0.2` cap damp bursts of same-incident re-tickets.
- **maintenance_due_days** = `maintenance_cycle_months × 30.44 − days_since_last_ticket`. Negative ⇒ overdue.
- **replacement_recommended** = age_ratio ≥ 1.0 **or** (ticket_count ≥ 6 and repeat ≥ 2) **or** health < 25.
- **failure_trend** (needs ≥3 tickets): compares ticket rate in the later half vs earlier half → Rapidly Degrading (>2.5×), Degrading (>1.3×), Improving (<0.6×), else Stable.
- **expected_cost_1y** = `λ × 365 × avg_ticket_cost`.
- **avg_ticket_cost** = mean `closure_cost` (currently all null) → mean `ticket_resolutions.total_cost` → else ₹500 default.

### 0.6 Global data flow
```
maintenance_tickets
   ↓ (bed_id → beds_master → bed_code/apartment ; issue_type_id → maintenance_items → asset_type)
_map_tickets  → confidence (Verified/High/Medium/Low)
   ↓ (keep Verified+High, group by asset_id)
build → per-asset scores (health, risk, prob, due, replacement, trend)
   ↓
business_views / coverage / room_intelligence / asset_type_intelligence / executive / sla / brand / purchase
   ↓
_ticket_room_view (app.py) — room-first re-grouping by ticket bed_id
   ↓
dashboard render
```

### 0.7 Prediction method summary (applies to all predictive sections)
**Used:** purchase date, allocation date (fallback), expected life, maintenance cycle, repeat failures, recent failures, ticket frequency, issue history (issue→type bridge), room history (per-apartment/per-bed ticket grouping), asset history (per-asset ticket gaps).
**Not used:** any machine-learning model, weather/seasonality inputs, tenant behaviour, external sensors, real repair-cost data (all `closure_cost` null → cost is an estimate), vendor SLAs as a prediction input.

---

## SECTION 1 — Maintenance Health Score (hero card)

**1. Business Purpose** — One headline number the owner sees first: overall maintenance condition of the whole portfolio. Answers "is my property estate healthy or in trouble right now?"

**2. Data Sources** — Derived, not a table. Uses `built["assets"].health_score` (per-asset health from `_score`) and `coverage()` totals for the side stats. Columns required: `health_score` (the average), `total_tickets`, `total_assets`, `coverage_pct` (context stats).

**3. Data Flow** — `maintenance_tickets → build → per-asset health_score → mean() → hero`. Side stats come from `coverage()`.

**4. Logic Used** — Score = **average of every scored asset's health score**, rounded to a whole number. Status band: **≥75 Healthy (green), 60–74 Attention Needed (amber), <60 Critical (red)**. No new model — it is the mean of existing health scores.

**5. Prediction Method** — Not a prediction; an aggregate of existing health scores (which themselves use age, tickets, recency, repeats, overdue). Uses asset history + age anchors; does not use SLA, vendor, or brand.

**6. Confidence** — Not applicable (aggregate KPI, no confidence label).

**7. Output** — `79 / 100 · 🟢 Healthy`. It is `round(mean(health_score))` over the 251 scored assets (mean ≈ 79.4 → 79). Side tiles: **Total Tickets 1,505**, **Total Assets 1,700**, **Coverage % 31.8%**. Caption explains age sourcing (purchase vs allocation) and processed-ticket ratio.

**8. Charts** — None (single hero card).

**9. Tables** — None.

**10. Business Meaning** — Owner: one-glance portfolio health and trend band. Ops manager: whether to escalate this month. Maintenance team: context, not a work item.

**11. Limitations** — Average is over the **251 scored assets only** (Verified+High), not all 1,700 assets. Assets with no Verified/High ticket are absent from the mean. Health depends on age, and 232/251 scored assets are aged from allocation date (purchase date missing) so their age — and thus health — can be understated.

---

## SECTION 2 — Engine Coverage

**1. Business Purpose** — Honesty panel: how much of the maintenance history the engine can actually act on, and how assets are aged. Owner wants to know how much to trust the predictions.

**2. Data Sources** — `coverage()` over `maintenance_tickets` + `build` KPIs.
- `maintenance_tickets.apartment_id`, `issue_type_id` — to decide if a ticket is "processable".
- `maintenance_items` (issue→type bridge) — a ticket is mappable if its issue type has a known asset type.
- `asset_master` count, `asset_allocations` (allocated count), age-source counts from `build`.

**3. Data Flow** — `maintenance_tickets → (has apartment OR issue mappable?) → processed count`; `build KPIs → asset/allocation/age-source counts`.

**4. Logic Used** —
- **tickets_processed** = tickets where `apartment_id` present **or** issue type is mappable.
- **coverage_pct** = processed ÷ total × 100.
- **asset_coverage_pct** = (Verified + High) ÷ total × 100.
- **assets_with_purchase_date / assets_using_allocation_date** = counts of scored assets by age source.

**5. Prediction Method** — None (descriptive coverage).

**6. Confidence** — Reports the confidence mix indirectly (asset-pinned = Verified+High). No new confidence assigned here.

**7. Output** — **Total Tickets 1,505**; **Total Assets 1,700**; **Allocated Assets 1,528**; **Coverage % 31.8%** (479 asset-pinned ÷ 1,505). Caption: 308 assets aged by purchase date, remainder by allocation date; "1,505/1,505 tickets processed" style line.

**8. Charts** — None.

**9. Tables** — None (KPI cards + caption).

**10. Business Meaning** — Owner: understand that ~32% of tickets pin to a specific asset; the rest still inform room/type views. Developer: sanity metric after any data reload. Ops: knows asset-level precision is partial.

**11. Limitations** — Coverage % measures ticket *participation*, not accuracy. Only 266 tickets carry a real `asset_id`; the rest are inferred or unmapped. Purchase-date coverage is low (308/1,700 = 18%).

---

## SECTION 3 — Asset Health & Risk

**1. Business Purpose** — Portfolio risk distribution at a glance: how many assets are healthy vs at risk, and how many need replacement. Owner wants the size of the problem.

**2. Data Sources** — `build` KPIs (`healthy`, `medium_risk`, `high_risk`, `critical`, `due_replacement`) from per-asset `risk_level` and `replacement_recommended`.

**3. Data Flow** — `build → per-asset risk_level & replacement flag → counts → cards`.

**4. Logic Used** — Simple counts of assets by risk band (see 0.5) and count where `replacement_recommended` is true.

**5. Prediction Method** — Uses the per-asset risk model (age, tickets, recency, repeats, overdue). Not SLA/vendor/brand.

**6. Confidence** — Not applicable (counts of already-scored assets).

**7. Output** — One grouped card "Asset Risk Distribution: 🟢 Healthy N · 🟡 Medium N · 🟠 High N · 🔴 Critical N" and a "Replacement Due" card. Each number is a straight count over the 251 scored assets.

**8. Charts** — None.

**9. Tables** — None.

**10. Business Meaning** — Owner: budget sizing (how many criticals/replacements). Ops: staffing for the month. Team: not directly actionable (see Preventive Queue / Room Intelligence).

**11. Limitations** — Counts cover scored assets only (251), not all 1,700. Risk depends on age accuracy (allocation-date fallback).

---

## SECTION 4 — Room Intelligence (room-first, Apartment-Bed)

This is the primary operational section. Room = the **ticket's own `bed_id`** resolved to `bed_code` (the room the technician selected), **not** the asset's current allocation. Built by `_ticket_room_view` in `app.py`.

**1. Business Purpose** — Vishful raises tickets against a tenant's room (Apartment-Bed), not an asset. This section answers: which room complains most, which issues recur there, which asset is the likely cause, how likely it is to fail again, and what to inspect first — so a technician fixes the root cause in one visit.

**2. Data Sources** —
- `maintenance_tickets`: `bed_id` (authoritative room), `apartment_id` (fallback), `issue_type_id`, `created_at` — room grouping, issue counts, recency.
- `beds_master_uuid`: `bed_id → bed_code` and `bed_id → apartment_id` — the room label.
- `apartment_master`: `apartment_code`.
- `issue_types`: issue names.
- `business_views` asset superset (per-asset scores + `avg_interval_days`) — matched to the room by `(apartment_code, bed_code)` to attach the prediction.

**3. Data Flow** —
```
maintenance_tickets
   ↓ bed_id
beds_master → bed_code (Room) + apartment_id
   ↓ apartment_master → apartment_code
group tickets by (apartment_code, bed_code)   ← authoritative room
   ↓ match assets where asset.apartment==room.apartment AND asset.bed_code==room.bed
     (fallback: any asset in the apartment)
attach engine per-asset scores (prob, health, repeat, age, due, trend)
   ↓
Room Intelligence table + drill-down
```

**4. Logic Used** (per room) —
- **Total tickets** = count of tickets whose `bed_id` resolves to this room.
- **Distinct issues** = number of distinct issue types in the room.
- **Similar issue tickets** = count of the single most-frequent issue type.
- **Last 30/90 days** = tickets in those windows (`created_at`, tz-normalised).
- **Predicted asset** = the room's matched asset with the **highest `failure_prob_30d`**.
- **Failure probability / repeat failures / historical asset tickets / last maintenance / age / trend** = that predicted asset's engine values.
- **Expected next maintenance** = `maintenance_due_days` phrased "Within N days" / "Overdue".
- **Estimated failure window** = `last_ticket + avg_interval_days` (from business_views) → a date; shown only when that interval exists, else "—".
- **Inspection Priority** = every asset in the room sorted by `failure_prob_30d` desc, each with a short reason.
- **Recommendation** (predictive wording) = "This room has a high probability of future {top asset} failure. During the next maintenance visit, also inspect {others with prob≥50% or High/Critical} to reduce repeat visits."
- **Business Impact** = Likely repeat ticket (prob≥60%), Estimated effort (High/Medium/Low by count of High/Critical assets: ≥3/=2/else), Visit priority (High if Critical/High else risk band), Preventive recommended (Yes unless risk Low and not due).
- **Room-risk composite** (presentation ranking only): `min(total,20)×1.5 + min(repeat,5)×6 + prob×0.25 + max(0,100−health)×0.15 + min(recent30,5)×2` → Critical≥70 / High≥50 / Medium≥30 / Low.

**5. Prediction Method** — Room from ticket `bed_id` (authoritative), asset-in-room from allocation (only asset↔bed link available). Uses: room ticket history, issue history, per-asset history, repeat & recent failures, age (purchase/allocation), expected life, maintenance cycle, avg failure interval. Does not use ML, SLA, vendor, or brand.

**6. Confidence** (per room) — **High** = a scored asset sits at that exact ticket-selected bed. **Medium** = no bed match; fell back to an apartment-level asset. **Low** = no scored asset at all (prediction shown from ticket history only). (This is a room-level rollup of the underlying Verified/High mapping confidence from 0.2.)

**7. Output** — Example room **A41-C1**: Total 36 · Distinct 9 · Most frequent Electrical Issues · Predicted Air Conditioner (VH-AC-A41-0003) · 99.8% · High confidence. Numbers: 36 from tickets on that bed; 99.8% is the AC asset's Poisson prob; "Within 167 days" from its maintenance-due days.
Section KPIs: **Inspect Today** (rooms with Critical/overdue top asset), **Preventive This Week** (due ≤7d or High), **Rooms w/ Multiple High-Risk Assets** (≥2 High/Critical), **Likely Another Ticket (30d)** (top prob ≥60%). Plus Rooms with Tickets, Rooms Active (30d), High-Confidence Predictions.
"Today's Priority Rooms" (top of section) = Top 10 rooms by the room-risk composite.

**8. Charts** — "Ticket Volume by Room (Apartment-Bed, top 15)": x-axis = room label (`A41-C1`), y-axis = total tickets, grouping = per room, aggregation = count of tickets.

**9. Tables** —
- *Today's Priority Rooms*: Rank (by composite), Room, Risk (composite band), Predicted Asset, Probability, Repeats, Recent (30d), Total Tickets, Visit Priority.
- *Room Intelligence*: Room (Apt-Bed), Apartment, Bed, Total Tickets, Distinct Issues, Most Frequent Issue, Predicted Asset, Asset Code, Brand, Failure Probability, Historical Asset Tickets, Last Maintenance, Expected Next Maintenance, Confidence, Recommendation. Every column's source is listed in point 4.
- Drill-down (per room): Historical Tickets, Issues raised breakdown, Prediction Evidence (historical/similar/repeat/recent/age/trend/probability/confidence), Prediction Timeline (last maintenance, expected next maintenance window, estimated failure window), Inspection Priority list, Recommendation, Business Impact.

**10. Business Meaning** — Team: go room-first; fix the top inspection-priority asset and check the others listed to avoid a return visit. Ops manager: use "Inspect Today"/"This Week" KPIs and Today's Priority Rooms to plan the day. Owner: see which rooms drive cost.

**11. Limitations** — When no asset is allocated to the exact bed, the prediction falls back to apartment-level assets (Medium confidence) and the inspection list becomes apartment-wide. Estimated failure window is "—" unless the asset has ≥2 asset-pinned tickets (needed for an interval). Room grouping only covers tickets whose `bed_id` resolves in `beds_master` (1,496/1,505). Asset location comes from current allocation, which can differ from where the asset was when the ticket was raised.

---

## SECTION 5 — Room Maintenance Risk Ranking

**1. Business Purpose** — A single prioritised worklist of every room for the supervisor: which rooms to visit first and how much effort each needs.

**2. Data Sources** — Same room view as Section 4 (`_ticket_room_view`); no extra tables.

**3. Data Flow** — Room view → sort by the room-risk composite → rank.

**4. Logic Used** — Sort by `risk_score` (the composite in 4.4) desc, tie-break by probability. **High-Risk Assets in Room** = count of that room's assets at High/Critical. **Estimated Maintenance Effort** = High (≥3), Medium (=2), Low (otherwise) of those high-risk assets.

**5. Prediction Method** — Reuses room-view predictions; the composite is presentation-only prioritisation, it does **not** alter any engine score.

**6. Confidence** — Inherits each room's confidence from Section 4.

**7. Output** — Cards: Critical Rooms / High-Risk Rooms / Medium-Risk Rooms counts (live: 4 / 39 / 79). Table rows ranked 1..N. Example: `1 · B13-B1 · Critical · Air Conditioner · 99.3%`.

**8. Charts** — None.

**9. Tables** — Rank, Room, Risk, Predicted Asset, Probability, Total Tickets, High-Risk Assets in Room, Estimated Maintenance Effort, Recommendation. Sources per points 4/4.4.

**10. Business Meaning** — Supervisor: top-down visit order + effort estimate for scheduling. Ops: staffing. Owner: concentration of risk.

**11. Limitations** — The composite is a presentation heuristic (fixed weights), not a validated risk model. Effort counts high-risk assets from the matched set (apartment-wide when bed match fails).

---

## SECTION 6 — Asset-Type Intelligence

**1. Business Purpose** — Which *categories* of asset (AC, Fan, RO…) fail most across the estate, and which are ageing. Owner wants category-level procurement and reliability insight.

**2. Data Sources** — `asset_type_intelligence()`.
- `maintenance_tickets.issue_type_id` + `maintenance_items` bridge → failure frequency per asset type (all tickets).
- `asset_master` (+`asset_types`) → count per type, expected life, purchase date.
- `asset_allocations` → allocation-date age fallback.
- `build` assets → avg tickets/asset, avg health; `mapped` gaps → avg maintenance interval per type.

**3. Data Flow** — `tickets → issue→type bridge → per-type failure counts`; `assets → per-type age/health`; combined per type.

**4. Logic Used** — **failure_frequency** = number of issue-tickets whose issue maps to that type. **avg_age_months** = mean age of that type's assets. **age_ratio** = avg age ÷ expected life. **approaching_end_of_life** = age_ratio ≥ 0.8. **high_risk** = avg_health < 55 **or** failure_frequency ≥ 100. **avg_maintenance_interval_days** = mean gap between consecutive asset-pinned tickets for that type.

**5. Prediction Method** — Descriptive/aggregate. Uses issue history, asset age, expected life, ticket frequency. One issue ticket can count toward several asset types if the issue maps to several types.

**6. Confidence** — Not applicable (aggregate). Failure frequency uses all tickets via the issue→type bridge (not the per-ticket confidence).

**7. Output** — "High-Risk Asset Types" count card. Detail rows per type with the columns above.

**8. Charts** — "Failure Frequency by Asset Type": x = asset_type, y = failure_frequency, aggregation = count via issue→type bridge, top 15.

**9. Tables** — asset_type, assets (count), failure_frequency, avg_age_months, avg_tickets_per_asset, avg_maintenance_interval_days, avg_health, approaching_end_of_life, high_risk. Sources per point 4.

**10. Business Meaning** — Owner/procurement: which categories cost most and age fastest. Ops: category-level preventive planning. Team: context.

**11. Limitations** — Failure frequency double-counts when an issue maps to multiple asset types. Age uses allocation fallback. Interval needs ≥2 asset-pinned tickets, so sparse types show blank.

---

## SECTION 7 — Owner Action List (formerly "Executive Dashboard")

> Renamed to **Owner Action List**. It now leads with a Top-15 action table **ranked by Composite Maintenance Priority** (the same room-risk composite used by the Room Maintenance Risk Ranking), followed by the three summary cards and supporting lists (each labelled with its own ranking metric). A **Today's Priority Rooms** table (Top 10 by Composite Room Risk — Room, Predicted Asset, Failure Probability, Recommended Action) now sits at the very top of the page, immediately under the Maintenance Health Score hero. Every section on the page now shows an explicit "**Ranked by:** …" label so apartments/rooms recurring across sections are explained by their metric (Average Apartment Health vs Recent Ticket Volume vs Composite Risk vs Failure Probability), not treated as inconsistent.

**1. Business Purpose** — Management summary + prioritised action list: the single worst apartment, the worst asset category, how many assets are about to fail, and the ranked "do this first" worklist — the "what matters this month / act now" view.

**2. Data Sources** — `executive()` over `room_intelligence`, `asset_type_intelligence`, `build` assets, and `mapped` dates.

**3. Data Flow** — `room_intel (top) + type_intel (top) + assets(prob≥60) + mapped(recent counts) → cards + tables`.

**4. Logic Used** — **likely_fail_30d** = assets with `failure_prob_30d ≥ 60`. **Top Risk Apartment** = highest room-risk apartment (from room_intelligence). **Top Failing Asset Type** = highest failure_frequency type. (Insights text and a 30-day workload forecast — `last30` and `last30×√(last30/prev30)` — are computed in the engine; the current page surfaces the three cards + tables.)

**5. Prediction Method** — Uses failure probability (assets), room history, issue history. Does not use ML/SLA/vendor.

**6. Confidence** — Not applicable at card level (aggregates of already-scored data).

**7. Output** — Three insight cards: 🔴 Top Risk Apartment (with its focus asset), 🟠 Top Failing Asset Type, 🟡 Assets Likely to Fail (30 Days) = count of prob≥60 assets. Tables in expanders.

**8. Charts** — None (cards + tables).

**9. Tables** — *Top Risk Apartments* (apartment_code, total_tickets, demand_trend, room_risk_level + predicted asset/location/prob/reason attached in presentation), *Top Failing Asset Types* (asset_type, failure_frequency, high_risk, approaching_end_of_life), *Assets Likely to Fail Next 30 Days* (asset_code, asset_type, location, failure_prob_30d, risk_level, recommendation).

**10. Business Meaning** — Owner: the headline risks. Ops: focus areas. Team: not the work list.

**11. Limitations** — "Likely to fail" threshold is a fixed 60% on a capped Poisson probability; it flags relative risk, not a calibrated failure forecast.

---

## SECTION 8 — Predictive Visualizations

**1. Business Purpose** — Visual shape of the portfolio: health spread, failure trend over time, and which categories fail most.

**2. Data Sources** — `charts()` over `build` assets and `mapped` tickets.

**3. Data Flow** — `assets → health buckets & type sums`; `mapped → monthly counts`.

**4. Logic Used** — **health_distribution**: assets bucketed into 0-35 / 35-55 / 55-75 / 75-100. **monthly_trend**: asset-pinned tickets grouped by calendar month. **failure_by_type**: sum of ticket_count per asset type (top 20).

**5. Prediction Method** — Descriptive (no prediction).

**6. Confidence** — Not applicable.

**7. Output** — Three charts + a "Top Frequently-Failing Assets" table.

**8. Charts** —
- Asset Health Distribution: x = health bucket, y = asset count.
- Monthly Failure Trend: x = month, y = ticket count (asset-pinned).
- Failures by Asset Type: x = asset_type, y = summed ticket_count.

**9. Tables** — Top Frequently-Failing Assets: asset_code, asset_type, location, ticket_count, risk_level.

**10. Business Meaning** — Owner: trend direction. Ops: seasonality/peaks. Team: worst assets.

**11. Limitations** — Monthly trend and health buckets use asset-pinned tickets/assets only (Verified+High).

---

## SECTION 9 — Predictive Recommendations

**1. Business Purpose** — The engine's per-asset action list: which assets need action and what action.

**2. Data Sources** — `build` assets (`recommendation`, `risk_level`, `failure_prob_30d`, `age_months`, `reason`, location).

**3. Data Flow** — `assets → filter recommendation ≠ "No Action Needed" → sorted by prob`.

**4. Logic Used** — Recommendation rule (see 0.5): Replace Asset / Repair Immediately / Schedule Maintenance / Monitor Closely / No Action Needed. Count cards = value counts of recommendation.

**5. Prediction Method** — Uses risk level, recency, overdue, replacement flag, trend. Not SLA/vendor.

**6. Confidence** — Assets shown are Verified/High-mapped (only those are scored). No separate confidence column here.

**7. Output** — Count card per recommendation type; a table of assets needing action.

**8. Charts** — None.

**9. Tables** — asset_code, asset_type, location (Apt/Bed), risk_level, recommendation, failure_prob_30d, age_months, reason.

**10. Business Meaning** — Team: direct work list. Ops: volume per action type. Owner: replace vs repair split.

**11. Limitations** — Covers 251 scored assets only. Recommendation is rule-based, thresholds fixed.

---

## SECTION 10 — Active Alerts

**1. Business Purpose** — Exception list: assets tripping specific alert conditions that need attention now.

**2. Data Sources** — `build` assets `alerts` string (from `_alerts`).

**3. Data Flow** — `assets → rows where alerts text is non-empty`.

**4. Logic Used** — Alerts raised (from `_alerts`): "Maintenance Overdue" (past cycle); "Immediate Maintenance Required" (High/Critical **and** overdue); "Replacement Recommended"; "High Failure Frequency" (ticket_count ≥ 4); "Repeated Same Issue" (repeat_count ≥ 2); "Warranty Expired" (`warranty_expiry` < today).

**5. Prediction Method** — Threshold flags on existing per-asset values + warranty date. No probability model.

**6. Confidence** — Not applicable (deterministic flags on scored assets).

**7. Output** — Count in the heading + table of alerting assets.

**8. Charts** — None.

**9. Tables** — asset_code, asset_type, location, risk_level, alerts. Sources per point 4.

**10. Business Meaning** — Team: immediate escalations. Ops: SLA/warranty watch. Owner: warranty leakage.

**11. Limitations** — Warranty alerts depend on `warranty_expiry` being populated. Only scored assets appear.

---

## SECTION 11 — Apartment Health Score

**1. Business Purpose** — Which whole apartments are healthiest/worst — a building-level rollup above room level.

**2. Data Sources** — `business_views.apartment_health` over `build` assets + `mapped` tickets, grouped by apartment.

**3. Data Flow** — `assets + mapped → per-apartment aggregates → health score`.

**4. Logic Used** — Per apartment: total_tickets, repeat_failures, high_risk_assets, overdue_assets, avg_asset_age. **Health score** = `100 − penalty`, penalty = `min(tickets×1.5,30) + min(repeat×6,24) + min(high×8,24) + min(overdue×6,12) + min((avg_age/60)×10,10)`. Level via the reliability bands (High/Medium/Low/Poor).

**5. Prediction Method** — Aggregate of asset scores + ticket counts. Uses asset history, age, repeats, overdue. Not SLA/vendor.

**6. Confidence** — Not applicable (aggregate). In presentation, each apartment row is annotated with its top predicted asset + bed from the ticket-room view.

**7. Output** — Chart of the 15 worst apartments; detail table; each row also shows predicted_asset / predicted_location / failure_prob / reason (from the room view).

**8. Charts** — Apartment Health (lowest = worst, top 15): x = apartment_code, y = health_score.

**9. Tables** — apartment_code, total_tickets, repeat_failures, high_risk_assets, overdue_assets, avg_asset_age_months, health_score, health_level, plus predicted asset columns (presentation).

**10. Business Meaning** — Owner: worst buildings. Ops: building-level planning. Team: drill from here into rooms.

**11. Limitations** — Aggregates over scored assets; apartments with only unmapped tickets under-represent. Age fallback caveat applies.

---

## SECTION 12 — Failure Hotspots

**1. Business Purpose** — Where and when failures cluster — by apartment, by month, by floor, by asset type.

**2. Data Sources** — `business_views.hotspots` over `mapped` tickets + `apartment_master.floor_number` + asset_type_health.

**3. Data Flow** — `mapped tickets → group by apartment / month / floor`; `type failures from asset_type_health`.

**4. Logic Used** — apartment_density = ticket count per apartment (top 20). monthly_trend = tickets per month. floor_density = tickets per floor (via apartment→floor). type_failures = failure totals per type.

**5. Prediction Method** — Descriptive density (no prediction).

**6. Confidence** — Not applicable.

**7. Output** — Four charts + an enriched "Hotspot Apartments" table (with predicted asset + bed per apartment).

**8. Charts** — Apartment-wise Ticket Density (x apartment, y tickets); Monthly Ticket Trend (x month, y tickets); Asset-Type-wise Failures (x type, y total_failures); Floor-wise Ticket Density (x floor, y tickets).

**9. Tables** — Hotspot Apartments: apartment_code, tickets, + predicted asset/location/prob/reason (presentation).

**10. Business Meaning** — Owner: geographic/temporal concentration. Ops: crew routing, seasonal prep. Team: high-traffic areas.

**11. Limitations** — Floor density needs `floor_number`. Uses all mapped tickets (apartment-level), not bed-level.

---

## SECTION 13 — Preventive Maintenance Queue

**1. Business Purpose** — A prioritised, reasoned preventive worklist so failures are pre-empted.

**2. Data Sources** — `business_views.preventive_queue` over `build` assets.

**3. Data Flow** — `assets where risk ≠ Low → priority + why → sorted by health`.

**4. Logic Used** — Includes every asset with risk ≠ Low. Priority = its risk level. "Why" text assembled from: maintenance overdue, ≥2 repeat failures, recent failures, age ≥ 85% of life, failure_prob ≥ 60. Sorted by health ascending (worst first).

**5. Prediction Method** — Uses risk, overdue, repeats, recency, age ratio, probability. Not SLA/vendor.

**6. Confidence** — Not applicable (scored assets only).

**7. Output** — Cards 🔴 Critical / 🟠 High / 🟡 Medium counts; table with reason and location.

**8. Charts** — None.

**9. Tables** — asset_code, asset_type, location, priority, health_score, failure_prob_30d, maintenance_due_days, recommendation, why.

**10. Business Meaning** — Team: the daily/weekly preventive list. Ops: capacity planning. Owner: proactive vs reactive ratio.

**11. Limitations** — Scored assets only. "Why" thresholds are fixed.

---

## SECTION 14 — Business Recommendations

**1. Business Purpose** — Management-level recommendations with WHY and EXPECTED IMPACT, prioritised Critical→Low.

**2. Data Sources** — `business_recommendations()` derived from business_views, replacement planner, apartment report card, and ROI (all engine-computed from the same tickets/assets).

**3. Data Flow** — `built + business_views + planner + report_card + roi → recommendation list`.

**4. Logic Used** — Engine composes prioritised recommendations (priority, recommendation, why, expected_impact). The page sorts them Critical→High→Medium→Low for display.

**5. Prediction Method** — Aggregated from existing scores/aggregates. No new model.

**6. Confidence** — Not applicable.

**7. Output** — Bulleted recommendations, each with a priority badge, WHY and EXPECTED IMPACT; CSV export.

**8. Charts** — None.

**9. Tables** — Downloadable recommendations (priority, recommendation, why, expected_impact).

**10. Business Meaning** — Owner/ops: strategic actions. Team: context.

**11. Limitations** — Derived from partial cost/age data; "expected impact" is qualitative.

---

## SECTION 15 — Maintenance SLA Dashboard

**1. Business Purpose** — Are maintenance tickets resolved on time? Speed and SLA compliance of operations.

**2. Data Sources** — `sla_dashboard()` over `maintenance_tickets`.
- `created_at`, `resolved_at` — resolution time.
- `sla_deadline` — SLA compliance.
- `status` — open vs closed.
- `assigned_to` — technician grouping (UUID; no name table).
- `apartment_id`, `issue_type_id` (→ asset type) — group breakdowns.

**3. Data Flow** — `tickets → resolution hours & SLA met flags → overall + by apartment/issue/asset-type/technician`.

**4. Logic Used** — **resolution hours** = (resolved − created)/3600, negatives dropped. **SLA measured** only where both resolved and deadline exist; **met** if resolved ≤ deadline, **violated** if after. **open** = status not in {closed, resolved, completed, done}. Group SLA-met % = met ÷ (rows with both) per group.

**5. Prediction Method** — None. Pure operational measurement (auto-hidden if `created_at`/`resolved_at` absent).

**6. Confidence** — Not applicable.

**7. Output** — 🟢 SLA Met % and 🔴 SLA Violated % cards; Total/Open/Closed; resolution stats caption (avg/median/fastest/slowest, and how many tickets each was measured on).

**8. Charts** — None (cards + tables).

**9. Tables** — SLA by Apartment / by Issue Type / by Asset Type / Technician (assigned_to id shortened): tickets, avg_resolution_hours, sla_met_pct.

**10. Business Meaning** — Ops: technician & area performance. Owner: service quality. Team: own turnaround.

**11. Limitations** — SLA measurable only on tickets with both resolved + deadline (~250); resolution on ~412. Technician shown as a shortened UUID (no name table).

---

## SECTION 16 — Vendor / Brand Analysis

**1. Business Purpose** — Which asset brands are reliable vs failure-prone — for procurement decisions.

**2. Data Sources** — `brand_analysis()` over `build`/business_views assets: `brand`, `ticket_count`, `age_months`, `health_score`, `reliability_score`, `risk_level`, `replacement_recommended`, `asset_type`.

**3. Data Flow** — `assets with brand → group by brand → reliability metrics`; `group by (asset_type, brand) → comparative recommendations`.

**4. Logic Used** — Per brand: total_assets, total_failures (sum ticket_count), avg_age, avg_health, avg_reliability, high_risk count, replacement count; **failure_rate** = total_failures ÷ total_assets. Recommendation when, within an asset type, one brand's failure rate is ≥ 1.5× another (min 2 assets each).

**5. Prediction Method** — Descriptive reliability comparison. Uses asset ticket history, age, health/reliability. Not a forecast.

**6. Confidence** — Not applicable (auto-hidden if no brand data).

**7. Output** — 🏆 Best / ⚠️ Worst Performing Brand cards (brand + reliability + failure rate); brand recommendations list.

**8. Charts** — None.

**9. Tables** — Brand Reliability (all brands), Most Reliable Brands, Worst Performing Brands: brand, total_assets, failure_rate, avg_reliability.

**10. Business Meaning** — Owner/procurement: what to buy/avoid. Ops: reliability by brand. Team: context.

**11. Limitations** — Brand populated on only ~344/1,700 assets; brand metrics cover assets with a brand only. Failure counts are ticket counts, not warranty claims.

---

## SECTION 17 — Purchase Recommendation

**1. Business Purpose** — Per asset type, should we keep buying it? Procurement guidance.

**2. Data Sources** — `purchase_recommendation()` over assets: `asset_type`, `brand`, `reliability_score`, `ticket_count`, `expected_life_months`, `age_months`.

**3. Data Flow** — `assets → group by asset_type → best brand + reliability tier → Continue/Monitor/Reduce/Avoid`.

**4. Logic Used** — Per type: current/best brand (best = highest avg reliability when ≥2 brands), failure_rate (avg ticket_count), avg_reliability, maintenance frequency/yr, remaining useful life (expected_life − age). Tier by avg reliability: **≥75 Continue Purchasing** (Low priority), **≥55 Monitor** (Medium), **≥35 Reduce Purchase** (High), **else Avoid Purchase** (Critical).

**5. Prediction Method** — Aggregated reliability tiers. Uses reliability (age/tickets/repeats/recency/overdue), expected life, age. Not a demand forecast.

**6. Confidence** — Not applicable.

**7. Output** — Count cards 🟢 Continue / 🟡 Monitor / 🟠 Reduce / 🔴 Avoid; table with reason/benefit/priority.

**8. Charts** — None.

**9. Tables** — asset_type, current_brand (best), failure_rate, avg_reliability, avg_maintenance_frequency_per_year, avg_remaining_useful_life_months, recommendation, reason, expected_benefit, priority.

**10. Business Meaning** — Owner/procurement: buy/avoid by category. Ops: reliability expectations. Team: n/a.

**11. Limitations** — Reliability depends on age (allocation fallback) and ticket coverage. RUL blank where life/age missing.

---

## SECTION 18 — Asset Search & Profile

**1. Business Purpose** — Look up any specific asset and see its full profile and ticket timeline.

**2. Data Sources** — `build` assets (all per-asset fields) + `mapped` tickets (timeline).

**3. Data Flow** — `assets → filter by text/apartment/type/risk → profile`; `mapped filtered by asset_id → timeline`.

**4. Logic Used** — Free-text match on asset_code/brand/model + dropdown filters. Profile shows the asset's engine values. Timeline = every mapped ticket for that asset (date, issue, confidence). "Next Predicted" = today + maintenance_due_days.

**5. Prediction Method** — Displays the asset's own engine values (see 0.5). Uses full per-asset history + age + cycle.

**6. Confidence** — Timeline shows each ticket's mapping confidence (Verified/High/Medium/Low from 0.2).

**7. Output** — Result table (location in Apt/Bed) + profile metrics: Health Score, 30d Failure Prob, Failure Trend, Total Tickets, Age (months) + source, Expected Life, Maint. Cycle, Maintenance Due (days), Purchase Date, Recent (30d), Exp. Cost 1y; plus a 📍 location line.

**8. Charts** — None.

**9. Tables** — Search results (asset_code, asset_type, location, age_months, age_source, health_score, risk_level, failure_prob_30d, ticket_count, recommendation) and the per-asset ticket timeline (created_at, issue_type, confidence).

**10. Business Meaning** — Team: asset-level investigation. Ops: audit a specific complaint. Owner: spot-check.

**11. Limitations** — Only the 251 scored assets are searchable here. Exp. Cost uses the estimated avg ticket cost (closure_cost null).

---

## SECTION 19 — Export Reports

**1. Business Purpose** — Download the analytics as CSV for offline use, sharing, or archival.

**2. Data Sources** — `exports()` over `build` assets.

**3. Data Flow** — `assets → three report frames → CSV download buttons`.

**4. Logic Used** — **Full Asset Health Report** = all scored assets (minus internal id/date cols). **Maintenance Schedule** = assets with a maintenance_due_days, sorted soonest-first. **Replacement Plan** = assets where replacement_recommended is true.

**5. Prediction Method** — None (export of computed results).

**6. Confidence** — Not applicable.

**7. Output** — Three CSV buttons with row counts (e.g. Full Asset Health Report 251 rows; Maintenance Schedule 179; Replacement Plan 1).

**8. Charts** — None.

**9. Tables** — As per the three frames above.

**10. Business Meaning** — Owner: board packs. Ops: work orders. Team: printable lists.

**11. Limitations** — Exports cover scored assets only; costs are estimates; every table across the page also has its own inline CSV.

---

## APPENDIX — Cross-cutting limitations (apply everywhere)

1. **Purchase date sparse** — 308/1,700 assets have a purchase date; 232/251 scored assets are aged from allocation date, so age (and anything derived from it: health, age_ratio, replacement, RUL, lifecycle) can be understated.
2. **Asset-pinned coverage partial** — only 266 tickets carry a real `asset_id`; 213 more are High-confidence inferred (479 total → 251 assets). Medium (69) and Low/unmapped (957) tickets do not feed asset scoring.
3. **Cost is an estimate** — `closure_cost` is entirely null; all money figures use a fallback average (ticket_resolutions.total_cost, else ₹500).
4. **Room vs allocation** — ticket room is authoritative (`bed_id`), but the asset placed in that room comes from current allocation, which may have changed since the ticket.
5. **Failure probability is a capped Poisson** — floored exposure (30 days) and capped rate (λ≤0.2) deliberately damp re-ticket bursts; it ranks relative risk, it is not a calibrated absolute failure forecast.
6. **Not machine learning** — every score is a transparent rule/statistic; there is no trained model, no seasonality, no external signals.
7. **Wall-clock dependence** — recency windows, maintenance-due days, and estimated failure windows shift with today's date.
