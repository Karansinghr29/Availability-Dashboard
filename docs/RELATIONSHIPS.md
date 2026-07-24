# Dataset Relationships

The data lives in **two key universes** that must be bridged:

* **Text-key universe** — joined on `property_name` + `apartment_code` + `bed_code`.
  Tables: `tenants` (current allocation cols), `bookings`, `maintenance`,
  `electricity_bills`, `electricity_readings`.
* **UUID universe** — joined on `tenant_id` / `allotment_id`.
  Tables: `invoices`, `payments` (and `tenants.id`).

`tenants` is the **bridge**: it has both `id` (UUID, used by invoices/payments)
and the text keys `property_name` / `apartment_code` / `bed_code`.

## Entity relationship overview

```
                         ┌────────────────────────────┐
                         │          tenants            │
                         │  PK: id (uuid)              │
                         │  home: city, state          │
                         │  current: property_name,    │
                         │           apartment_code,   │
                         │           bed_code          │
                         └───────┬─────────────┬───────┘
             text keys           │             │  id (uuid) = tenant_id
   (property_name+apartment+bed) │             │
                                 │             ▼
        full_name+phone          │      ┌──────────────┐   allotment_id
                ▼                │      │   invoices   │◄──────────────┐
        ┌───────────────┐        │      │ PK id        │               │
        │   bookings    │        │      │ tenant_id FK │               │
        │ apartment_code│        │      │ apartment_id │ (uuid, no map)│
        │ bed_code      │        │      └──────┬───────┘               │
        │ bed_type      │        │             │ allotment_id          │
        │ monthly_rental│        │             ▼                       │
        │ onboarding/   │        │      ┌──────────────┐               │
        │ exit/notice   │        │      │   payments   │───────────────┘
        └───────────────┘        │      │ tenant_id FK │
                                 │      │ tenant_allotment_id
        ┌───────────────┐        │      └──────────────┘
        │ maintenance   │◄───────┤
        │ apt+bed codes │        │
        └───────────────┘        │
        ┌───────────────┐        │
        │ electricity_* │◄───────┘   (property_name + apartment_code)
        └───────────────┘
```

## Join keys, table by table

| From | To | Join | Notes |
|---|---|---|---|
| `bookings` | `tenants` | `full_name` + `phone` | Enriches a booking with tenant home `state`. |
| `bookings` | `tenants` (current) | `apartment_code` + `bed_code` | Who currently holds the bed. |
| `tenants` | `invoices` | `tenants.id = invoices.tenant_id` | Bridge UUID ↔ text world. |
| `tenants` | `payments` | `tenants.id = payments.tenant_id` | Collections per tenant. |
| `invoices` | `payments` | `invoices.allotment_id = payments.tenant_allotment_id` | Same allocation. |
| `maintenance` | `bookings`/`tenants` | `apartment_code` (+`bed_code`) | Tickets per unit. |
| `electricity_bills` | `bookings`/`tenants` | `property_name` + `apartment_code` | Utility cost per apartment. |
| `electricity_readings` | `electricity_bills` | `property_name` + `apartment_code` | Consumption vs billed. |

## Location hierarchy (verified against the data)

```
Property (property_name, e.g. Vista Heights)
  └─ Apartment (apartment_code, e.g. C11)        # INTERNAL ranking only
       └─ Room  (apartment_code + bed letter, e.g. C11-A)   # customer-facing unit
            └─ Bed (apartment_code + bed_code, e.g. C11-A1)  # the specific bed
```

* Room identity is produced in exactly ONE place —
  `preprocessing.build_room_inventory()` — and every other module treats
  `room_code` as **opaque**. No other module derives a room from `bed_code`.
* **Today's CSV fallback convention** (inside `build_room_inventory` only):
  a room = `apartment_code` + the first (letter) char of `bed_code`.
* **Future DB:** if the source provides `room_id` / `room_code` / `capacity`,
  those are authoritative and the fallback is skipped — no downstream change.
* Room **capacity**: today derived from the data (`bed_type` Single=1 / Double=2
  / Triple=3, cross-checked with distinct beds seen per room); tomorrow taken
  from the source capacity column when present. Never hardcoded as the sole truth.
* Evidence for the current convention: apartment `C11` has room `A` (Single, bed
  `A1`), room `B` (Double, beds `B1,B2`), room `C` (Double, beds `C1,C2`); Triple
  rooms encode three beds `X1,X2,X3` (verified in `A24-B`, `B24-B`, `C24-B`,
  `A41-B`). The same letter can be a different type in a different apartment
  (e.g. `B` is Double in `C11` but Triple in `B24`), so a room is always keyed by
  `apartment_code` + letter, never by letter alone.

## How this powers the business modules

* **Room occupancy** (customer-facing) — for each room, occupied beds =
  `bookings` rows with `actual_exit_date IS NULL` in that room; available beds =
  capacity − occupied. Recommendations are always a room + a specific available
  `bed_code`.
* **Apartment occupancy / revenue** (internal ranking only) — aggregated to
  `apartment_code` to decide which apartments to fill first; never shown to the
  customer.
* **Same-state matching** — occupied beds joined to `tenants.state`; the
  customer's `city` is resolved to a state via a map learned from
  `tenants(city, state)` (+ static fallback).
* **Revenue per apartment** — primary metric is sum of `monthly_rental` of
  occupied beds (`bookings`). Actual collected revenue can be derived from
  `invoices`/`payments` via the `tenant_id` bridge.
* **Historical fill time** — per bed, gap between one tenant's `actual_exit_date`
  and the next tenant's `onboarding_date` in `bookings`.
* **Revenue loss** — `monthly_rental` (bed rent) × current vacant days.

## Important structural caveats (see gap report in README)

1. **No explicit bed/room master.** Bed inventory and room capacity are *derived*
   from `bookings`. If a bed has never been booked, it is invisible.
2. **No `apartment_id` ↔ `apartment_code` map.** `invoices`/`payments` use UUID
   `apartment_id`/`bed_id`, while everything else uses text codes. Apartment-level
   revenue from invoices must go through the `tenant_id` bridge (reflects a
   tenant's *current* apartment, so historical attribution can drift).
3. **No property location.** Properties have no `city`/`state`, so the customer
   `city` is used only to derive the customer's state for roommate matching, not
   to geo-filter properties.
