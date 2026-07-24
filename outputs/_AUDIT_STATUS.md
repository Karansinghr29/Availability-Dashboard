# Audit artifact status

_Regenerated 2026-07-23 09:12:59Z by `src/_audit_datasources.py`._

Sources now in effect: bookings=Q35, tenants=Q42+Q32 (merge), beds_master=Q37(+Q33 backfill), current_occupancy=**Q50**, transfers=Q56.

## CURRENT (regenerated from live sources)
- `_dataloader_path_audit.json`
- `_dup_bed_audit_compact.json`
- `_recommendation_validation_audit.json`

## LEGACY / STALE (preserved for traceability — NOT regenerated)
Original generators are not in this repo, so these still describe the old Q31/Q33-based classification. Do not treat them as current:
- `_dataset_audit.json` — LEGACY
- `_dup_bed_audit.json` — LEGACY
- `_val_stderr.txt` — LEGACY
