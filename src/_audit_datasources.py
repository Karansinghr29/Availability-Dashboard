"""_audit_datasources.py — DIAGNOSTIC (no business logic)
=========================================================

Repeatable regenerator for the loader-derived audit artifacts in ``outputs/``.
It reflects the CURRENT data sources selected by ``DataLoader`` (Q35 bookings,
Q42+Q32 tenants, Q37 beds, **Q50 current occupancy**, Q56 transfers, ...), so the
audits no longer describe the old Q31/Q33-based classification.

It regenerates ONLY the artifacts that are objectively derivable from the live
loader, and writes a status index (``_AUDIT_STATUS.md``) that marks the remaining
legacy audits (whose original generators are not in this repo) as stale without
deleting them.

Regenerated (current):
  * outputs/_dataloader_path_audit.json        (discovery + primary source per table)
  * outputs/_dup_bed_audit_compact.json        (beds_master duplicate resolution)
  * outputs/_recommendation_validation_audit.json (scenario smoke check)

Marked legacy (preserved, not deleted) via _AUDIT_STATUS.md:
  * outputs/_dataset_audit.json  * outputs/_dup_bed_audit.json  * outputs/_val_stderr.txt

Run:  python src/_audit_datasources.py
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from data_loader import DataLoader, TABLE_REGISTRY
from preprocessing import normalize_beds_master, resolve_beds_master

logging.getLogger().setLevel(logging.ERROR)

_OUT = Path(__file__).resolve().parents[1] / "outputs"
_STAMP = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")

# Legacy audits whose original generators are not in the repo (can't be
# faithfully regenerated here). Preserved for traceability; flagged stale.
_LEGACY_FILES = (
    "_dataset_audit.json",
    "_dup_bed_audit.json",
    "_val_stderr.txt",
)


def _provenance() -> dict:
    return {
        "_regenerated_utc": _STAMP,
        "_generator": "src/_audit_datasources.py",
        "_note": "Reflects current DataLoader sources (Q35/Q42+Q32/Q37/Q50/Q56).",
    }


def _first_codes(df: pd.DataFrame, n: int = 3) -> list:
    if df is None or df.empty or "apartment_code" not in df.columns:
        return []
    return [str(x) for x in df["apartment_code"].head(n).tolist()]


def regen_path_audit(loader: DataLoader) -> dict:
    mapping = loader._discover()
    tables: dict = {}
    for name in TABLE_REGISTRY:
        files = mapping.get(name, [])
        discovered = [
            {
                "filename": fp.name,
                "absolute_path": str(fp),
                "last_modified": datetime.fromtimestamp(
                    fp.stat().st_mtime
                ).strftime("%Y-%m-%d %H:%M:%S"),
                "size_bytes": fp.stat().st_size,
            }
            for fp in files
        ]
        primary, secondaries, rows, first_codes, err = None, [], 0, [], None
        if files:
            selected = loader._select_primary_csv_files(name, files)
            primary = selected[0].name if selected else None
            secondaries = [p.name for p in selected[1:]]
        try:
            df = loader.load(name)
            rows = int(len(df))
            first_codes = _first_codes(df)
        except FileNotFoundError:
            err = "no source file"
        except Exception as exc:  # noqa: BLE001
            err = str(exc)
        tables[name] = {
            "discovered_files": discovered,
            "primary_file": primary,
            "secondary_files_backfill": secondaries,
            "load_error": err,
            "rows": rows,
            "first_apartment_codes": first_codes,
        }
    return {
        **_provenance(),
        "data_dir": str(loader.data_dir),
        "primary_source_by_table": {
            n: t["primary_file"] for n, t in tables.items() if t["primary_file"]
        },
        "tables": tables,
    }


def regen_dup_bed_compact(loader: DataLoader) -> list:
    try:
        raw = loader.beds_master()
    except Exception:  # noqa: BLE001
        return []
    master = normalize_beds_master(raw)
    if master.empty:
        return []
    resolved, _report = resolve_beds_master(raw)
    resolved_by_key = {
        (str(r["apartment_code"]).strip().upper(), str(r["bed_code"]).strip().upper()): r
        for _, r in resolved.iterrows()
    }
    work = master.copy()
    work["_apt"] = work["apartment_code"].astype(str).str.strip().str.upper()
    work["_bed"] = work["bed_code"].astype(str).str.strip().str.upper()
    rows = []
    for (apt, bed), g in work.groupby(["_apt", "_bed"], sort=True):
        if len(g) < 2:
            continue  # only report duplicate keys
        variants = "; ".join(
            f"{str(t).strip() or '?'}@{('' if pd.isna(rt) else rt)}"
            for t, rt in zip(g["toilet_type"], g["current_rate"])
        )
        pick = resolved_by_key.get((apt, bed))
        sel = (
            f"{str(pick['toilet_type']).strip()} @ {pick['current_rate']}"
            if pick is not None
            else "?"
        )
        rows.append(
            {
                "apartment_code": apt,
                "bed_code": bed,
                "variants": variants,
                "selected": sel,
                "reason": "Prefer Common; latest valid in class (see resolve_beds_master).",
                "ok": pick is not None,
            }
        )
    return rows


def regen_reco_validation(loader: DataLoader) -> dict:
    from preprocessing import build_room_inventory
    from recommendation_engine import recommend_rooms

    tenants = None
    try:
        tenants = loader.tenants()
    except Exception:  # noqa: BLE001
        tenants = None
    beds_master = None
    try:
        beds_master = loader.beds_master()
    except Exception:  # noqa: BLE001
        beds_master = None
    inv = build_room_inventory(loader.bookings(), tenants=tenants, beds_master=beds_master)

    scenarios = [
        ("Chennai", "Male", None), ("Chennai", "Female", None),
        ("Kerala", "Male", "Double"), ("Tamil Nadu", "Male", None),
        ("West Bengal", "Male", None), ("Mumbai", "Female", None),
        ("Bangalore", "Male", "Single"), ("Delhi", None, None),
    ]
    from inventory_views import active_vacant_rooms
    from roommate_matching import same_state_rooms

    results, total_cards = [], 0
    for city, gender, bed in scenarios:
        rec = recommend_rooms(
            loader, city=city, bed_type=bed, gender=gender,
            room_inventory=inv, top_n=5,
        )
        state = rec.attrs.get("detected_state")
        pool_same = (
            bool(state) and not same_state_rooms(active_vacant_rooms(inv), state).empty
        )
        total_cards += len(rec)
        results.append(
            {
                "name": f"city={city}|gender={gender}|bed={bed}",
                "cards": int(len(rec)),
                "detected_state": state,
                "pool_has_same_state": pool_same,
            }
        )
    return {
        **_provenance(),
        "total_recommendation_cards_checked": total_cards,
        "scenarios": results,
        "inventory_rooms": int(len(inv)),
        "verdict": "Regenerated smoke check (observational; no failures asserted).",
    }


def write_status_index() -> None:
    lines = [
        "# Audit artifact status",
        "",
        f"_Regenerated {_STAMP} by `src/_audit_datasources.py`._",
        "",
        "Sources now in effect: bookings=Q35, tenants=Q42+Q32 (merge), "
        "beds_master=Q37(+Q33 backfill), current_occupancy=**Q50**, transfers=Q56.",
        "",
        "## CURRENT (regenerated from live sources)",
        "- `_dataloader_path_audit.json`",
        "- `_dup_bed_audit_compact.json`",
        "- `_recommendation_validation_audit.json`",
        "",
        "## LEGACY / STALE (preserved for traceability — NOT regenerated)",
        "Original generators are not in this repo, so these still describe the old "
        "Q31/Q33-based classification. Do not treat them as current:",
    ]
    for f in _LEGACY_FILES:
        exists = (_OUT / f).exists()
        lines.append(f"- `{f}`{'' if exists else ' (missing)'} — LEGACY")
    (_OUT / "_AUDIT_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    _OUT.mkdir(parents=True, exist_ok=True)
    loader = DataLoader()

    (_OUT / "_dataloader_path_audit.json").write_text(
        json.dumps(regen_path_audit(loader), indent=2, default=str), encoding="utf-8"
    )
    (_OUT / "_dup_bed_audit_compact.json").write_text(
        json.dumps(regen_dup_bed_compact(loader), indent=2, default=str), encoding="utf-8"
    )
    (_OUT / "_recommendation_validation_audit.json").write_text(
        json.dumps(regen_reco_validation(loader), indent=2, default=str), encoding="utf-8"
    )
    write_status_index()
    print("Regenerated: _dataloader_path_audit.json, _dup_bed_audit_compact.json, "
          "_recommendation_validation_audit.json")
    print(f"Legacy marked in _AUDIT_STATUS.md: {', '.join(_LEGACY_FILES)}")


if __name__ == "__main__":
    main()
