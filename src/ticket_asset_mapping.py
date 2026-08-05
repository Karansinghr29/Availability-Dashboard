"""Phase 1 — Ticket → Asset mapping (maintenance fraud/anomaly pipeline).

Goal
----
Attach every maintenance ticket to the physical asset it most plausibly
concerns, with an explicit *confidence* tier, so later phases (asset-level
history, anomaly scoring) only trust reliable links and never silently invent
asset joins.

Why a tiered mapping
--------------------
Only ~18% of tickets carry a direct ``asset_id``. The rest must be inferred
from the room (``bed_id`` / ``apartment_id``) via ``asset_allocations`` and
disambiguated by the issue type using the ``maintenance_items`` bridge
(``issue_type_id`` -> ``asset_type_id``). That inference is sometimes unique
(one matching asset in the room) and sometimes ambiguous (several) — the
confidence tier records exactly how strong each link is.

Confidence tiers
----------------
Verified  : ticket.asset_id present (ground truth).
High      : no asset_id, but the room's allocation pool contains exactly ONE
            asset whose type matches the issue type (bed-level pool).
Medium    : unique match came only from the apartment-level pool, OR the room
            pool has 2-3 type-matching candidates (best guess recorded).
Low       : issue type has no asset-type bridge, or >3 candidates, or the pool
            has no type match — an asset link is not trustworthy.
Unmapped  : no allocation pool for the room at all (no bed/apartment asset) —
            cannot be mapped to any asset.

Only Verified + High are considered *reliable* for asset-level fraud scoring.
Medium is carried with a weight; Low/Unmapped stay at bed/tenant grain only.

This module is import-safe (returns dataframes) and also runnable as a script
to (re)generate the confidence report under ``outputs/``. It does NOT touch any
dashboard page.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

VERIFIED, HIGH, MEDIUM, LOW, UNMAPPED = "Verified", "High", "Medium", "Low", "Unmapped"
_TIERS = [VERIFIED, HIGH, MEDIUM, LOW, UNMAPPED]


# --------------------------------------------------------------------------- #
# small shared normalisers (kept local so the module is self-contained)
# --------------------------------------------------------------------------- #
def _s(v) -> str:
    """Normalise a scalar to a clean string; blanks/null-likes -> ''."""
    t = "" if v is None else str(v).strip()
    return "" if t.lower() in ("", "null", "nan", "none", "<na>", "nat") else t


def _dt(series) -> pd.Series:
    d = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return d.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return pd.to_datetime(series, errors="coerce")


# --------------------------------------------------------------------------- #
# reference lookups
# --------------------------------------------------------------------------- #
def _reference(loader) -> Dict:
    """Build id->name / issue->asset-type bridge / per-asset metadata."""
    atypes, assets = loader.asset_types(), loader.asset_master()
    issue, items = loader.issue_types(), loader.maintenance_items()
    aptm = loader.apartment_master()

    tname = (
        dict(zip(atypes["id"].map(_s), atypes["name"].map(_s)))
        if not atypes.empty else {}
    )
    iname = (
        dict(zip(issue["id"].map(_s), issue["name"].map(_s)))
        if not issue.empty else {}
    )
    apt_code = (
        dict(zip(aptm["id"].map(_s), aptm["apartment_code"].map(_s)))
        if not aptm.empty else {}
    )

    # issue_type_id -> {asset_type_name, ...}  (maintenance_items bridge)
    issue_to_types: Dict[str, set] = {}
    if not items.empty:
        for _, r in items.iterrows():
            it, at = _s(r.get("issue_type_id")), tname.get(_s(r.get("asset_type_id")), "")
            if it and at:
                issue_to_types.setdefault(it, set()).add(at)

    # per-asset metadata (type + purchase date), keyed by asset uuid
    meta: Dict[str, Dict] = {}
    if not assets.empty:
        pur = _dt(assets.get("purchase_date"))
        for i, aid in enumerate(assets["id"].map(_s)):
            if not aid:
                continue
            tid = _s(assets.iloc[i].get("asset_type_id"))
            meta[aid] = {
                "asset_code": _s(assets.iloc[i].get("asset_code")),
                "asset_type": tname.get(tid, ""),
                "purchase_date": pur.iloc[i],
            }
    return {
        "meta": meta,
        "issue_to_types": issue_to_types,
        "iname": iname,
        "apt_code": apt_code,
    }


def _allocations(loader) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Latest asset->room allocation, indexed as per-bed and per-apartment pools.

    Returns (per_bed, per_apt) where each maps a room id to the list of asset
    ids currently allocated there (deduplicated to each asset's latest row).
    """
    al = loader.asset_allocations().copy()
    if al.empty:
        return {}, {}
    al["_ad"] = _dt(al.get("allocated_date"))
    al["_apt"] = al["apartment_id"].map(_s)
    al["_bed"] = al["bed_id"].map(_s)
    al["_aid"] = al["asset_id"].map(_s)
    al = al[al["_aid"] != ""]
    # keep each asset's most recent allocation row
    latest = al.sort_values("_ad").drop_duplicates("_aid", keep="last")
    per_bed = (
        latest[latest["_bed"] != ""].groupby("_bed")["_aid"].apply(list).to_dict()
    )
    per_apt = (
        latest[latest["_apt"] != ""].groupby("_apt")["_aid"].apply(list).to_dict()
    )
    return per_bed, per_apt


# --------------------------------------------------------------------------- #
# core mapping
# --------------------------------------------------------------------------- #
def _classify(
    aid: str,
    apt: str,
    bed: str,
    issue_type_id: str,
    meta: Dict,
    i2t: Dict[str, set],
    per_bed: Dict[str, List[str]],
    per_apt: Dict[str, List[str]],
) -> Dict:
    """Resolve one ticket to (asset_id, confidence, method, candidate_count)."""
    if aid:
        return {
            "mapped_asset_id": aid,
            "mapping_confidence": VERIFIED,
            "mapping_method": "direct_asset_id",
            "candidate_count": 1,
            "candidate_asset_ids": aid,
        }

    types = i2t.get(issue_type_id, set())
    # prefer the tighter bed-level pool, fall back to apartment-level
    bed_pool = per_bed.get(bed) or []
    apt_pool = per_apt.get(apt) or []
    pool, grain = (bed_pool, "bed") if bed_pool else (apt_pool, "apartment") if apt_pool else ([], None)

    if grain is None:
        return {
            "mapped_asset_id": "",
            "mapping_confidence": UNMAPPED,
            "mapping_method": "no_allocation_pool",
            "candidate_count": 0,
            "candidate_asset_ids": "",
        }

    if not types:
        # issue type has no asset-type bridge -> cannot disambiguate the pool
        return {
            "mapped_asset_id": "",
            "mapping_confidence": LOW,
            "mapping_method": f"{grain}_issue_type_unmapped",
            "candidate_count": len(pool),
            "candidate_asset_ids": "",
        }

    cand = sorted(x for x in pool if meta.get(x, {}).get("asset_type", "") in types)
    n = len(cand)
    if n == 1:
        conf = HIGH if grain == "bed" else MEDIUM
        method = f"{grain}_type_unique"
        chosen = cand[0]
    elif 2 <= n <= 3:
        conf, method, chosen = MEDIUM, f"{grain}_type_multi", cand[0]
    elif n > 3:
        conf, method, chosen = LOW, f"{grain}_type_many", ""
    else:  # n == 0, pool exists but nothing of the matching type
        conf, method, chosen = LOW, f"{grain}_no_type_match", ""

    return {
        "mapped_asset_id": chosen,
        "mapping_confidence": conf,
        "mapping_method": method,
        "candidate_count": n,
        "candidate_asset_ids": ",".join(cand),
    }


def build_ticket_asset_mapping(loader=None) -> pd.DataFrame:
    """Return a per-ticket mapping dataframe with confidence + provenance.

    One row per maintenance ticket. Import this from later phases; it never
    mutates loader state or any dashboard artefact.
    """
    if loader is None:
        from data_loader import DataLoader  # local import: avoids hard dependency
        loader = DataLoader()

    mt = loader.maintenance_tickets().copy()
    if mt.empty:
        return pd.DataFrame()

    ref = _reference(loader)
    meta, i2t, iname, apt_code = (
        ref["meta"], ref["issue_to_types"], ref["iname"], ref["apt_code"]
    )
    per_bed, per_apt = _allocations(loader)

    created = _dt(mt.get("created_at"))
    rows = []
    for pos, (_, t) in enumerate(mt.iterrows()):
        aid = _s(t.get("asset_id"))
        apt = _s(t.get("apartment_id"))
        bed = _s(t.get("bed_id"))
        it = _s(t.get("issue_type_id"))
        res = _classify(aid, apt, bed, it, meta, i2t, per_bed, per_apt)
        chosen = res["mapped_asset_id"]
        rows.append({
            "ticket_id": _s(t.get("id")),
            "ticket_number": _s(t.get("ticket_number")),
            "tenant_id": _s(t.get("tenant_id")),
            "apartment_id": apt,
            "apartment_code": apt_code.get(apt, apt),
            "bed_id": bed,
            "issue_type_id": it,
            "issue_type": iname.get(it, ""),
            "created_at": created.iloc[pos],
            **res,
            "mapped_asset_type": meta.get(chosen, {}).get("asset_type", ""),
            "mapped_asset_purchase_date": meta.get(chosen, {}).get("purchase_date", pd.NaT),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# confidence report
# --------------------------------------------------------------------------- #
def mapping_report(mapping: pd.DataFrame) -> Dict:
    """Summarise coverage / confidence for the owner-facing report."""
    total = len(mapping)
    conf = mapping["mapping_confidence"].value_counts().to_dict()
    counts = {k: int(conf.get(k, 0)) for k in _TIERS}
    reliable = counts[VERIFIED] + counts[HIGH]
    incl_medium = reliable + counts[MEDIUM]

    def pct(x: int) -> float:
        return round(100 * x / total, 1) if total else 0.0

    linked = mapping["mapped_asset_id"].map(_s) != ""
    return {
        "total_tickets": total,
        "by_confidence": counts,
        "by_confidence_pct": {k: pct(v) for k, v in counts.items()},
        "reliable_verified_high": reliable,
        "reliable_pct": pct(reliable),
        "incl_medium": incl_medium,
        "incl_medium_pct": pct(incl_medium),
        "tickets_with_asset_link": int(linked.sum()),
        "distinct_assets_linked": int(mapping.loc[linked, "mapped_asset_id"].nunique()),
        "by_method": mapping["mapping_method"].value_counts().to_dict(),
    }


def _format_report(rep: Dict) -> str:
    lines = ["=" * 60, "TICKET -> ASSET MAPPING — CONFIDENCE REPORT", "=" * 60]
    t = rep["total_tickets"]
    lines.append(f"Total tickets: {t}")
    lines.append("")
    lines.append("By confidence tier:")
    for k in _TIERS:
        c = rep["by_confidence"][k]
        lines.append(f"  {k:<9} {c:>5}  ({rep['by_confidence_pct'][k]:>5.1f}%)")
    lines.append("")
    lines.append(f"Reliable (Verified+High) : {rep['reliable_verified_high']:>5}  "
                 f"({rep['reliable_pct']}%)  <- used for asset-level scoring")
    lines.append(f"Incl. Medium             : {rep['incl_medium']:>5}  "
                 f"({rep['incl_medium_pct']}%)")
    lines.append(f"Any asset link           : {rep['tickets_with_asset_link']:>5}")
    lines.append(f"Distinct assets linked   : {rep['distinct_assets_linked']:>5}")
    lines.append("")
    lines.append("By mapping method:")
    for m, c in sorted(rep["by_method"].items(), key=lambda kv: -kv[1]):
        lines.append(f"  {m:<28} {c:>5}")
    lines.append("=" * 60)
    return "\n".join(lines)


def _report_frame(rep: Dict) -> pd.DataFrame:
    """Flat one-metric-per-row frame for CSV export."""
    recs = [{"metric": "total_tickets", "value": rep["total_tickets"]}]
    for k in _TIERS:
        recs.append({"metric": f"confidence.{k}", "value": rep["by_confidence"][k]})
        recs.append({"metric": f"confidence_pct.{k}", "value": rep["by_confidence_pct"][k]})
    for k in ("reliable_verified_high", "reliable_pct", "incl_medium",
              "incl_medium_pct", "tickets_with_asset_link", "distinct_assets_linked"):
        recs.append({"metric": k, "value": rep[k]})
    for m, c in rep["by_method"].items():
        recs.append({"metric": f"method.{m}", "value": c})
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# script entry point
# --------------------------------------------------------------------------- #
def main(out_dir: Optional[str] = None) -> Dict:
    from data_loader import DataLoader

    loader = DataLoader()
    mapping = build_ticket_asset_mapping(loader)
    rep = mapping_report(mapping)

    base = Path(out_dir) if out_dir else Path(__file__).resolve().parents[1] / "outputs"
    base.mkdir(parents=True, exist_ok=True)
    map_path = base / "ticket_asset_mapping.csv"
    rep_path = base / "ticket_asset_mapping_report.csv"
    mapping.to_csv(map_path, index=False)
    _report_frame(rep).to_csv(rep_path, index=False)

    print(_format_report(rep))
    print(f"\nWrote per-ticket mapping : {map_path}")
    print(f"Wrote confidence report  : {rep_path}")
    return rep


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
