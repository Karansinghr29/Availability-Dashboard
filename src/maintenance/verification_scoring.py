"""Deterministic asset-history Verification decision (NOT a fraud/ML model).

Owner's rule — driven ONLY by the comparison between the current repair time and
the asset's own historical repair duration:

  For every completed ticket:
    1. Historical baseline = median resolution time of that asset's PREVIOUS
       completed repairs (the current ticket is excluded).
    2. If the asset lacks enough history, fall back to the asset-type baseline,
       then the issue-type baseline.
    3. If the current repair is significantly faster than that baseline
       (current < fast_ratio * baseline, default 25%) -> "Verification Required".
    4. Otherwise -> "Normal".

The decision is NOT a fraud claim and does NOT use repeat counts, tenant, or
technician behaviour. Repeated repairs only supply the history used to establish
the asset's normal duration; the decision is purely the speed comparison.

The SAME function scores the 1505 historical tickets (investigation queue) and
any future completed ticket — ``evaluate_ticket`` is a pure, deterministic
function of its inputs.

Baseline history (verified-first, provisional fallback to all completed repairs)
and the two-history model are reused from ``verification_queue`` unchanged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from verification_queue import (
    VerifyConfig, _asset_prior_baseline, _baseline_pool, _export_histories,
    _fallback_lookups, _hours_str, _recency, _spine, _verified_ticket_ids,
    _CONFIDENCE, build_asset_baselines,
)

_ROOT = Path(__file__).resolve().parents[2]
_OUT = _ROOT / "outputs"


def _pick_baseline(l2, l3m, l3n, gasset, r, cfg: VerifyConfig):
    """Asset median (PRIOR verified repairs only) -> asset_type -> issue_type."""
    b = _asset_prior_baseline(gasset, r, cfg.min_asset_history, cfg.baseline_method)
    if b:
        return b[0], "asset", b[1]
    hit = l2.get((r["asset_type"], r["issue_type"]))
    if hit:
        return float(hit[0]), "asset_type", int(hit[1])
    gm = l3m.get(r["issue_type"])
    if gm is not None and gm > 0:
        return float(gm), "issue_type", int(l3n.get(r["issue_type"], 0))
    return None


def _explain_required(source: str, baseline_h: float, current_h: float,
                      faster_pct: float, asset_type: str, issue_type: str,
                      provisional: bool) -> str:
    if source == "asset":
        basis = "based on its historical repair records"
    elif source == "asset_type":
        basis = (f"based on similar {asset_type} repairs "
                 f"(this asset lacks enough of its own history)")
    else:
        basis = (f"based on {issue_type} repairs "
                 f"(this asset lacks enough of its own history)")
    note = ("" if not provisional else
            " These records are not yet manager-verified, so the baseline is provisional.")
    return (f"This asset normally takes about {_hours_str(baseline_h)} to repair "
            f"{basis}. This ticket was completed in only {_hours_str(current_h)} "
            f"({faster_pct:.0f}% faster than normal). Verification is recommended."
            f"{note}")


def evaluate_ticket(r: pd.Series, baselines: Dict, cfg: VerifyConfig) -> Dict:
    """Deterministic verification decision for ONE completed ticket.

    Pure function — identical result for a historical or a future ticket given
    the same asset repair history. Decision depends ONLY on the speed comparison.
    """
    rh = r["resolution_hours"]
    graw = baselines["raw_by"].get(r["asset_id"])
    gver = baselines["ver_by"].get(r["asset_id"])
    # recent repeat count is CONTEXT only (never part of the decision)
    _, days_since, recent_ct = _recency(graw, r, cfg)

    baseline, source, sample_n, hist_type = np.nan, "", 0, ""
    picked = _pick_baseline(baselines["l2_v"], baselines["l3m_v"], baselines["l3n_v"], gver, r, cfg)
    if picked:
        baseline, source, sample_n, hist_type = *picked, "Verified"
    elif cfg.allow_provisional_raw:
        picked = _pick_baseline(baselines["l2_r"], baselines["l3m_r"], baselines["l3n_r"], graw, r, cfg)
        if picked:
            baseline, source, sample_n, hist_type = *picked, "Provisional"

    decision = "Normal"
    faster_pct = np.nan
    explanation = ""
    if pd.isna(rh):
        explanation = "No recorded resolution time; ticket not evaluated."
    elif source == "":
        explanation = ("No manager-verified repair history for this asset, asset "
                       "type, or issue type yet; no verified baseline available, "
                       "so this ticket is not evaluated.")
    else:
        pct = rh / baseline
        faster_pct = round((1.0 - pct) * 100.0, 1)
        provisional = hist_type != "Verified"
        if pct < cfg.fast_ratio:
            decision = "Verification Required"
            explanation = _explain_required(source, baseline, rh, faster_pct,
                                            r["asset_type"], r["issue_type"], provisional)
        else:
            explanation = (f"Completed in {_hours_str(rh)}, in line with this asset's "
                           f"normal repair time (~{_hours_str(baseline)}). No verification needed.")

    return {
        "verification_decision": decision,
        "asset_historical_baseline_hours": round(baseline, 2) if pd.notna(baseline) else np.nan,
        "baseline_source": source,
        "baseline_confidence": _CONFIDENCE.get(source, ""),
        "baseline_history_type": hist_type,
        "baseline_sample_size": sample_n,
        "faster_percentage": faster_pct,
        "recent_repeat_count_90d": recent_ct,   # context only
        "days_since_previous_repair": round(float(days_since), 1) if pd.notna(days_since) else np.nan,
        "explanation": explanation,
    }


def _baselines(df: pd.DataFrame, base: Path, cfg: VerifyConfig) -> Dict:
    raw_hist = df[(df["asset_id"] != "") & df["resolution_hours"].notna()].copy()
    verified_ids = _verified_ticket_ids(base)
    raw_hist["is_verified"] = raw_hist["ticket_id"].astype(str).isin(verified_ids)
    ver_hist = raw_hist[raw_hist["is_verified"]].copy()
    # maintain the two histories (Raw = audit, Verified = baseline source)
    _export_histories(raw_hist, ver_hist, base)
    # baseline pool: exclude System Closures (<15min) + archived Asset-Replaced
    ver_hist = _baseline_pool(ver_hist, df, base, cfg)
    l2_v, (l3m_v, l3n_v) = _fallback_lookups(ver_hist, cfg)
    l2_r, (l3m_r, l3n_r) = _fallback_lookups(raw_hist, cfg)
    return {
        "raw_by": {a: g.sort_values("resolved_at") for a, g in raw_hist.groupby("asset_id")},
        "ver_by": {a: g.sort_values("resolved_at") for a, g in ver_hist.groupby("asset_id")},
        "l2_v": l2_v, "l3m_v": l3m_v, "l3n_v": l3n_v,
        "l2_r": l2_r, "l3m_r": l3m_r, "l3n_r": l3n_r,
    }


def build_verification_scores(loader=None, cfg: Optional[VerifyConfig] = None,
                              out_dir: Optional[str] = None,
                              force_rebuild: bool = False
                              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run the decision on all completed tickets; write full table + queue.

    FROZEN LEDGER: a ticket's decision is computed ONCE, with the baseline that
    existed at that time, and then frozen — later manager verifications never
    change an already-decided ticket. Only tickets not yet in the ledger are
    evaluated (with the current/latest baseline). ``force_rebuild=True`` re-stamps
    every ticket (use only for a deliberate full recompute).
    """
    cfg = cfg or VerifyConfig()
    base = Path(out_dir) if out_dir else _OUT
    if loader is None:
        from data_loader import DataLoader
        loader = DataLoader()

    df = _spine(loader, base)
    baselines = _baselines(df, base, cfg)
    build_asset_baselines(df=df, base=base, cfg=cfg)   # maintain per-asset store
    from verification_queue import _load_store
    store = _load_store(base)
    ver_of = dict(zip(store.get("asset_id", pd.Series(dtype=str)).astype(str),
                      store.get("baseline_version", pd.Series(dtype=str)))) if not store.empty else {}

    # load the frozen ledger (only if it already carries the freeze schema)
    frozen: Dict[str, dict] = {}
    ledger = base / "maintenance_ticket_verification.csv"
    if ledger.exists() and not force_rebuild:
        old = pd.read_csv(ledger, dtype={"ticket_id": str})
        if "evaluated_at" in old.columns:
            frozen = {str(r["ticket_id"]): r.to_dict() for _, r in old.iterrows()}

    now = pd.Timestamp.now().isoformat(timespec="seconds")
    rows = []
    for _, r in df.iterrows():
        tid = str(r["ticket_id"])
        if tid in frozen:                 # decision frozen at its original baseline
            rows.append(frozen[tid])
            continue
        res = evaluate_ticket(r, baselines, cfg)
        rows.append({
            "ticket_id": r["ticket_id"],
            "ticket_number": r["ticket_number"],
            "asset_id": r["asset_id"],
            "asset_id_source": r["asset_id_source"],
            "asset_code": r["asset_code"],
            "issue_type": r["issue_type"],
            "current_repair_date": r["resolved_at"],
            "current_resolution_hours": round(float(r["resolution_hours"]), 2)
            if pd.notna(r["resolution_hours"]) else np.nan,
            "mapping_confidence": r["mapping_confidence"],
            **res,
            # ---- audit snapshot: the exact baseline that produced this decision
            "baseline_method": cfg.baseline_method,
            "baseline_version": (ver_of.get(str(r["asset_id"]), "")
                                 if res["baseline_source"] == "asset" else ""),
            "evaluated_at": now,
        })
    scores = pd.DataFrame(rows)
    # order: Required first, then by how much faster than baseline
    scores["_req"] = (scores["verification_decision"] == "Verification Required").astype(int)
    scores = scores.sort_values(["_req", "faster_percentage"],
                                ascending=[False, False]).drop(columns="_req").reset_index(drop=True)
    queue = scores[scores["verification_decision"] == "Verification Required"].copy()

    report = _validate(scores, cfg)
    base.mkdir(parents=True, exist_ok=True)
    scores.to_csv(base / "maintenance_ticket_verification.csv", index=False)
    queue.to_csv(base / "maintenance_verification_queue.csv", index=False)
    report.to_csv(base / "maintenance_verification_score_validation.csv", index=False)
    return scores, report


def _validate(scores: pd.DataFrame, cfg: VerifyConfig) -> pd.DataFrame:
    req = scores["verification_decision"] == "Verification Required"
    evaluated = scores["faster_percentage"].notna()
    recs = [
        {"metric": "tickets_scored", "value": len(scores)},
        {"metric": "tickets_evaluated (had baseline + resolution time)", "value": int(evaluated.sum())},
        {"metric": "verification_required", "value": int(req.sum())},
        {"metric": "normal", "value": int((~req).sum())},
        {"metric": "fast_ratio_threshold", "value": cfg.fast_ratio},
    ]
    if req.any():
        for src, c in scores.loc[req, "baseline_source"].value_counts().items():
            recs.append({"metric": f"required.baseline_source.{src}", "value": int(c)})
        for ht, c in scores.loc[req, "baseline_history_type"].value_counts().items():
            recs.append({"metric": f"required.baseline_history.{ht}", "value": int(c)})
        fp = scores.loc[req, "faster_percentage"]
        recs += [{"metric": "required.faster_pct.min", "value": round(float(fp.min()), 1)},
                 {"metric": "required.faster_pct.median", "value": round(float(fp.median()), 1)},
                 {"metric": "required.faster_pct.max", "value": round(float(fp.max()), 1)}]
    return pd.DataFrame(recs)


def _print(scores: pd.DataFrame, report: pd.DataFrame) -> None:
    print("=" * 68)
    print("ASSET-HISTORY VERIFICATION DECISION (all completed tickets)")
    print("=" * 68)
    for _, r in report.iterrows():
        print(f"  {r['metric']:<48} {r['value']}")
    req = scores[scores["verification_decision"] == "Verification Required"]
    print("-" * 68)
    for _, r in req.head(4).iterrows():
        print(f"  {r['asset_code']:<16} {str(r['explanation'])[:80]}")
    print("=" * 68)


def main() -> None:
    scores, report = build_verification_scores()
    _print(scores, report)
    print(f"\nWrote maintenance_ticket_verification.csv + queue + validation -> {_OUT}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
