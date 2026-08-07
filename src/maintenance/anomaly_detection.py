"""Phase 3 — Maintenance investigation risk scoring (NOT a fraud classifier).

No confirmed fraud labels exist, so this produces an *investigation risk score*
(0-100) that ranks tickets / assets / tenants / technicians for human review.
Two independent signals are combined:

  1. Rule-based HARD FLAGS   — deterministic, explainable, domain-driven.
  2. IsolationForest anomaly — unsupervised, catches multivariate oddness.

risk_score = 100 * ( w_model * confidence * anomaly_norm + w_rule * rule_norm )
with a floor applied when a *critical* hard flag fires, so a decisive rule
(e.g. cost far above expected) can never be buried by a calm anomaly score.

Every flagged record carries human-readable reasons ("Same issue repeated 14
times", "Ticket count 5x the Fan peer median", ...).

Consumes ONLY the Phase-2 feature CSVs (plus a light optional join to raw
tickets for the ticket-level rejection / process-edit flags, which are not in
the feature CSV). Writes risk CSVs + a validation report. No dashboard changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

_ROOT = Path(__file__).resolve().parents[2]
_OUT = _ROOT / "outputs"

_CONF_WEIGHT = {"Verified": 1.0, "High": 0.85, "Medium": 0.55,
                "Low": 0.30, "Unmapped": 0.20}


@dataclass
class RiskConfig:
    contamination: float = 0.05
    w_model: float = 0.55
    w_rule: float = 0.45
    rule_cap: float = 4.0            # rule weight sum that saturates rule_norm=1
    critical_floor: int = 70         # min risk when a critical flag fires
    min_rows_for_iforest: int = 12
    # risk-level cutoffs
    lvl_critical: int = 75
    lvl_high: int = 55
    lvl_medium: int = 35
    random_state: int = 42
    # fast-resolution HIERARCHICAL baseline (asset -> asset_type -> issue_type)
    fast_ratio: float = 0.25              # flag if resolution < ratio * baseline
    min_asset_baseline_n: int = 3         # level 1: asset_id + issue_type
    min_type_baseline_n: int = 5          # level 2: asset_type + issue_type
    # recent-repeat fast-resolution rule (additive, owner requirement)
    recent_repeat_window_days: int = 30   # "another repair within N days"
    allow_related_issue_groups: bool = False  # False -> require identical issue_type
    flag_weights: Dict[str, int] = field(default_factory=lambda: {
        # ticket
        "repeated_issue": 1, "fast_resolution": 2, "tenant_rejected": 2,
        "cost_over_expected": 3, "process_edit_unlocked": 3,
        "recent_repeat_fast_resolution": 3,
        # asset
        "high_frequency": 2, "repeated_same_issue": 2, "severe_repeat": 3,
        "high_vs_age": 2, "peer_outlier": 2, "severe_peer_outlier": 3,
        # tenant
        "high_ticket_count": 2, "high_rejection_rate": 3, "freq_outlier": 2,
        # technician
        "high_reject_rate": 3, "fast_and_costly": 2, "volume_dominant": 1,
    })


# --------------------------------------------------------------------------- #
# small numeric helpers
# --------------------------------------------------------------------------- #
def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _dt(series) -> pd.Series:
    d = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return d.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return pd.to_datetime(series, errors="coerce")


def _hours_str(h: float) -> str:
    """Human-friendly duration: minutes below 1h, else hours."""
    if pd.isna(h):
        return "—"
    if h < 1:
        return f"{int(round(h * 60))} minutes"
    return f"{h:.1f} hours"


def _robust_z(s: pd.Series) -> pd.Series:
    """Median/MAD z-score, NaN-safe (0 where undefined)."""
    x = _num(s)
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        std = x.std()
        if not np.isfinite(std) or std == 0:
            return pd.Series(0.0, index=s.index)
        return ((x - x.mean()) / std).fillna(0.0)
    return (0.6745 * (x - med) / mad).fillna(0.0)


def _rank_norm(s: pd.Series) -> pd.Series:
    """Rank to 0-1 (higher = larger)."""
    r = s.rank(method="average", na_option="keep")
    n = r.notna().sum()
    if n <= 1:
        return pd.Series(0.0, index=s.index).fillna(0.0)
    return ((r - 1) / (n - 1)).fillna(0.0)


def _iforest_anomaly(feat: pd.DataFrame, cfg: RiskConfig) -> pd.Series:
    """Return 0-1 anomaly (1 = most anomalous). Median-impute, RobustScale."""
    if len(feat) < cfg.min_rows_for_iforest or feat.shape[1] == 0:
        return pd.Series(0.0, index=feat.index)
    X = feat.apply(_num)
    X = X.fillna(X.median(numeric_only=True))
    X = X.fillna(0.0)
    # drop zero-variance columns (add nothing, can destabilise scaler)
    keep = X.columns[X.std(numeric_only=True) > 0]
    if len(keep) == 0:
        return pd.Series(0.0, index=feat.index)
    Xs = RobustScaler().fit_transform(X[keep])
    model = IsolationForest(
        n_estimators=300, contamination=cfg.contamination,
        random_state=cfg.random_state,
    )
    model.fit(Xs)
    raw = -model.score_samples(Xs)          # higher = more anomalous
    return _rank_norm(pd.Series(raw, index=feat.index))


# --------------------------------------------------------------------------- #
# risk assembly shared by every grain
# --------------------------------------------------------------------------- #
def _assemble(
    ids: pd.Series,
    anomaly: pd.Series,
    flags_per_row: List[List[str]],
    reasons_per_row: List[List[str]],
    cfg: RiskConfig,
    confidence: Optional[pd.Series] = None,
    critical_flags: Optional[set] = None,
) -> pd.DataFrame:
    critical_flags = critical_flags or set()
    conf = confidence if confidence is not None else pd.Series(1.0, index=ids.index)
    rows = []
    for i, idx in enumerate(ids.index):
        flags = flags_per_row[i]
        rule_sum = sum(cfg.flag_weights.get(f, 1) for f in flags)
        rule_norm = min(1.0, rule_sum / cfg.rule_cap)
        anom = float(anomaly.loc[idx]) if idx in anomaly.index else 0.0
        c = float(conf.loc[idx]) if idx in conf.index else 1.0
        base = cfg.w_model * c * anom + cfg.w_rule * rule_norm
        score = round(100.0 * base, 1)
        if flags and (set(flags) & critical_flags):
            score = max(score, float(cfg.critical_floor))
        score = float(min(100.0, max(0.0, score)))
        rows.append({
            "entity_id": ids.loc[idx],
            "risk_score": score,
            "anomaly_score": round(anom, 3),
            "flags": ";".join(flags),
            "explanation": " | ".join(reasons_per_row[i]) if reasons_per_row[i] else "",
        })
    out = pd.DataFrame(rows)
    out["risk_level"] = out["risk_score"].map(lambda s: _level(s, cfg))
    return out[["entity_id", "risk_score", "risk_level", "anomaly_score",
                "flags", "explanation"]].sort_values("risk_score", ascending=False)


def _level(score: float, cfg: RiskConfig) -> str:
    if score >= cfg.lvl_critical:
        return "Critical"
    if score >= cfg.lvl_high:
        return "High"
    if score >= cfg.lvl_medium:
        return "Medium"
    return "Low"


# --------------------------------------------------------------------------- #
# asset-issue historical resolution baseline (additive rule signal)
# --------------------------------------------------------------------------- #
def _ticket_asset_type(base: Path) -> Dict[str, str]:
    """ticket_id -> mapped_asset_type from the Phase-1 mapping artefact."""
    p = base / "ticket_asset_mapping.csv"
    if not p.exists():
        return {}
    mp = pd.read_csv(p, dtype=str)
    if "mapped_asset_type" not in mp.columns:
        return {}
    out = {}
    for tid, at in zip(mp["ticket_id"].astype(str), mp["mapped_asset_type"]):
        a = str(at).strip()
        if a and a.lower() not in ("nan", "none", ""):
            out[tid] = a
    return out


def _ticket_asset_id(base: Path) -> Dict[str, str]:
    """ticket_id -> mapped_asset_id from the Phase-1 mapping artefact."""
    p = base / "ticket_asset_mapping.csv"
    if not p.exists():
        return {}
    mp = pd.read_csv(p, dtype=str)
    if "mapped_asset_id" not in mp.columns:
        return {}
    out = {}
    for tid, aid in zip(mp["ticket_id"].astype(str), mp["mapped_asset_id"]):
        a = str(aid).strip()
        if a and a.lower() not in ("nan", "none", ""):
            out[tid] = a
    return out


def asset_issue_resolution_baseline(tk: pd.DataFrame, at_map: Dict[str, str],
                                    cfg: RiskConfig) -> pd.DataFrame:
    """Historical median resolution time per (asset_type, issue_type).

    Built from tickets that HAVE both a resolution time and a mapped asset type.
    Only combinations with >= cfg.min_type_baseline_n samples are considered
    trustworthy; the caller ignores the rest. Exported for transparency.
    """
    d = tk.copy()
    d["resolution_hours"] = _num(d["resolution_hours"])
    d["asset_type"] = d["ticket_id"].astype(str).map(at_map)
    d = d[d["resolution_hours"].notna() & d["asset_type"].notna()
          & (d["asset_type"] != "") & (d["issue_type"].fillna("") != "")]
    if d.empty:
        return pd.DataFrame(columns=["asset_type", "issue_type",
                                     "baseline_median_hours", "n"])
    g = d.groupby(["asset_type", "issue_type"])["resolution_hours"]
    base = g.median().rename("baseline_median_hours").reset_index()
    base["n"] = g.size().values
    base["trustworthy"] = base["n"] >= cfg.min_type_baseline_n
    return base.sort_values(["asset_type", "issue_type"]).reset_index(drop=True)


def _hier_baseline_lookups(tk: pd.DataFrame, at_map: Dict[str, str],
                           aid_map: Dict[str, str], cfg: RiskConfig):
    """Return the three baseline lookups for hierarchical selection.

    L1 (asset)      : (asset_id,   issue_type) -> (median, n)  n >= min_asset_baseline_n
    L2 (asset_type) : (asset_type, issue_type) -> (median, n)  n >= min_type_baseline_n
    L3 (issue_type) : issue_type -> (median, n)                any n >= 1
    """
    d = tk.copy()
    d["resolution_hours"] = _num(d["resolution_hours"])
    d = d[d["resolution_hours"].notna() & (d["issue_type"].fillna("") != "")]
    d["_aid"] = d["ticket_id"].astype(str).map(aid_map)
    d["_atype"] = d["ticket_id"].astype(str).map(at_map)

    l3_med = d.groupby("issue_type")["resolution_hours"].median().to_dict()
    l3_n = d.groupby("issue_type")["resolution_hours"].size().to_dict()

    dt = d[d["_atype"].notna() & (d["_atype"] != "")]
    gt = dt.groupby(["_atype", "issue_type"])["resolution_hours"]
    l2 = {k: (v, int(gt.size()[k])) for k, v in gt.median().items()
          if gt.size()[k] >= cfg.min_type_baseline_n}

    da = d[d["_aid"].notna() & (d["_aid"] != "")]
    ga = da.groupby(["_aid", "issue_type"])["resolution_hours"]
    l1 = {k: (v, int(ga.size()[k])) for k, v in ga.median().items()
          if ga.size()[k] >= cfg.min_asset_baseline_n}
    return l1, l2, (l3_med, l3_n)


# --------------------------------------------------------------------------- #
# TICKET grain
# --------------------------------------------------------------------------- #
def score_tickets(tk: pd.DataFrame, cfg: RiskConfig, loader=None,
                  at_map: Optional[Dict[str, str]] = None,
                  aid_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    tk = tk.copy()
    tk["resolution_hours"] = _num(tk["resolution_hours"])
    tk["cost_difference"] = _num(tk["cost_difference"])
    tk["tenant_ticket_count"] = _num(tk["tenant_ticket_count"])
    tk["repeat_issue_flag"] = _num(tk["repeat_issue_flag"]).fillna(0)

    # ---- HIERARCHICAL fast-resolution baseline selection -------------------- #
    # For every ticket pick the highest-confidence baseline available:
    #   1) asset_id + issue_type   (>= min_asset_baseline_n repairs)  source=asset
    #   2) asset_type + issue_type (>= min_type_baseline_n)           source=asset_type
    #   3) global issue_type median (fallback, existing behaviour)    source=issue_type
    # The chosen baseline value, source and sample size are stored per ticket;
    # a single `fast_resolution` flag fires when resolution < fast_ratio*baseline.
    if at_map is None:
        at_map = _ticket_asset_type(_OUT)
    if aid_map is None:
        aid_map = _ticket_asset_id(_OUT)
    l1, l2, (l3_med, l3_n) = _hier_baseline_lookups(tk, at_map, aid_map, cfg)

    tk["_atype"] = tk["ticket_id"].astype(str).map(at_map)
    tk["_aid"] = tk["ticket_id"].astype(str).map(aid_map)
    base_val, base_src, base_n = [], [], []
    for _, r in tk.iterrows():
        issue = r["issue_type"]
        hit = l1.get((r["_aid"], issue))
        if hit:
            base_val.append(hit[0]); base_src.append("asset"); base_n.append(hit[1])
            continue
        hit = l2.get((r["_atype"], issue))
        if hit:
            base_val.append(hit[0]); base_src.append("asset_type"); base_n.append(hit[1])
            continue
        gm = l3_med.get(issue)
        if gm is not None and gm > 0:
            base_val.append(gm); base_src.append("issue_type")
            base_n.append(int(l3_n.get(issue, 0)))
        else:
            base_val.append(np.nan); base_src.append(""); base_n.append(0)
    tk["_base"] = base_val
    tk["_base_src"] = base_src
    tk["_base_n"] = base_n

    # optional raw-ticket join: rejection + process-edit + completion date
    rejected = pd.Series(False, index=tk.index)
    edit_unlocked = pd.Series(False, index=tk.index)
    asset_code_map: Dict[str, str] = {}
    if loader is not None:
        try:
            raw = loader.maintenance_tickets()
            rid = raw["id"].astype(str)
            appr = dict(zip(rid, raw["tenant_approved"].astype(str).str.strip().str.lower()))
            unl = dict(zip(rid, raw["resolution_edit_unlocked_at"].astype(str).str.strip().str.lower()))
            resolved = dict(zip(rid, _dt(raw.get("resolved_at"))))
            tid = tk["ticket_id"].astype(str)
            rejected = tid.map(lambda x: appr.get(x, "") == "false")
            edit_unlocked = tid.map(
                lambda x: unl.get(x, "") not in ("", "nan", "none", "null", "nat"))
            tk["_resolved"] = tid.map(resolved)
        except Exception:  # noqa: BLE001
            tk["_resolved"] = pd.NaT
        try:
            am = loader.asset_master()
            asset_code_map = {str(k).strip(): str(v).strip()
                              for k, v in zip(am["id"], am.get("asset_code", am["id"]))}
        except Exception:  # noqa: BLE001
            asset_code_map = {}
    else:
        tk["_resolved"] = pd.NaT

    # ---- recent-repeat fast-resolution: previous completed repair per asset -- #
    # For each ticket find the immediately-preceding completed repair on the SAME
    # asset (ordered by resolved_at). Stored regardless of the alert; the alert
    # itself has extra conditions (window + same issue + fast vs baseline).
    tk["previous_repair_date"] = pd.NaT
    tk["previous_ticket_id"] = ""
    tk["_prev_issue"] = ""
    tk["days_since_previous_repair"] = np.nan
    has_hist = tk["_aid"].fillna("").ne("") & tk["_resolved"].notna()
    sub = tk[has_hist].sort_values("_resolved")
    if not sub.empty:
        grp = sub.groupby("_aid", sort=False)
        prev_date = grp["_resolved"].shift(1)
        prev_tid = grp["ticket_id"].shift(1)
        prev_issue = grp["issue_type"].shift(1)
        tk.loc[sub.index, "previous_repair_date"] = prev_date
        tk.loc[sub.index, "previous_ticket_id"] = prev_tid.fillna("")
        tk.loc[sub.index, "_prev_issue"] = prev_issue.fillna("")
        days = (sub["_resolved"] - prev_date).dt.total_seconds() / 86400.0
        tk.loc[sub.index, "days_since_previous_repair"] = days.values

    # anomaly feature block
    feat = pd.DataFrame({
        "resolution_hours": tk["resolution_hours"],
        "repeat_issue_flag": tk["repeat_issue_flag"],
        "tenant_ticket_count": tk["tenant_ticket_count"],
        "rejected": rejected.astype(int),
        "cost_difference": tk["cost_difference"],
    })
    anomaly = _iforest_anomaly(feat, cfg)

    flags, reasons = [], []
    for idx, r in tk.iterrows():
        f, why = [], []
        if r["repeat_issue_flag"] == 1:
            f.append("repeated_issue")
            why.append("Repeated issue at this location")
        rh, base, src, bn = (r["resolution_hours"], r["_base"],
                             r["_base_src"], r["_base_n"])
        if pd.notna(rh) and pd.notna(base) and base > 0 and rh < cfg.fast_ratio * base:
            f.append("fast_resolution")
            scope = {
                "asset": f"this asset ({r['_atype']}/{r['issue_type']})",
                "asset_type": f"{r['_atype']}/{r['issue_type']}",
                "issue_type": f"{r['issue_type']}",
            }.get(src, r["issue_type"])
            why.append(f"Resolved in {rh:.1f}h vs {base:.1f}h {src} baseline "
                       f"for {scope} (n={int(bn)})")
        # NEW owner rule: same asset repaired again within window + much faster
        dsp = r["days_since_previous_repair"]
        same_issue = (cfg.allow_related_issue_groups
                      or (r["_prev_issue"] and r["_prev_issue"] == r["issue_type"]))
        if pd.notna(rh) and pd.notna(base) and base > 0 and pd.notna(dsp) \
                and dsp <= cfg.recent_repeat_window_days and same_issue \
                and rh < cfg.fast_ratio * base:
            f.append("recent_repeat_fast_resolution")
            acode = asset_code_map.get(str(r["_aid"]), str(r["_aid"])[:8])
            why.append(
                f"Asset {acode} was repaired again after {dsp:.0f} days. "
                f"Historical median resolution time ({src} baseline, n={int(bn)}) "
                f"is {base:.1f}h, but this repair was completed in "
                f"{_hours_str(rh)}. Investigation recommended.")
        if bool(rejected.loc[idx]):
            f.append("tenant_rejected")
            why.append("Tenant rejected / did not approve the resolution")
        cd, ec = r["cost_difference"], _num(pd.Series([r.get("expected_cost")])).iloc[0]
        if pd.notna(cd) and pd.notna(ec) and ec > 0 and cd > 0.5 * ec:
            f.append("cost_over_expected")
            why.append(f"Cost ₹{r.get('cost')} exceeds expected ₹{ec:.0f} (+₹{cd:.0f})")
        if bool(edit_unlocked.loc[idx]):
            f.append("process_edit_unlocked")
            why.append("Resolution was edit-unlocked after closure")
        flags.append(f)
        reasons.append(why)

    conf = tk["asset_mapping_confidence"].map(_CONF_WEIGHT).fillna(0.3)
    out = _assemble(
        tk["ticket_id"], anomaly, flags, reasons, cfg, confidence=conf,
        critical_flags={"cost_over_expected", "process_edit_unlocked",
                        "recent_repeat_fast_resolution"},
    )
    # attach baseline + recurrence provenance (additive cols; scoring untouched)
    meta = pd.DataFrame({
        "entity_id": tk["ticket_id"].values,
        "previous_repair_date": tk["previous_repair_date"].values,
        "days_since_previous_repair": tk["days_since_previous_repair"].round(1).values,
        "previous_ticket_id": tk["previous_ticket_id"].values,
        "historical_resolution_baseline": pd.Series(tk["_base"].values).round(2),
        "baseline_source": tk["_base_src"].values,
        "baseline_sample_size": tk["_base_n"].values,
    })
    return out.merge(meta, on="entity_id", how="left")


# --------------------------------------------------------------------------- #
# ASSET grain
# --------------------------------------------------------------------------- #
def score_assets(a: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    a = a.copy()
    for c in ["total_tickets", "repeat_issue_count", "max_same_issue_repeats",
              "tickets_per_month", "asset_age_days", "avg_resolution_hours"]:
        a[c] = _num(a[c])

    # peer group = asset_type; z of ticket volume within type
    a["_peer_med"] = a.groupby("asset_type")["total_tickets"].transform("median")
    a["_peer_z"] = a.groupby("asset_type")["total_tickets"].transform(_robust_z)
    freq_hi = a["total_tickets"].quantile(0.90)

    feat = a[["total_tickets", "repeat_issue_count", "tickets_per_month",
              "asset_age_days", "avg_resolution_hours"]]
    anomaly = _iforest_anomaly(feat, cfg)

    flags, reasons = [], []
    for _, r in a.iterrows():
        f, why = [], []
        if pd.notna(r["total_tickets"]) and r["total_tickets"] >= max(freq_hi, 4):
            f.append("high_frequency")
            why.append(f"High ticket count ({int(r['total_tickets'])})")
        msr = r["max_same_issue_repeats"]
        if pd.notna(msr) and msr >= 10:
            f.append("severe_repeat")
            why.append(f"Same issue repeated {int(msr)} times")
        elif pd.notna(msr) and msr >= 4:
            f.append("repeated_same_issue")
            why.append(f"Same issue repeated {int(msr)} times")
        if pd.notna(r["asset_age_days"]) and r["asset_age_days"] >= 0:
            age_m = max(r["asset_age_days"] / 30.44, 0.5)
            per = r["total_tickets"] / age_m
            if per >= 2 and r["total_tickets"] >= 3:
                f.append("high_vs_age")
                why.append(f"{int(r['total_tickets'])} tickets on a "
                           f"{int(r['asset_age_days'])}-day-old asset")
        z = r["_peer_z"]
        if pd.notna(z) and z >= 4:
            f.append("severe_peer_outlier")
            why.append(f"Ticket count {_x(r['total_tickets'], r['_peer_med'])} "
                       f"the {r['asset_type']} peer median")
        elif pd.notna(z) and z >= 2.5:
            f.append("peer_outlier")
            why.append(f"Ticket count {_x(r['total_tickets'], r['_peer_med'])} "
                       f"the {r['asset_type']} peer median")
        flags.append(f)
        reasons.append(why)

    return _assemble(
        a["asset_id"], anomaly, flags, reasons, cfg,
        critical_flags={"severe_repeat", "severe_peer_outlier"},
    )


def _x(v, base) -> str:
    if pd.notna(base) and base and base > 0 and pd.notna(v):
        return f"{v / base:.1f}x"
    return "well above"


# --------------------------------------------------------------------------- #
# TENANT grain
# --------------------------------------------------------------------------- #
def score_tenants(t: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    t = t.copy()
    for c in ["total_tickets", "avg_resolution_hours", "rejection_rate",
              "rejection_count"]:
        t[c] = _num(t[c])
    hi = t["total_tickets"].quantile(0.95)
    t["_z"] = _robust_z(t["total_tickets"])
    rej_mean = t["rejection_rate"].mean()
    rej_std = t["rejection_rate"].std()

    feat = t[["total_tickets", "avg_resolution_hours", "rejection_rate"]]
    anomaly = _iforest_anomaly(feat, cfg)

    flags, reasons = [], []
    for _, r in t.iterrows():
        f, why = [], []
        if pd.notna(r["total_tickets"]) and r["total_tickets"] >= max(hi, 5):
            f.append("high_ticket_count")
            why.append(f"Raised {int(r['total_tickets'])} tickets (population p95={hi:.0f})")
        rr = r["rejection_rate"]
        thresh = (rej_mean + 2 * rej_std) if np.isfinite(rej_std) else 0.2
        if pd.notna(rr) and rr > max(thresh, 0.15) and r["rejection_count"] >= 2:
            f.append("high_rejection_rate")
            why.append(f"Rejection rate {rr:.0%} ({int(r['rejection_count'])} rejected) "
                       f"above avg {rej_mean:.0%}")
        if pd.notna(r["_z"]) and r["_z"] >= 3:
            f.append("freq_outlier")
            why.append("Ticket frequency far above tenant peers")
        flags.append(f)
        reasons.append(why)

    return _assemble(
        t["tenant_id"], anomaly, flags, reasons, cfg,
        critical_flags={"high_rejection_rate"},
    )


# --------------------------------------------------------------------------- #
# TECHNICIAN grain (small n -> rules only, no IForest)
# --------------------------------------------------------------------------- #
def score_technicians(t: pd.DataFrame, cfg: RiskConfig) -> pd.DataFrame:
    t = t.copy()
    for c in ["tickets_closed", "avg_resolution_time", "average_cost",
              "rejected_count"]:
        t[c] = _num(t[c])
    t["_reject_rate"] = (t["rejected_count"] / t["tickets_closed"]).replace(
        [np.inf, -np.inf], np.nan)
    med_res = t["avg_resolution_time"].median()
    med_cost = t["average_cost"].median()
    rr_mean = t["_reject_rate"].mean()
    vol_hi = t["tickets_closed"].quantile(0.75)

    anomaly = pd.Series(0.0, index=t.index)   # n too small for IForest
    flags, reasons = [], []
    for _, r in t.iterrows():
        f, why = [], []
        if pd.notna(r["_reject_rate"]) and pd.notna(rr_mean) and \
                r["_reject_rate"] > max(rr_mean * 1.5, 0.05) and r["rejected_count"] >= 3:
            f.append("high_reject_rate")
            why.append(f"Rejection rate {r['_reject_rate']:.0%} "
                       f"({int(r['rejected_count'])} of {int(r['tickets_closed'])})")
        if pd.notna(r["avg_resolution_time"]) and pd.notna(med_res) and \
                pd.notna(r["average_cost"]) and pd.notna(med_cost) and \
                r["avg_resolution_time"] < med_res and r["average_cost"] > med_cost:
            f.append("fast_and_costly")
            why.append("Faster than peers yet higher average cost")
        if pd.notna(r["tickets_closed"]) and r["tickets_closed"] >= max(vol_hi, 50):
            f.append("volume_dominant")
            why.append(f"Closed {int(r['tickets_closed'])} tickets (dominant share)")
        flags.append(f)
        reasons.append(why)

    return _assemble(t["technician_id"], anomaly, flags, reasons, cfg,
                     critical_flags={"high_reject_rate"})


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _validate(risks: Dict[str, pd.DataFrame], cfg: RiskConfig) -> pd.DataFrame:
    recs = []

    def add(m, v):
        recs.append({"metric": m, "value": v})

    all_reasons_flags: List[str] = []
    for name, df in risks.items():
        add(f"rows.{name}", len(df))
        lv = df["risk_level"].value_counts()
        for level in ("Critical", "High", "Medium", "Low"):
            add(f"level.{name}.{level}", int(lv.get(level, 0)))
        add(f"high_or_critical.{name}",
            int((df["risk_level"].isin(["High", "Critical"])).sum()))
        s = df["risk_score"]
        for q, lbl in [(.5, "p50"), (.9, "p90"), (.99, "p99")]:
            add(f"score.{name}.{lbl}", round(float(s.quantile(q)), 1) if len(s) else 0.0)
        add(f"score.{name}.max", round(float(s.max()), 1) if len(s) else 0.0)
        for fl in df["flags"]:
            all_reasons_flags.extend([x for x in str(fl).split(";") if x])

    top = pd.Series(all_reasons_flags).value_counts().head(10)
    for flag, c in top.items():
        add(f"top_flag.{flag}", int(c))
    add("config.contamination", cfg.contamination)
    add("config.weights", f"model={cfg.w_model},rule={cfg.w_rule}")
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_all(cfg: Optional[RiskConfig] = None, loader=None,
              out_dir: Optional[str] = None
              ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
    cfg = cfg or RiskConfig()
    base = Path(out_dir) if out_dir else _OUT

    tk = pd.read_csv(base / "maintenance_ticket_features.csv", dtype={"ticket_id": str})
    a = pd.read_csv(base / "maintenance_asset_features.csv")
    tn = pd.read_csv(base / "maintenance_tenant_features.csv")
    tc = pd.read_csv(base / "maintenance_technician_features.csv")

    if loader is None:
        try:
            from data_loader import DataLoader
            loader = DataLoader()
        except Exception:  # noqa: BLE001
            loader = None

    # hierarchical fast-resolution baseline maps (asset -> asset_type -> issue)
    at_map = _ticket_asset_type(base)
    aid_map = _ticket_asset_id(base)
    baseline = asset_issue_resolution_baseline(tk, at_map, cfg)  # asset_type tier export

    risks = {
        "ticket": score_tickets(tk, cfg, loader=loader, at_map=at_map, aid_map=aid_map),
        "asset": score_assets(a, cfg),
        "tenant": score_tenants(tn, cfg),
        "technician": score_technicians(tc, cfg),
    }
    report = _validate(risks, cfg)
    tr = risks["ticket"]

    def _has(flag):
        return tr["flags"].fillna("").apply(lambda s: flag in str(s).split(";"))

    # baseline-source distribution among tickets that fired fast_resolution (exact)
    fired = tr[_has("fast_resolution")]
    src_counts = fired["baseline_source"].value_counts().to_dict()
    extra = [{"metric": f"fast_resolution.baseline_source.{k}", "value": int(v)}
             for k, v in src_counts.items()]
    extra.append({"metric": "fast_resolution.total_fired", "value": int(len(fired))})

    # recent-repeat validation summary
    dsp = _num(tr["days_since_previous_repair"])
    win = cfg.recent_repeat_window_days
    alerts = tr[_has("recent_repeat_fast_resolution")]
    extra += [
        {"metric": "recent_repeat.tickets_with_previous_repair", "value": int(dsp.notna().sum())},
        {"metric": f"recent_repeat.repaired_within_{win}d", "value": int((dsp <= win).sum())},
        {"metric": "recent_repeat.alerts", "value": int(len(alerts))},
    ]
    for q, lbl in [(.25, "p25"), (.5, "p50"), (.75, "p75"), (.9, "p90")]:
        v = dsp.dropna().quantile(q) if dsp.notna().any() else float("nan")
        extra.append({"metric": f"recent_repeat.days_between.{lbl}", "value": round(float(v), 1)})
    if dsp.notna().any():
        extra += [{"metric": "recent_repeat.days_between.min", "value": round(float(dsp.min()), 1)},
                  {"metric": "recent_repeat.days_between.max", "value": round(float(dsp.max()), 1)}]
    report = pd.concat([report, pd.DataFrame(extra)], ignore_index=True)

    base.mkdir(parents=True, exist_ok=True)
    for name in risks:
        risks[name].to_csv(base / f"maintenance_{name}_risk.csv", index=False)
    baseline.to_csv(base / "maintenance_asset_issue_baseline.csv", index=False)
    report.to_csv(base / "maintenance_anomaly_validation.csv", index=False)
    # dedicated recent-repeat validation summary
    pd.DataFrame([r for r in extra if str(r["metric"]).startswith("recent_repeat")]) \
        .to_csv(base / "maintenance_recent_repeat_validation.csv", index=False)
    return risks, report


def _print_summary(risks: Dict[str, pd.DataFrame]) -> None:
    print("=" * 64)
    print("PHASE 3 — MAINTENANCE INVESTIGATION RISK (rules + IsolationForest)")
    print("=" * 64)
    for name, df in risks.items():
        lv = df["risk_level"].value_counts()
        hc = int((df["risk_level"].isin(["High", "Critical"])).sum())
        print(f"  {name:<11}: {len(df):>4} scored | "
              f"Crit {int(lv.get('Critical',0)):>3}  High {int(lv.get('High',0)):>3}  "
              f"Med {int(lv.get('Medium',0)):>3}  Low {int(lv.get('Low',0)):>4} | "
              f"investigate={hc}")
    print("-" * 64)
    for name, df in risks.items():
        top = df.head(1)
        if len(top):
            r = top.iloc[0]
            print(f"  top {name}: score {r['risk_score']} [{r['risk_level']}] "
                  f"-> {r['explanation'][:70]}")
    print("=" * 64)


def main() -> None:
    risks, _ = build_all()
    _print_summary(risks)
    print(f"\nWrote 4 risk CSVs + maintenance_anomaly_validation.csv -> {_OUT}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
