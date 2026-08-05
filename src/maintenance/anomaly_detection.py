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
    flag_weights: Dict[str, int] = field(default_factory=lambda: {
        # ticket
        "repeated_issue": 1, "fast_resolution": 2, "tenant_rejected": 2,
        "cost_over_expected": 3, "process_edit_unlocked": 3,
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
# TICKET grain
# --------------------------------------------------------------------------- #
def score_tickets(tk: pd.DataFrame, cfg: RiskConfig,
                  loader=None) -> pd.DataFrame:
    tk = tk.copy()
    tk["resolution_hours"] = _num(tk["resolution_hours"])
    tk["cost_difference"] = _num(tk["cost_difference"])
    tk["tenant_ticket_count"] = _num(tk["tenant_ticket_count"])
    tk["repeat_issue_flag"] = _num(tk["repeat_issue_flag"]).fillna(0)

    # issue-type median resolution (for the "unusually fast" flag)
    med = tk.groupby("issue_type")["resolution_hours"].median()
    tk["_issue_med"] = tk["issue_type"].map(med)

    # optional raw-ticket join: rejection + process-edit (not in feature CSV)
    rejected = pd.Series(False, index=tk.index)
    edit_unlocked = pd.Series(False, index=tk.index)
    if loader is not None:
        try:
            raw = loader.maintenance_tickets()
            rid = raw["id"].astype(str)
            appr = dict(zip(rid, raw["tenant_approved"].astype(str).str.strip().str.lower()))
            unl = dict(zip(rid, raw["resolution_edit_unlocked_at"].astype(str).str.strip().str.lower()))
            tid = tk["ticket_id"].astype(str)
            rejected = tid.map(lambda x: appr.get(x, "") == "false")
            edit_unlocked = tid.map(
                lambda x: unl.get(x, "") not in ("", "nan", "none", "null", "nat"))
        except Exception:  # noqa: BLE001
            pass

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
        rh, m = r["resolution_hours"], r["_issue_med"]
        if pd.notna(rh) and pd.notna(m) and m > 0 and rh < 0.25 * m:
            f.append("fast_resolution")
            why.append(f"Resolved in {rh:.1f}h vs {m:.1f}h typical for {r['issue_type']}")
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
    return _assemble(
        tk["ticket_id"], anomaly, flags, reasons, cfg, confidence=conf,
        critical_flags={"cost_over_expected", "process_edit_unlocked"},
    )


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

    risks = {
        "ticket": score_tickets(tk, cfg, loader=loader),
        "asset": score_assets(a, cfg),
        "tenant": score_tenants(tn, cfg),
        "technician": score_technicians(tc, cfg),
    }
    report = _validate(risks, cfg)

    base.mkdir(parents=True, exist_ok=True)
    for name in risks:
        risks[name].to_csv(base / f"maintenance_{name}_risk.csv", index=False)
    report.to_csv(base / "maintenance_anomaly_validation.csv", index=False)
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
