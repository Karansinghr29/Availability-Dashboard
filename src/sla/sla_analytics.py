"""SLA Performance Analytics — operations dashboard module (deterministic).

Standalone and independent: does NOT import or touch the Historical Repair
Audit, Future Verification, fake-repair detection, baseline engine, or anomaly
analytics. Pure descriptive SLA reporting. No ML, no prediction.

Definitions (per ticket)
------------------------
    SLA hours        = issue_types.sla_hours (fallback cfg.default_sla_hours)
    SLA deadline     = created_at + SLA hours
    Resolution hours = resolved_at - created_at
    Closure hours    = closed_at  - created_at
    Resolution delay = Resolution hours - SLA hours
    Closure  delay   = Closure hours  - SLA hours
    Status (delay<=0 -> Within SLA, delay>0 -> SLA Breached)
The primary SLA status uses CLOSURE when closed_at exists, else resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_OUT = _ROOT / "outputs"


@dataclass
class SLAConfig:
    default_sla_hours: float = 24.0
    use_issue_type_sla: bool = True
    sla_overrides: Dict[str, float] = field(default_factory=dict)
    delay_buckets: tuple = (6.0, 24.0, 72.0)


def _s(v) -> str:
    t = "" if v is None else str(v).strip()
    return "" if t.lower() in ("", "nan", "none", "null", "<na>", "nat") else t


def _dt(series) -> pd.Series:
    d = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return d.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return pd.to_datetime(series, errors="coerce")


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _hours(a: pd.Series, b: pd.Series) -> pd.Series:
    """(a - b) in hours; negative durations dropped to NaN."""
    h = (a - b).dt.total_seconds() / 3600.0
    return h.where(h >= 0)


# --------------------------------------------------------------------------- #
# per-ticket base frame
# --------------------------------------------------------------------------- #
def _base(loader, cfg: SLAConfig):
    t = loader.maintenance_tickets().copy()

    it = loader.issue_types()
    sla_map, iname = {}, {}
    if not it.empty:
        sla_map = {_s(k): float(v) for k, v in zip(it["id"], _num(it.get("sla_hours")))
                   if pd.notna(v)}
        iname = dict(zip(it["id"].map(_s), it["name"].map(_s)))
    sla_map.update({k: float(v) for k, v in cfg.sla_overrides.items()})

    # resolver (technician) from ticket_resolutions.resolved_by — the ONLY
    # reliable actor. closed_at is NOT attributed to any person (see data audit).
    resolver = {}
    try:
        tr = loader.ticket_resolutions()
        resolver = {_s(k): _s(v) for k, v in zip(tr["ticket_id"], tr.get("resolved_by"))
                    if _s(k) and _s(v)}
    except Exception:  # noqa: BLE001
        resolver = {}

    apt = loader.apartment_master()
    apt_code = dict(zip(apt["id"].map(_s), apt["apartment_code"].map(_s))) if not apt.empty else {}
    am, atn = loader.asset_master(), loader.asset_types()
    tname = dict(zip(atn["id"].map(_s), atn["name"].map(_s))) if not atn.empty else {}
    acode = dict(zip(am["id"].map(_s), am["asset_code"].map(_s))) \
        if (not am.empty and "asset_code" in am.columns) else {}
    atype = {_s(k): tname.get(_s(v), "") for k, v in zip(am["id"], am.get("asset_type_id"))} \
        if (not am.empty and "asset_type_id" in am.columns) else {}

    created, resolved, closed = _dt(t.get("created_at")), _dt(t.get("resolved_at")), _dt(t.get("closed_at"))
    issue = t["issue_type_id"].map(_s)
    sla_h = issue.map(sla_map).fillna(cfg.default_sla_hours) if cfg.use_issue_type_sla \
        else pd.Series(cfg.default_sla_hours, index=t.index)

    df = pd.DataFrame({
        "ticket_number": t.get("ticket_number").map(_s),
        "ticket_id": t["id"].map(_s),
        "issue_type": issue.map(lambda x: iname.get(x, x)),
        "apartment": t["apartment_id"].map(lambda x: apt_code.get(_s(x), _s(x)[:8])),
        "asset": t["asset_id"].map(lambda x: acode.get(_s(x), "") if _s(x) else ""),
        "asset_type": t["asset_id"].map(lambda x: atype.get(_s(x), "") if _s(x) else ""),
        "technician": t["assigned_to"].map(_s),
        "resolver": t["id"].map(lambda x: resolver.get(_s(x), "")),
        "created_at": created,
        "resolved_at": resolved,
        "closed_at": closed,
        "sla_hours": sla_h.round(2),
    })
    df["sla_deadline"] = created + pd.to_timedelta(sla_h, unit="h")
    df["resolution_hours"] = _hours(resolved, created).round(2)   # raw (unchanged)
    df["closure_hours"] = _hours(closed, created).round(2)
    df["resolution_delay"] = (df["resolution_hours"] - df["sla_hours"]).round(2)
    df["closure_delay"] = (df["closure_hours"] - df["sla_hours"]).round(2)
    # TRACKED resolution = resolved_at strictly AFTER created_at. resolved_at ==
    # created_at is a defaulted 0h (not a genuine resolution) and is treated as
    # untracked. Resolution KPIs/averages/details use the tracked version only;
    # closure metrics and the primary SLA comparison are NOT affected.
    df["resolution_tracked"] = df["resolution_hours"] > 0
    df["resolution_hours_tracked"] = df["resolution_hours"].where(df["resolution_tracked"])
    df["resolution_delay_tracked"] = (df["resolution_hours_tracked"] - df["sla_hours"]).round(2)
    # SECONDARY DIAGNOSTIC (does NOT affect primary SLA): time from resolution to
    # closure = closed_at - resolved_at. Computed ONLY where resolution is tracked
    # (resolved_at > created_at); untracked/defaulted rows stay NaN (shown as '—'),
    # never a spurious 0. Locates delay AFTER resolution (SLA -> Resolution vs
    # Resolution -> Closure). Negative spans (closed before resolved) drop to NaN.
    df["post_resolution_closure_hours"] = (
        _hours(closed, resolved).where(df["resolution_tracked"]).round(2))
    # PRIMARY SLA comparison = SLA vs CLOSURE (owner-approved). Status is set only
    # where closed_at exists; resolution is shown alongside for the delay story.
    df["primary_delay"] = df["closure_delay"]
    df["sla_status"] = df["primary_delay"].map(
        lambda d: "" if pd.isna(d) else ("SLA Breached" if d > 0 else "Within SLA"))
    df["breached"] = df["sla_status"] == "SLA Breached"
    df["within_sla"] = df["sla_status"] == "Within SLA"
    df["month"] = created.dt.strftime("%Y-%m")

    # evaluable = has created + at least one completion timestamp
    ev = df[created.notna() & (df["resolution_hours"].notna() | df["closure_hours"].notna())].copy()

    coverage = _coverage(df, created, resolved, closed)
    return ev, coverage


def _coverage(df, created, resolved, closed) -> pd.DataFrame:
    n = len(df)
    has_c, has_r, has_cl = created.notna(), resolved.notna(), closed.notna()
    ev = has_c & (has_r | has_cl)
    excl_no_created = int((~has_c).sum())
    excl_no_completion = int((has_c & ~(has_r | has_cl)).sum())
    return pd.DataFrame([
        {"metric": "total_tickets", "value": n},
        {"metric": "with_created_at", "value": int(has_c.sum())},
        {"metric": "with_resolved_at", "value": int(has_r.sum())},
        {"metric": "with_closed_at", "value": int(has_cl.sum())},
        {"metric": "evaluable_tickets", "value": int(ev.sum())},
        {"metric": "resolution_tracked_tickets",
         "value": int((df["resolution_hours"] > 0).sum())},
        {"metric": "excluded_total", "value": int((~ev).sum())},
        {"metric": "excluded_missing_created_at", "value": excl_no_created},
        {"metric": "excluded_missing_resolved_and_closed", "value": excl_no_completion},
    ])


# --------------------------------------------------------------------------- #
# aggregations
# --------------------------------------------------------------------------- #
def _perf(df: pd.DataFrame, by: str, label: str) -> pd.DataFrame:
    cols = [label, "tickets", "within_sla", "breached", "breach_pct",
            "avg_sla_hours", "avg_resolution_hours", "avg_closure_hours", "avg_delay_hours"]
    if df.empty or by not in df.columns:
        return pd.DataFrame(columns=cols)
    g = df.groupby(df[by].replace("", "(unknown)"))
    out = g.agg(tickets=("ticket_id", "size"),
                within_sla=("within_sla", "sum"),
                breached=("breached", "sum"),
                avg_sla_hours=("sla_hours", "mean"),
                avg_resolution_hours=("resolution_hours", "mean"),
                avg_closure_hours=("closure_hours", "mean"),
                avg_delay_hours=("primary_delay", "mean")).reset_index()
    out = out.rename(columns={by: label})
    closed = (out["within_sla"] + out["breached"]).replace(0, pd.NA)
    out["breach_pct"] = (100 * out["breached"] / closed).round(1)   # of closed tickets
    for c in ("avg_sla_hours", "avg_resolution_hours", "avg_closure_hours", "avg_delay_hours"):
        out[c] = out[c].round(1)
    return out.sort_values("breach_pct", ascending=False)[cols]


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("month")
    m = g.agg(tickets=("ticket_id", "size"),
              avg_sla_hours=("sla_hours", "mean"),
              avg_resolution_hours=("resolution_hours_tracked", "mean"),   # tracked only; NaN if none
              avg_closure_hours=("closure_hours", "mean"),
              avg_resolution_delay=("resolution_delay_tracked", "mean"),   # tracked only
              avg_post_resolution_closure=("post_resolution_closure_hours", "mean"),  # tracked only
              avg_closure_delay=("closure_delay", "mean"),
              resolution_tracked=("resolution_tracked", "sum"),
              within_sla=("within_sla", "sum"),
              breached=("breached", "sum")).reset_index()
    for c in ("avg_sla_hours", "avg_resolution_hours", "avg_closure_hours",
              "avg_resolution_delay", "avg_post_resolution_closure", "avg_closure_delay"):
        m[c] = m[c].round(1)
    # Breach % = of CLOSED tickets (primary SLA-vs-Closure comparison)
    closed = (m["within_sla"] + m["breached"]).replace(0, pd.NA)
    m["breach_pct"] = (100 * m["breached"] / closed).round(1)
    m["avg_closure_exceeds_sla"] = m["avg_closure_hours"] > m["avg_sla_hours"]
    m["avg_resolution_exceeds_sla"] = m["avg_resolution_hours"] > m["avg_sla_hours"]
    return m.sort_values("month").reset_index(drop=True)


def delay_distribution(df: pd.DataFrame, cfg: SLAConfig) -> pd.DataFrame:
    d = df["primary_delay"]
    c = cfg.delay_buckets
    labels = ["Within SLA", f"0-{c[0]:.0f}h Late", f"{c[0]:.0f}-{c[1]:.0f}h Late",
              f"{c[1]:.0f}-{c[2]:.0f}h Late", f"More than {c[2]:.0f}h Late"]
    counts = [int((d <= 0).sum()), int(((d > 0) & (d <= c[0])).sum()),
              int(((d > c[0]) & (d <= c[1])).sum()), int(((d > c[1]) & (d <= c[2])).sum()),
              int((d > c[2]).sum())]
    return pd.DataFrame({"delay_bucket": labels, "tickets": counts})


def ticket_details(df: pd.DataFrame) -> pd.DataFrame:
    # Built explicitly (not via rename) to avoid a duplicate 'resolution_hours'
    # column. Resolution columns use the TRACKED values (blank when untracked).
    out = pd.DataFrame({
        "ticket": df["ticket_number"],
        "technician_resolver": df["resolver"],
        "created_at": df["created_at"],
        "sla": df["sla_hours"],
        "sla_deadline": df["sla_deadline"],
        "resolved_at": df["resolved_at"],
        "resolution_hours": df["resolution_hours_tracked"],
        "resolution_delay": df["resolution_delay_tracked"],
        "closed_at": df["closed_at"],
        "closure_hours": df["closure_hours"],
        "post_resolution_closure_hours": df["post_resolution_closure_hours"],  # tracked only; NaN -> '—'
        "closure_delay": df["closure_delay"],
        "sla_status": df["sla_status"],
    })
    return out.sort_values("created_at").reset_index(drop=True)


def _kpis(df: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    n = len(df)
    breached = int(df["breached"].sum())
    recs = [
        {"metric": "total_tickets", "value": n},
        {"metric": "within_sla", "value": int(df["within_sla"].sum())},
        {"metric": "breached", "value": breached},
        {"metric": "breach_pct", "value": round(100 * breached / n, 1) if n else 0.0},
        {"metric": "avg_sla_hours", "value": round(float(df["sla_hours"].mean()), 1) if n else 0.0},
        {"metric": "avg_resolution_hours", "value": round(float(df["resolution_hours_tracked"].mean()), 1)
         if df["resolution_hours_tracked"].notna().any() else 0.0},   # tracked only
        {"metric": "resolution_tracked_tickets", "value": int(df["resolution_tracked"].sum())},
        {"metric": "avg_resolution_delay_hours",
         "value": round(float(df["resolution_delay_tracked"].mean()), 1)
         if df["resolution_delay_tracked"].notna().any() else 0.0},   # tracked only
        {"metric": "avg_post_resolution_closure_hours",
         "value": round(float(df["post_resolution_closure_hours"].mean()), 1)
         if df["post_resolution_closure_hours"].notna().any() else 0.0},   # tracked only
        {"metric": "avg_closure_hours", "value": round(float(df["closure_hours"].mean()), 1)
         if df["closure_hours"].notna().any() else 0.0},
        {"metric": "avg_delay_hours", "value": round(float(df["primary_delay"].mean()), 1)
         if df["primary_delay"].notna().any() else 0.0},
    ]
    # PRIMARY alert: avg CLOSURE > avg SLA for 3 consecutive months
    this_msg = ""
    if len(monthly) and bool(monthly.iloc[-1].get("avg_closure_exceeds_sla", False)):
        this_msg = "Average closure time exceeded the configured SLA this month."
    cflags = monthly.get("avg_closure_exceeds_sla", pd.Series(dtype=bool)).tolist()
    op_alert = len(cflags) >= 3 and all(bool(x) for x in cflags[-3:])
    op_msg = ("Operational Alert: Ticket closure time has exceeded the configured SLA "
              "for three consecutive months.") if op_alert else (this_msg or
              "Average closure time within the configured SLA.")
    # SECONDARY (optional) alert: avg RESOLUTION > avg SLA for 3 consecutive months
    rflags = monthly.get("avg_resolution_exceeds_sla", pd.Series(dtype=bool)).tolist()
    res_alert = len(rflags) >= 3 and all(bool(x) for x in rflags[-3:])
    res_msg = ("Note: Average resolution time has also exceeded the configured SLA for "
               "three consecutive months.") if res_alert else ""
    recs += [{"metric": "month_exceeded_message", "value": this_msg},
             {"metric": "operational_alert", "value": op_alert},
             {"metric": "operational_alert_message", "value": op_msg},
             {"metric": "resolution_alert", "value": res_alert},
             {"metric": "resolution_alert_message", "value": res_msg}]
    return pd.DataFrame(recs)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_sla_analytics(loader=None, cfg: Optional[SLAConfig] = None,
                        out_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    cfg = cfg or SLAConfig()
    base = Path(out_dir) if out_dir else _OUT
    if loader is None:
        from data_loader import DataLoader
        loader = DataLoader()

    df, coverage = _base(loader, cfg)
    monthly = monthly_summary(df)
    tables = {
        "monthly_summary": monthly,
        "apartment": _perf(df, "apartment", "apartment"),
        "asset_type": _perf(df[df["asset_type"] != ""], "asset_type", "asset_type"),
        "technician": _perf(df, "technician", "technician"),
        "issue_type": _perf(df, "issue_type", "issue_type"),
        "delay_distribution": delay_distribution(df, cfg),
        "ticket_details": ticket_details(df),
        "kpis": _kpis(df, monthly),
        "coverage": coverage,
    }
    base.mkdir(parents=True, exist_ok=True)
    for name, tbl in tables.items():
        tbl.to_csv(base / f"sla_{name}.csv", index=False)
    return tables


def main() -> None:
    tables = build_sla_analytics()
    k = {r["metric"]: r["value"] for _, r in tables["kpis"].iterrows()}
    cov = {r["metric"]: r["value"] for _, r in tables["coverage"].iterrows()}
    print("=" * 64)
    print("SLA PERFORMANCE ANALYTICS")
    print("=" * 64)
    print(f"  Coverage: {cov['evaluable_tickets']}/{cov['total_tickets']} evaluable | "
          f"created {cov['with_created_at']} · resolved {cov['with_resolved_at']} · "
          f"closed {cov['with_closed_at']}")
    print(f"  Excluded {cov['excluded_total']} "
          f"(no created_at: {cov['excluded_missing_created_at']}, "
          f"no resolved & no closed: {cov['excluded_missing_resolved_and_closed']})")
    print(f"  Avg SLA {k['avg_sla_hours']}h · Avg Resolution {k['avg_resolution_hours']}h · "
          f"Avg Closure {k['avg_closure_hours']}h · Avg Delay {k['avg_delay_hours']}h · "
          f"Breach {k['breach_pct']}%")
    print(f"  {k['operational_alert_message']}")
    print("-" * 64)
    print(tables["monthly_summary"].to_string(index=False))
    print(f"\nWrote sla_*.csv -> {_OUT}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
