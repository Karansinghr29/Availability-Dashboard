"""Asset Repair Verification Queue (owner business rule).

Deterministic, explainable, investigation-only. NOT an anomaly model.

Owner's question, verbatim intent:
    "Why was this ticket solved much faster than THIS ASSET is normally
     repaired?"

Business logic
--------------
1. Every asset maintains a historical repair duration = the MEDIAN
   ``resolution_hours`` across its completed repairs.
2. When a new ticket for that asset is completed, compare the ticket's
   resolution time against the asset's historical median (leave-one-out: the
   current ticket is excluded from its own baseline).
3. If the current repair is significantly faster than the asset's own history
   (``current < fast_ratio * asset_median``, default 25%), raise a
   **Verification Required** alert.
4. If the asset has insufficient own history (< ``min_asset_history`` prior
   repairs with a resolution time), fall back to the hierarchical baseline
   (asset_type + issue_type  ->  global issue_type). Never "last repair".
5. ``previous_repair_date`` / ``days_since_previous_repair`` / recent-repeat
   count are CONTEXT and a SEVERITY factor only — never the comparison.

The comparison is always against the asset's *historical repair behaviour*.
Output is a queue for a human to verify whether a request was necessary; it
never concludes fraud.

Only Verified/High ticket->asset mappings are used, so an alert always points
at the correct physical asset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_OUT = _ROOT / "outputs"


@dataclass
class VerifyConfig:
    fast_ratio: float = 0.25          # alert if current < ratio * baseline
    baseline_method: str = "median"   # metric used for the DECISION ("median" | "average")
    min_asset_history: int = 3        # prior repairs (with resolution time) for own baseline
    min_type_baseline_n: int = 5      # fallback tier 2: asset_type + issue_type
    minimum_valid_resolution_hours: float = 0.25  # 15 min; below = System Closure, excluded from baselines
    mature_min_count: int = 5         # >= this many verified repairs -> Mature baseline
    fake_note_min: int = 2            # >= this many fake repairs -> business note (audit only)
    recent_window_days: int = 30      # "repaired again recently" severity boost
    recent_lookback_days: int = 90    # window for recent-repeat context count
    # deterministic severity thresholds (faster_percentage = how far below baseline)
    crit_faster_pct: float = 90.0     # >= this -> Critical regardless of source
    high_faster_pct: float = 80.0     # >= this -> at least High
    recent_repeat_escalate: int = 2   # this many recent repeats escalates a step
    # Verified vs Raw history: baselines use VERIFIED history ONLY (manager marked
    # "Verified Genuine"). Repairs marked Verified Fake / Pending Review / Not
    # Enough Evidence never contribute. Raw history is retained for audit only.
    # allow_provisional_raw is an OPT-IN bootstrap: when True it surfaces
    # candidates from raw history (clearly labelled "Provisional") so a manager
    # can seed the first reviews before any Verified history exists. Default OFF
    # so fake/suspicious repairs can never contaminate a baseline.
    allow_provisional_raw: bool = False


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


def _hours_str(h: float) -> str:
    if pd.isna(h):
        return "—"
    if h < 1:
        return f"{int(round(h * 60))} minutes"
    return f"{h:.1f} hours"


# --------------------------------------------------------------------------- #
# assemble the per-ticket spine (reliable asset link + timestamps + context)
# --------------------------------------------------------------------------- #
def _spine(loader, base: Path) -> pd.DataFrame:
    """Per-ticket spine keyed on a resolved ``asset_id`` (the PRIMARY KEY).

    asset_id resolution precedence (real asset_id is always source of truth):
      1. ``ticket.asset_id`` present  -> use it, asset_id_source = "direct".
      2. else reliable Phase-1 mapping (Verified/High mapped_asset_id)
                                        -> asset_id_source = "mapped".
      3. else no asset_id              -> baseline falls back to asset_type/issue.
    """
    t = loader.maintenance_tickets().copy()
    mp = pd.read_csv(base / "ticket_asset_mapping.csv", dtype=str)

    # direct asset_id straight off the ticket = authoritative
    direct = dict(zip(t["id"].map(_s), t.get("asset_id").map(_s)))
    # mapping fallback (reliable inferred link) when no direct asset_id
    reliable = mp[mp["mapping_confidence"].isin(["Verified", "High"])]
    mapped = dict(zip(reliable["ticket_id"].astype(str),
                      reliable["mapped_asset_id"].astype(str)))
    map_atype = {}
    if "mapped_asset_type" in mp.columns:
        map_atype = {r["ticket_id"]: _s(r["mapped_asset_type"])
                     for _, r in mp.iterrows() if _s(r.get("mapped_asset_id"))}
    conf = dict(zip(mp["ticket_id"].astype(str), mp["mapping_confidence"].astype(str)))

    iname = {}
    it = loader.issue_types()
    if not it.empty:
        iname = dict(zip(it["id"].map(_s), it["name"].map(_s)))
    # asset_master keyed lookups (code + type) by the real asset_id
    acode, atype_by_id = {}, {}
    am = loader.asset_master()
    if not am.empty:
        if "asset_code" in am.columns:
            acode = {_s(k): _s(v) for k, v in zip(am["id"], am["asset_code"])}
        atn = loader.asset_types()
        tname = dict(zip(atn["id"].map(_s), atn["name"].map(_s))) if not atn.empty else {}
        if "asset_type_id" in am.columns:
            atype_by_id = {_s(k): tname.get(_s(v), "")
                           for k, v in zip(am["id"], am["asset_type_id"])}

    created = _dt(t.get("created_at"))
    resolved = _dt(t.get("resolved_at"))
    rh = (resolved - created).dt.total_seconds() / 3600.0

    df = pd.DataFrame({
        "ticket_id": t["id"].map(_s),
        "ticket_number": t.get("ticket_number").map(_s),
        "issue_type_id": t["issue_type_id"].map(_s),
        "tenant_approved": t["tenant_approved"].map(_s).str.lower(),
        "resolved_at": resolved,
        "resolution_hours": rh.where(rh >= 0),
    })

    def _resolve(tid: str):
        d = direct.get(tid, "")
        if d:
            return d, "direct"
        m = mapped.get(tid, "")
        if m:
            return m, "mapped"
        return "", ""

    res = df["ticket_id"].map(_resolve)
    df["asset_id"] = [a for a, _ in res]
    df["asset_id_source"] = [s for _, s in res]
    # asset_type from asset_master by the real asset_id, else mapping's type
    df["asset_type"] = [
        atype_by_id.get(a) or map_atype.get(tid, "")
        for a, tid in zip(df["asset_id"], df["ticket_id"])
    ]
    df["asset_code"] = df["asset_id"].map(lambda x: acode.get(x, x[:8] if x else ""))
    df["issue_type"] = df["issue_type_id"].map(lambda x: iname.get(x, x))
    df["mapping_confidence"] = df["ticket_id"].map(conf).fillna("")

    # cost + resolver from ticket_resolutions
    try:
        r = loader.ticket_resolutions()
        r["ticket_id"] = r["ticket_id"].map(_s)
        cost = r.groupby("ticket_id")["total_cost"].apply(lambda s: _num(s).sum())
        resolver = r.groupby("ticket_id")["resolved_by"].last()
        df["cost"] = df["ticket_id"].map(cost)
        df["resolver"] = df["ticket_id"].map(resolver).map(_s)
    except Exception:  # noqa: BLE001
        df["cost"] = np.nan
        df["resolver"] = ""
    return df


# --------------------------------------------------------------------------- #
# fallback hierarchical baseline (asset_type+issue -> global issue)
# --------------------------------------------------------------------------- #
def _fallback_lookups(df: pd.DataFrame, cfg: VerifyConfig):
    d = df[df["resolution_hours"].notna() & (df["issue_type"].fillna("") != "")]
    l3_med = d.groupby("issue_type")["resolution_hours"].median().to_dict()
    l3_n = d.groupby("issue_type")["resolution_hours"].size().to_dict()
    dt = d[d["asset_type"].fillna("") != ""]
    g = dt.groupby(["asset_type", "issue_type"])["resolution_hours"]
    l2 = {k: (v, int(g.size()[k])) for k, v in g.median().items()
          if g.size()[k] >= cfg.min_type_baseline_n}
    return l2, (l3_med, l3_n)


_CONFIDENCE = {"asset": "High", "asset_type": "Medium", "issue_type": "Low"}


def _severity_level(faster_pct: float, source: str, hist_n: int,
                    recent_ct: int, cfg: VerifyConfig) -> str:
    """Deterministic Critical/High/Medium from the three business factors:
    how much faster than baseline, how much history backs it, recent repeats.
    """
    own = source == "asset"
    # base tier on how far below baseline
    if faster_pct >= cfg.crit_faster_pct:
        level = "Critical"
    elif faster_pct >= cfg.high_faster_pct:
        level = "High"
    else:
        level = "Medium"
    # own-history evidence is stronger -> at least High
    if own and level == "Medium":
        level = "High"
    # a fast repair on an asset repaired repeatedly of late escalates one step
    if recent_ct >= cfg.recent_repeat_escalate and faster_pct >= cfg.high_faster_pct:
        level = "Critical" if level == "High" else level
    # thin fallback evidence (issue_type) never rates Critical on its own
    if source == "issue_type" and level == "Critical" and recent_ct < cfg.recent_repeat_escalate:
        level = "High"
    return level


def _timeline(history: pd.DataFrame, current_rh: float, cur_date,
              max_rows: int = 6) -> str:
    """Compact repair timeline for the drill-down: 'DD Mon YYYY -> Xh'."""
    rows = []
    h = history.sort_values("resolved_at").tail(max_rows)
    for _, e in h.iterrows():
        d = e["resolved_at"]
        when = d.strftime("%d %b %Y") if pd.notna(d) else "unknown"
        tag = "  [verified]" if bool(e.get("is_verified", False)) else ""
        rows.append(f"{when} -> {_hours_str(e['resolution_hours'])}{tag}")
    rows.append(f"Current -> {_hours_str(current_rh)}  [under review]")
    return "\n".join(rows)


def _verified_ticket_ids(base: Path) -> set:
    """ticket_ids the manager marked 'Verified Genuine' (from the review file).

    ONLY 'Verified Genuine' repairs enter any baseline. 'Verified Fake',
    'Pending Review', 'Not Enough Evidence' and unreviewed repairs are excluded
    by construction — the baseline always represents genuine repairs only. When a
    repair is changed to 'Verified Fake' it drops out of this set immediately, so
    a targeted rebuild removes it from that asset's median/average.
    """
    p = base / "maintenance_verification_reviews.csv"
    if not p.exists():
        return set()
    try:
        rv = pd.read_csv(p, dtype=str)
    except Exception:  # noqa: BLE001
        return set()
    if "review_status" not in rv.columns or "ticket_id" not in rv.columns:
        return set()
    ok = rv[rv["review_status"].astype(str).str.strip() == "Verified Genuine"]
    return set(ok["ticket_id"].astype(str))


_REVIEWED_STATUSES = {"Verified Genuine", "Verified Fake",
                      "Not Enough Evidence", "Asset Replaced"}


def _reviewed_ticket_ids(base: Path) -> set:
    """ticket_ids that have received a manager decision (any status except the
    default 'Pending Review'/blank). Used to drop reviewed repairs from the
    Historical Repair Audit queue."""
    p = base / "maintenance_verification_reviews.csv"
    if not p.exists():
        return set()
    try:
        rv = pd.read_csv(p, dtype=str)
    except Exception:  # noqa: BLE001
        return set()
    if "review_status" not in rv.columns or "ticket_id" not in rv.columns:
        return set()
    done = rv[rv["review_status"].astype(str).str.strip().isin(_REVIEWED_STATUSES)]
    return set(done["ticket_id"].astype(str))


def _export_histories(raw: pd.DataFrame, ver: pd.DataFrame, base: Path) -> None:
    """Persist the two per-asset histories (audit + baseline source of truth)."""
    cols = ["asset_id", "asset_code", "ticket_id", "ticket_number",
            "resolved_at", "resolution_hours"]
    raw_out = raw[[c for c in cols if c in raw.columns]].sort_values(
        ["asset_id", "resolved_at"])
    raw_out.to_csv(base / "maintenance_asset_raw_history.csv", index=False)
    ver_out = ver[[c for c in cols if c in ver.columns]].sort_values(
        ["asset_id", "resolved_at"])
    ver_out.to_csv(base / "maintenance_asset_verified_history.csv", index=False)


def _asset_baseline(g: Optional[pd.DataFrame], ticket_id: str, min_n: int):
    """Leave-one-out median of an asset's history; None if < min_n other repairs."""
    if g is None or len(g) <= 1:
        return None
    others = g[g["ticket_id"] != ticket_id]["resolution_hours"]
    if len(others) >= min_n:
        return float(others.median()), int(len(others))
    return None


def _asset_prior_baseline(g: Optional[pd.DataFrame], r: pd.Series, min_n: int,
                          method: str = "median"):
    """Baseline from an asset's repairs that completed BEFORE the current ticket.

    Temporal leave-one-out: only repairs with ``resolved_at`` strictly earlier
    than the current ticket count, so a ticket is never scored against its own
    or any later repair. ``g`` must already be the VERIFIED per-asset history.
    Uses ``method`` ("median" | "average") for the decision value. Returns
    (baseline_hours, prior_count) or None if < min_n prior repairs.
    """
    if g is None or g.empty or pd.isna(r["resolved_at"]):
        return None
    prior = g[(g["ticket_id"] != r["ticket_id"]) &
              (g["resolved_at"] < r["resolved_at"])]["resolution_hours"].dropna()
    if len(prior) >= min_n:
        val = prior.mean() if method == "average" else prior.median()
        return float(val), int(len(prior))
    return None


def _recency(g: Optional[pd.DataFrame], r: pd.Series, cfg: VerifyConfig):
    """Context from RAW history: previous repair date, days-since, recent count."""
    prev_date, days_since, recent_ct = pd.NaT, np.nan, 0
    if g is None or len(g) <= 1 or pd.isna(r["resolved_at"]):
        return prev_date, days_since, recent_ct
    prior = g[(g["ticket_id"] != r["ticket_id"]) & (g["resolved_at"] < r["resolved_at"])]
    if not prior.empty:
        prev_date = prior["resolved_at"].max()
        days_since = (r["resolved_at"] - prev_date).total_seconds() / 86400.0
    lo = r["resolved_at"] - pd.Timedelta(days=cfg.recent_lookback_days)
    recent_ct = int(((g["resolved_at"] >= lo) & (g["resolved_at"] < r["resolved_at"])).sum())
    return prev_date, days_since, recent_ct


# --------------------------------------------------------------------------- #
# core: build the verification queue
# --------------------------------------------------------------------------- #
def build_verification_queue(loader=None, cfg: Optional[VerifyConfig] = None,
                             out_dir: Optional[str] = None,
                             scope_ticket_ids: Optional[set] = None,
                             exclude_reviewed: bool = False,
                             out_name: str = "maintenance_verification_alerts.csv",
                             write_store: bool = True,
                             keep_all: bool = False,
                             status_label: str = "Verification Required",
                             recommendation: str = "Verification recommended.",
                             neutral_wording: bool = False
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Detect fast-vs-baseline repairs. Core rule unchanged. Reused by both
    business phases via the optional scope/exclude params:
      * scope_ticket_ids : only score these tickets (None = all).
      * exclude_reviewed : skip tickets already reviewed (Phase-1 audit queue).
      * out_name         : queue CSV filename (phase-specific).
      * write_store      : rebuild the per-asset baseline store (Phase-2 only).
    """
    cfg = cfg or VerifyConfig()
    base = Path(out_dir) if out_dir else _OUT
    if loader is None:
        from data_loader import DataLoader
        loader = DataLoader()

    full_df = _spine(loader, base)      # histories/baselines use ALL repairs
    if scope_ticket_ids is not None:    # scoring is restricted to the scope only
        score_df = full_df[full_df["ticket_id"].astype(str).isin(
            {str(t) for t in scope_ticket_ids})].copy()
    else:
        score_df = full_df
    skip_reviewed = _reviewed_ticket_ids(base) if exclude_reviewed else set()
    df = full_df                        # histories built from the full set below

    # ---- TWO HISTORIES per asset --------------------------------------------
    # Raw = every completed, timed repair (audit).  Verified = only repairs the
    # manager marked "Verified Genuine". ALL baselines use VERIFIED history;
    # "Needs Investigation"/"False Alert"/unreviewed repairs never contribute.
    raw_hist = df[(df["asset_id"] != "") & df["resolution_hours"].notna()].copy()
    verified_ids = _verified_ticket_ids(base)
    raw_hist["is_verified"] = raw_hist["ticket_id"].astype(str).isin(verified_ids)
    ver_hist = raw_hist[raw_hist["is_verified"]].copy()
    _export_histories(raw_hist, ver_hist, base)
    # baselines use only eligible verified repairs (System Closures + archived
    # Asset-Replaced history excluded); raw history untouched (audit).
    ver_hist = _baseline_pool(ver_hist, df, base, cfg)

    raw_by = {aid: g.sort_values("resolved_at") for aid, g in raw_hist.groupby("asset_id")}
    ver_by = {aid: g.sort_values("resolved_at") for aid, g in ver_hist.groupby("asset_id")}
    l2_v, (l3m_v, l3n_v) = _fallback_lookups(ver_hist, cfg)
    l2_r, (l3m_r, l3n_r) = _fallback_lookups(raw_hist, cfg)

    def _pick_baseline(l2, l3m, l3n, gasset, r):
        """Return (baseline, source, sample_n); asset tier = prior verified repairs."""
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

    alerts = []
    n_eval = 0
    for _, r in score_df.iterrows():
        aid, rh = r["asset_id"], r["resolution_hours"]
        if not aid or pd.isna(rh):
            continue                      # need a completed, timed repair on a known asset
        if str(r["ticket_id"]) in skip_reviewed:
            continue                      # already reviewed -> leaves the audit queue
        n_eval += 1
        graw, gver = raw_by.get(aid), ver_by.get(aid)
        prev_date, days_since, recent_ct = _recency(graw, r, cfg)
        raw_n = 0 if graw is None else int((graw["ticket_id"] != r["ticket_id"]).sum())
        ver_n = 0 if gver is None else int((gver["ticket_id"] != r["ticket_id"]).sum())

        # ---- baseline selection ----
        # VERIFIED history first (clean baseline). When an asset has no verified
        # genuine repairs yet (Historical Audit, early on), fall back to ALL
        # historical completed repairs (allow_provisional_raw). As the manager
        # confirms genuine repairs, that asset switches to its clean baseline and
        # the queue re-evaluates. Fake/Not-Enough/Pending never enter the verified
        # side. Same rule, threshold and prior-only median throughout.
        baseline = np.nan
        source = ""
        sample_n = 0
        history_type = ""
        picked = _pick_baseline(l2_v, l3m_v, l3n_v, gver, r)
        if picked:
            baseline, source, sample_n = picked
            history_type = "Verified"
        elif cfg.allow_provisional_raw:
            picked = _pick_baseline(l2_r, l3m_r, l3n_r, graw, r)
            if picked:
                baseline, source, sample_n = picked
                history_type = "Provisional"

        if source == "" or pd.isna(baseline) or baseline <= 0:
            continue
        pct = rh / baseline
        if pct >= cfg.fast_ratio and not keep_all:
            continue                      # not significantly faster than history
            # keep_all=True (frozen Historical-Audit worklist): a candidate stays
            # until reviewed, even if a cleaner baseline no longer flags it.

        faster_pct = round((1.0 - pct) * 100.0, 1)
        confidence = _CONFIDENCE[source]
        level = _severity_level(faster_pct, source, sample_n, recent_ct, cfg)
        # numeric severity: primarily faster%, nudged by recency/repeats — for ranking
        sev = faster_pct
        if pd.notna(days_since) and days_since <= cfg.recent_window_days:
            sev += 5.0
        if recent_ct >= cfg.recent_repeat_escalate:
            sev += 5.0
        sev = round(min(sev, 110.0), 1)

        timeline = _timeline(
            (graw[graw["ticket_id"] != r["ticket_id"]] if graw is not None
             else pd.DataFrame(columns=["resolved_at", "resolution_hours", "is_verified"])),
            rh, r["resolved_at"])

        # ---- business explanation (no ML wording) ----
        hist_word = ("verified " if history_type == "Verified" else "")
        if source == "asset":
            basis = (f"this asset's {hist_word}historical median ({baseline:.1f} hours, "
                     f"{sample_n} {hist_word}historical repairs)")
        else:
            label = (f"{r['asset_type']}/{r['issue_type']}" if source == "asset_type"
                     else r["issue_type"])
            basis = (f"the {label} {hist_word}baseline ({baseline:.1f} hours, n={sample_n}; "
                     f"this asset lacks enough own {hist_word}history)")
        repeat_txt = (f" Asset has been repaired {recent_ct} "
                      f"time{'s' if recent_ct != 1 else ''} in the last "
                      f"{cfg.recent_lookback_days} days." if recent_ct > 0 else "")
        prov_txt = ("" if history_type == "Verified" else
                    " (Provisional baseline from unverified history — will refine "
                    "as the manager verifies repairs.)")
        if neutral_wording:
            # Phase 1 — state the observed fact only; no action, no fake implication.
            explanation = (
                f"This historical repair was completed {faster_pct:.1f}% faster "
                f"than this asset's typical historical repair duration based on "
                f"available historical records."
            )
        else:
            explanation = (
                f"Resolved {faster_pct:.1f}% faster than {basis}.{repeat_txt} "
                f"{recommendation}{prov_txt}"
            )

        alerts.append({
            "asset_id": aid,
            "asset_id_source": r["asset_id_source"],
            "asset_code": r["asset_code"],
            "ticket_id": r["ticket_id"],
            "ticket_number": r["ticket_number"],
            "current_repair_date": r["resolved_at"],
            "current_resolution_hours": round(float(rh), 2),
            "asset_historical_median": round(baseline, 2),
            "asset_history_sample_size": sample_n,
            "baseline_source": source,
            "baseline_history_type": history_type,
            "baseline_confidence": confidence,
            "verified_history_size": ver_n,
            "raw_history_size": raw_n,
            "pct_of_baseline": round(float(pct), 3),
            "faster_percentage": faster_pct,
            "previous_repair_date": prev_date,
            "days_since_previous_repair": round(float(days_since), 1) if pd.notna(days_since) else np.nan,
            "recent_repeat_count_90d": recent_ct,
            "resolver": r["resolver"],
            "cost": r["cost"],
            "tenant_approved": r["tenant_approved"],
            "mapping_confidence": r["mapping_confidence"],
            "severity": sev,
            "severity_level": level,
            "verification_status": status_label,
            "repair_timeline": timeline,
            "explanation": explanation,
        })

    queue = pd.DataFrame(alerts)
    if not queue.empty:
        queue = queue.sort_values(["severity", "pct_of_baseline"],
                                  ascending=[False, True]).reset_index(drop=True)
    report = _validate(queue, n_eval, cfg)

    base.mkdir(parents=True, exist_ok=True)
    queue.to_csv(base / out_name, index=False)
    report.to_csv(base / "maintenance_verification_validation.csv", index=False)
    if write_store:                     # per-asset baseline store from ALL verified history
        build_asset_baselines(df=full_df, base=base, cfg=cfg)
    return queue, report


# --------------------------------------------------------------------------- #
# Two business phases (reuse build_verification_queue; no new scoring logic)
# --------------------------------------------------------------------------- #
_HISTORICAL_TICKETS = "maintenance_historical_tickets.csv"
_HISTORICAL_CANDIDATES = "maintenance_historical_audit_candidates.csv"
_HISTORICAL_QUEUE = "maintenance_historical_audit_queue.csv"
_FUTURE_QUEUE = "maintenance_future_verification_queue.csv"


def _historical_ticket_ids(base: Path, df: pd.DataFrame) -> set:
    """The one-time set of tickets that existed at go-live (Phase-1 scope).

    Snapshotted on first call and never changed, so any ticket appearing later
    is treated as a new/future repair (Phase 2).
    """
    p = base / _HISTORICAL_TICKETS
    if p.exists():
        try:
            return set(pd.read_csv(p, dtype=str)["ticket_id"].astype(str))
        except Exception:  # noqa: BLE001
            pass
    ids = set(df["ticket_id"].astype(str))
    base.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ticket_id": sorted(ids)}).to_csv(p, index=False)
    return ids


def build_historical_audit(loader=None, cfg: Optional[VerifyConfig] = None,
                           out_dir: Optional[str] = None
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """PHASE 1 — Historical Repair Audit that cleans itself, ticket by ticket.

    Ticket-level worklist:
      * On the FIRST run the initial suspicious tickets are captured as the
        permanent candidate set (using all historical completed repairs, since
        no verified history exists yet).
      * The queue thereafter = candidate set MINUS reviewed tickets. A ticket
        leaves ONLY when it is reviewed (exactly once); it is never dropped just
        because a cleaner baseline no longer flags it (keep_all=True).
      * Baselines are recomputed each run (VERIFIED-genuine-first, raw fallback
        while an asset lacks verified history), so the remaining tickets show the
        progressively cleaner baseline. An asset disappears only once ALL of its
        candidate tickets are reviewed.
    Same core rule + threshold; does NOT write the permanent baseline store.
    """
    from dataclasses import replace
    cfg = cfg or VerifyConfig()
    cfg = replace(cfg, allow_provisional_raw=True)
    base = Path(out_dir) if out_dir else _OUT
    if loader is None:
        from data_loader import DataLoader
        loader = DataLoader()
    df = _spine(loader, base)
    historical = _historical_ticket_ids(base, df)

    # Phase-1 wording: NEUTRAL findings only — no action-oriented wording, no
    # "Verification Required". Reserved for Phase 2. The owner decides what to do.
    hist_label = "Historical Repair Observation"

    cand_p = base / _HISTORICAL_CANDIDATES
    if cand_p.exists():
        candidates = set(pd.read_csv(cand_p, dtype=str)["ticket_id"].astype(str))
    else:
        # first run: the initially-flagged suspicious tickets are the worklist
        gen, _ = build_verification_queue(
            loader=loader, cfg=cfg, out_dir=str(base), scope_ticket_ids=historical,
            exclude_reviewed=False, out_name=_HISTORICAL_QUEUE, write_store=False,
            status_label=hist_label, neutral_wording=True)
        candidates = set(gen["ticket_id"].astype(str)) if not gen.empty else set()
        base.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ticket_id": sorted(candidates)}).to_csv(cand_p, index=False)

    return build_verification_queue(
        loader=loader, cfg=cfg, out_dir=str(base), scope_ticket_ids=candidates,
        exclude_reviewed=True, out_name=_HISTORICAL_QUEUE, write_store=False,
        keep_all=True, status_label=hist_label, neutral_wording=True)


def build_future_verification(loader=None, cfg: Optional[VerifyConfig] = None,
                              out_dir: Optional[str] = None
                              ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """PHASE 2 — production Future Verification.

    Scores only NEW tickets (not in the go-live historical set) against each
    asset's CLEAN verified-genuine baseline, and refreshes the baseline store.
    """
    cfg = cfg or VerifyConfig()
    base = Path(out_dir) if out_dir else _OUT
    if loader is None:
        from data_loader import DataLoader
        loader = DataLoader()
    df = _spine(loader, base)
    historical = _historical_ticket_ids(base, df)
    future = set(df["ticket_id"].astype(str)) - historical
    return build_verification_queue(
        loader=loader, cfg=cfg, out_dir=str(base), scope_ticket_ids=future,
        exclude_reviewed=False, out_name=_FUTURE_QUEUE, write_store=True)


# --------------------------------------------------------------------------- #
# baseline data hygiene: system-closure filter + Asset-Replaced archiving
# --------------------------------------------------------------------------- #
def _replaced_assets(df: Optional[pd.DataFrame], base: Path) -> Dict[str, "pd.Timestamp"]:
    """asset_id -> replacement date (latest 'Asset Replaced' review for it)."""
    if df is None:
        return {}
    p = base / "maintenance_verification_reviews.csv"
    if not p.exists():
        return {}
    try:
        rv = pd.read_csv(p, dtype=str)
    except Exception:  # noqa: BLE001
        return {}
    if "review_status" not in rv.columns or "ticket_id" not in rv.columns:
        return {}
    rep = set(rv.loc[rv["review_status"].astype(str).str.strip() == "Asset Replaced",
                     "ticket_id"].astype(str))
    if not rep:
        return {}
    d = df[df["ticket_id"].astype(str).isin(rep)]
    out: Dict[str, pd.Timestamp] = {}
    for _, row in d.iterrows():
        aid, rd = row["asset_id"], row["resolved_at"]
        if aid and pd.notna(rd):
            out[aid] = max(out.get(aid, rd), rd)
    return out


def _baseline_pool(ver: pd.DataFrame, df: Optional[pd.DataFrame], base: Path,
                   cfg: VerifyConfig) -> pd.DataFrame:
    """Verified repairs eligible to build a baseline.

    Excludes (a) System Closures — repairs shorter than
    ``minimum_valid_resolution_hours`` — and (b) archived repairs for assets
    marked 'Asset Replaced' (anything at/before the replacement date).
    """
    if ver.empty:
        return ver
    v = ver[ver["resolution_hours"] >= cfg.minimum_valid_resolution_hours].copy()
    replaced = _replaced_assets(df, base)
    if replaced and not v.empty:
        rep_dt = v["asset_id"].map(replaced)
        v = v[~(rep_dt.notna() & (v["resolved_at"] <= rep_dt))]
    return v


# --------------------------------------------------------------------------- #
# per-asset baseline store (verified history) + targeted rebuild
# --------------------------------------------------------------------------- #
_BASELINE_STORE = "maintenance_asset_baselines.csv"
# Both median AND average are always stored; baseline_method records which one
# the DECISION used, baseline_version increments each time this asset's baseline
# is rebuilt — so every decision is traceable to an exact baseline version.
_STORE_COLS = ["asset_id", "asset_code", "baseline_version", "baseline_method",
               "baseline_status", "verified_repair_count",
               "median_resolution_hours", "average_resolution_hours",
               "last_baseline_updated_at"]


def _verified_asset_frame(loader, base: Path, df: Optional[pd.DataFrame] = None,
                          cfg: Optional[VerifyConfig] = None) -> pd.DataFrame:
    """VERIFIED-GENUINE repairs eligible for a baseline (system closures excluded,
    Asset-Replaced history archived)."""
    cfg = cfg or VerifyConfig()
    if df is None:
        df = _spine(loader, base)
    hist = df[(df["asset_id"] != "") & df["resolution_hours"].notna()].copy()
    vids = _verified_ticket_ids(base)
    ver = hist[hist["ticket_id"].astype(str).isin(vids)]
    return _baseline_pool(ver, df, base, cfg)


def _baseline_row(aid: str, g: pd.DataFrame, now: str, method: str, version: int,
                  cfg: VerifyConfig, asset_code: str = "") -> Dict:
    rh = g["resolution_hours"].dropna() if not g.empty else pd.Series(dtype=float)
    n = int(len(rh))
    return {
        "asset_id": aid,
        "asset_code": (g["asset_code"].iloc[0] if not g.empty else asset_code) or aid[:8],
        "baseline_version": int(version),
        "baseline_method": method,
        "baseline_status": "Mature" if n >= cfg.mature_min_count else "Immature",
        "verified_repair_count": n,
        "median_resolution_hours": round(float(rh.median()), 2) if n else np.nan,
        "average_resolution_hours": round(float(rh.mean()), 2) if n else np.nan,
        "last_baseline_updated_at": now,
    }


def build_asset_baselines(loader=None, base: Optional[Path] = None,
                          cfg: Optional[VerifyConfig] = None,
                          df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Rebuild the FULL per-asset baseline store from Verified History.

    ``baseline_version`` increments over the previous stored version (new assets
    start at 1). Assets marked 'Asset Replaced' are RESET: old verified history is
    archived, and the row becomes count 0 / NULL medians / version 1 until new
    post-replacement repairs are verified.
    """
    cfg = cfg or VerifyConfig()
    base = Path(base) if base else _OUT
    if loader is None and df is None:
        from data_loader import DataLoader
        loader = DataLoader()
    if df is None:
        df = _spine(loader, base)
    ver = _verified_asset_frame(loader, base, df=df, cfg=cfg)
    _archive_replaced(df, base, cfg)
    update_fake_history(base, df=df, cfg=cfg)     # audit-only, no baseline impact
    replaced = _replaced_assets(df, base)

    prev = _load_store(base)
    prev_ver = dict(zip(prev.get("asset_id", pd.Series(dtype=str)).astype(str),
                        pd.to_numeric(prev.get("baseline_version"), errors="coerce").fillna(0).astype(int))) \
        if not prev.empty else {}
    now = pd.Timestamp.now().isoformat(timespec="seconds")
    acode = dict(zip(df["asset_id"], df["asset_code"]))

    rows = []
    seen = set()
    for aid, g in ver.groupby("asset_id"):
        version = 1 if aid in replaced else prev_ver.get(str(aid), 0) + 1
        rows.append(_baseline_row(aid, g, now, cfg.baseline_method, version, cfg))
        seen.add(aid)
    # replaced assets with no post-replacement verified repairs -> reset placeholder
    for aid in replaced:
        if aid not in seen:
            rows.append(_baseline_row(aid, pd.DataFrame(columns=ver.columns), now,
                                      cfg.baseline_method, 1, cfg,
                                      asset_code=acode.get(aid, "")))
    store = pd.DataFrame(rows, columns=_STORE_COLS)
    base.mkdir(parents=True, exist_ok=True)
    store.to_csv(base / _BASELINE_STORE, index=False)
    return store


def _archive_replaced(df: pd.DataFrame, base: Path, cfg: VerifyConfig) -> None:
    """Append pre-replacement verified repairs of replaced assets to an archive."""
    replaced = _replaced_assets(df, base)
    if not replaced:
        return
    hist = df[(df["asset_id"] != "") & df["resolution_hours"].notna()].copy()
    vids = _verified_ticket_ids(base)
    ver = hist[hist["ticket_id"].astype(str).isin(vids)]
    rep_dt = ver["asset_id"].map(replaced)
    arch = ver[rep_dt.notna() & (ver["resolved_at"] <= rep_dt)]
    if arch.empty:
        return
    cols = [c for c in ["asset_id", "asset_code", "ticket_id", "ticket_number",
                        "resolved_at", "resolution_hours"] if c in arch.columns]
    out = arch[cols].copy()
    out["archived_reason"] = "Asset Replaced"
    p = base / "maintenance_asset_verified_history_archive.csv"
    if p.exists():
        try:
            prev = pd.read_csv(p, dtype=str)
            out = pd.concat([prev, out.astype(str)], ignore_index=True).drop_duplicates("ticket_id")
        except Exception:  # noqa: BLE001
            pass
    out.to_csv(p, index=False)


def _load_store(base: Path) -> pd.DataFrame:
    p = base / _BASELINE_STORE
    if p.exists():
        try:
            return pd.read_csv(p, dtype={"asset_id": str})
        except Exception:  # noqa: BLE001
            pass
    return pd.DataFrame(columns=_STORE_COLS)


# --------------------------------------------------------------------------- #
# fake-repair audit history (append-only) + per-asset fake metrics
# --------------------------------------------------------------------------- #
_FAKE_HISTORY = "maintenance_asset_fake_history.csv"
_FAKE_METRICS = "maintenance_asset_fake_metrics.csv"
_FAKE_HIST_COLS = ["asset_id", "asset_code", "ticket_id", "ticket_number",
                   "resolved_at", "resolution_hours", "review_status",
                   "fake_reason", "reviewed_by", "reviewed_at", "marked_fake_at"]
_FAKE_NOTE = "Repeated suspicious repair history. Management review recommended."


def update_fake_history(base: Path, loader=None, df: Optional[pd.DataFrame] = None,
                        cfg: Optional[VerifyConfig] = None) -> pd.DataFrame:
    """Maintain the ever-fake audit history (never deleted) + per-asset metrics.

    A repair once marked 'Verified Fake' stays in this table forever. If the
    manager later changes the decision, the record is KEPT and its
    ``review_status`` is updated to the current status (the ``fake_reason`` and
    ``marked_fake_at`` from the fake episode are preserved). Audit-only — this
    never feeds a baseline. Metrics count repairs whose CURRENT status is fake.
    """
    cfg = cfg or VerifyConfig()
    base = Path(base)
    if df is None:
        if loader is None:
            from data_loader import DataLoader
            loader = DataLoader()
        df = _spine(loader, base)

    # current reviews (status, reason, reviewer, time) per ticket
    cur: Dict[str, dict] = {}
    rp = base / "maintenance_verification_reviews.csv"
    if rp.exists():
        try:
            rv = pd.read_csv(rp, dtype=str)
            for _, x in rv.iterrows():
                cur[str(x.get("ticket_id"))] = {
                    "review_status": _s(x.get("review_status")),
                    "fake_reason": _s(x.get("fake_reason")),
                    "reviewed_by": _s(x.get("reviewed_by")),
                    "reviewed_at": _s(x.get("reviewed_at")),
                }
        except Exception:  # noqa: BLE001
            pass

    # existing audit rows (ever-fake tickets) keyed by ticket_id
    hp = base / _FAKE_HISTORY
    prev: Dict[str, dict] = {}
    if hp.exists():
        try:
            for _, x in pd.read_csv(hp, dtype=str).iterrows():
                prev[str(x.get("ticket_id"))] = x.to_dict()
        except Exception:  # noqa: BLE001
            pass

    currently_fake = {t for t, c in cur.items() if c["review_status"] == "Verified Fake"}
    ever_fake = set(prev) | currently_fake
    dmeta = {str(r["ticket_id"]): r for _, r in df.iterrows()}

    rows = []
    for tid in ever_fake:
        c = cur.get(tid, {})
        p = prev.get(tid, {})
        d = dmeta.get(tid)
        status = c.get("review_status", "") or _s(p.get("review_status"))
        is_fake_now = status == "Verified Fake"
        # preserve the fake-episode reason; refresh only while currently fake
        reason = c.get("fake_reason", "") if is_fake_now else _s(p.get("fake_reason"))
        reason = reason or _s(p.get("fake_reason"))
        marked = _s(p.get("marked_fake_at")) or (c.get("reviewed_at", "") if is_fake_now else "")
        rows.append({
            "asset_id": (str(d["asset_id"]) if d is not None else _s(p.get("asset_id"))),
            "asset_code": (str(d["asset_code"]) if d is not None else _s(p.get("asset_code"))),
            "ticket_id": tid,
            "ticket_number": (str(d["ticket_number"]) if d is not None else _s(p.get("ticket_number"))),
            "resolved_at": (str(d["resolved_at"]) if d is not None else _s(p.get("resolved_at"))),
            "resolution_hours": (pd.to_numeric(pd.Series([d["resolution_hours"]]), errors="coerce").iloc[0]
                                 if d is not None else _s(p.get("resolution_hours"))),
            "review_status": status,
            "fake_reason": reason,
            "reviewed_by": c.get("reviewed_by", "") or _s(p.get("reviewed_by")),
            "reviewed_at": c.get("reviewed_at", "") or _s(p.get("reviewed_at")),
            "marked_fake_at": marked,
        })
    hist = pd.DataFrame(rows, columns=_FAKE_HIST_COLS)
    base.mkdir(parents=True, exist_ok=True)
    hist.to_csv(hp, index=False)

    # per-asset metrics: count repairs whose CURRENT status is Verified Fake
    total_by_asset = df[df["asset_id"] != ""].groupby("asset_id").size().to_dict()
    rows = []
    if not hist.empty:
        active = hist[hist["review_status"] == "Verified Fake"].copy()
        active["_rd"] = _dt(active.get("resolved_at"))
        for aid, g in active.groupby("asset_id"):
            n = int(len(g))
            total = int(total_by_asset.get(str(aid), 0)) or n
            rows.append({
                "asset_id": aid,
                "asset_code": g["asset_code"].iloc[0],
                "fake_repair_count": n,
                "last_fake_repair_date": (g["_rd"].max().isoformat()
                                          if g["_rd"].notna().any() else ""),
                "fake_repair_percentage": round(100.0 * n / total, 1),
                "business_note": _FAKE_NOTE if n >= cfg.fake_note_min else "",
            })
    metrics = pd.DataFrame(rows, columns=["asset_id", "asset_code", "fake_repair_count",
                                          "last_fake_repair_date", "fake_repair_percentage",
                                          "business_note"])
    metrics.to_csv(base / _FAKE_METRICS, index=False)
    return metrics


def update_asset_baseline(asset_id: str, loader=None, base: Optional[Path] = None,
                          cfg: Optional[VerifyConfig] = None) -> Optional[Dict]:
    """Rebuild ONLY this asset_id's baseline row (targeted, not the whole store).

    Called when a repair for this asset becomes 'Verified Genuine'. Recomputes
    that asset's verified count + median/average and stamps
    last_baseline_updated_at; every OTHER asset's row is left untouched.
    """
    cfg = cfg or VerifyConfig()
    base = Path(base) if base else _OUT
    if loader is None:
        from data_loader import DataLoader
        loader = DataLoader()
    df = _spine(loader, base)
    ver = _verified_asset_frame(loader, base, df=df, cfg=cfg)
    update_fake_history(base, df=df, cfg=cfg)     # audit-only, no baseline impact
    replaced = _replaced_assets(df, base)
    is_replaced = str(asset_id) in replaced
    if is_replaced:
        _archive_replaced(df, base, cfg)
    g = ver[ver["asset_id"].astype(str) == str(asset_id)]
    store = _load_store(base)
    old = store[store["asset_id"].astype(str) == str(asset_id)]
    prev_ver = int(pd.to_numeric(old["baseline_version"], errors="coerce").fillna(0).iloc[0]) \
        if not old.empty else 0
    version = 1 if is_replaced else prev_ver + 1     # Asset Replaced resets to 1
    store = store[store["asset_id"].astype(str) != str(asset_id)]     # drop old row
    acode = dict(zip(df["asset_id"], df["asset_code"]))
    row = None
    if not g.empty or is_replaced:
        row = _baseline_row(str(asset_id), g,
                            pd.Timestamp.now().isoformat(timespec="seconds"),
                            cfg.baseline_method, version, cfg,
                            asset_code=acode.get(str(asset_id), ""))
        store = pd.concat([store, pd.DataFrame([row])], ignore_index=True)
    store.to_csv(base / _BASELINE_STORE, index=False)
    return row


def score_future_ticket(asset_id: str, resolution_hours: float, issue_type: str,
                        asset_type: str = "", loader=None,
                        base: Optional[Path] = None,
                        cfg: Optional[VerifyConfig] = None) -> Dict:
    """Verify a NEW completed ticket using the CURRENT stored asset baseline.

    Future tickets read the per-asset store directly, so a baseline updated by
    ``update_asset_baseline`` takes effect immediately. Falls back to the
    verified asset_type / issue_type median when the asset has too little history.
    Decision uses the resolved asset_id; deterministic; never labels fraud.
    """
    cfg = cfg or VerifyConfig()
    base = Path(base) if base else _OUT
    store = _load_store(base)
    baseline, source, n = np.nan, "", 0
    hit = store[store["asset_id"].astype(str) == str(asset_id)]
    if not hit.empty and int(hit.iloc[0]["verified_repair_count"]) >= cfg.min_asset_history:
        baseline = float(hit.iloc[0]["median_resolution_hours"])
        source, n = "asset", int(hit.iloc[0]["verified_repair_count"])
    else:
        if loader is None:
            from data_loader import DataLoader
            loader = DataLoader()
        ver = _verified_asset_frame(loader, base)
        l2, (l3m, l3n) = _fallback_lookups(ver, cfg)
        h = l2.get((asset_type, issue_type))
        if h:
            baseline, source, n = float(h[0]), "asset_type", int(h[1])
        else:
            gm = l3m.get(issue_type)
            if gm and gm > 0:
                baseline, source, n = float(gm), "issue_type", int(l3n.get(issue_type, 0))

    decision, faster_pct, explanation = "Normal", np.nan, "No verified baseline available."
    if pd.notna(baseline) and baseline > 0 and pd.notna(resolution_hours):
        faster_pct = round((1.0 - resolution_hours / baseline) * 100.0, 1)
        if resolution_hours < cfg.fast_ratio * baseline:
            decision = "Verification Required"
            explanation = (f"This asset normally takes about {_hours_str(baseline)} to "
                           f"repair based on its verified repair records (n={n}). This "
                           f"ticket was completed in only {_hours_str(resolution_hours)} "
                           f"({faster_pct:.0f}% faster than normal). Verification is recommended.")
        else:
            explanation = (f"Completed in {_hours_str(resolution_hours)}, in line with this "
                           f"asset's normal ~{_hours_str(baseline)}. No verification needed.")
    return {"asset_id": asset_id, "verification_decision": decision,
            "baseline_source": source, "baseline_hours": round(baseline, 2) if pd.notna(baseline) else np.nan,
            "verified_repair_count": n, "faster_percentage": faster_pct,
            "explanation": explanation}


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def _validate(queue: pd.DataFrame, n_eval: int, cfg: VerifyConfig) -> pd.DataFrame:
    recs = [{"metric": "tickets_evaluated (asset-linked, timed)", "value": n_eval},
            {"metric": "verification_alerts", "value": int(len(queue))}]
    if not queue.empty:
        for src, c in queue["baseline_source"].value_counts().items():
            recs.append({"metric": f"alerts.baseline_source.{src}", "value": int(c)})
        for cf, c in queue["baseline_confidence"].value_counts().items():
            recs.append({"metric": f"alerts.baseline_confidence.{cf}", "value": int(c)})
        for ht, c in queue["baseline_history_type"].value_counts().items():
            recs.append({"metric": f"alerts.baseline_history.{ht}", "value": int(c)})
        for lvl in ("Critical", "High", "Medium"):
            recs.append({"metric": f"alerts.severity.{lvl}",
                         "value": int((queue["severity_level"] == lvl).sum())})
        pod = queue["pct_of_baseline"]
        recs += [{"metric": "pct_of_baseline.min", "value": round(float(pod.min()), 3)},
                 {"metric": "pct_of_baseline.median", "value": round(float(pod.median()), 3)}]
        own = queue[queue["baseline_source"] == "asset"]
        recs.append({"metric": "alerts_using_own_asset_baseline", "value": int(len(own))})
    return pd.DataFrame(recs)


def _print(queue: pd.DataFrame, report: pd.DataFrame) -> None:
    print("=" * 66)
    print("ASSET REPAIR VERIFICATION QUEUE (owner business rule)")
    print("=" * 66)
    for _, r in report.iterrows():
        print(f"  {r['metric']:<42} {r['value']}")
    if not queue.empty:
        print("-" * 66)
        for _, r in queue.head(5).iterrows():
            print(f"  [{r['severity']:>5}] {r['baseline_source']:<10} "
                  f"{r['asset_code']:<16} {r['explanation'][:78]}")
    print("=" * 66)


def main() -> None:
    queue, report = build_verification_queue()
    _print(queue, report)
    print(f"\nWrote maintenance_verification_alerts.csv + validation -> {_OUT}")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    main()
