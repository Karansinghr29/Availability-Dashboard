"""
asset_engine.py
===============

Asset Predictive Analytics — ticket-centric engine (built from scratch).

Pipeline
--------
Maintenance Tickets
  -> asset_id present? yes: direct asset
     no: Apartment/Room -> Asset Allocation -> Issue Type -> Asset Type -> allocated asset
  -> Assets table -> Purchase Date (else earliest Allocation Date) -> Asset Age
  -> Predictive analytics (health, risk, maintenance-due, replacement, trend,
     room risk, asset-type risk).

Pure compute. Reads only via ``DataLoader``. No DB writes. Independent module —
does not import or reuse any prior asset-analytics code.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_DM = 30.44
VERIFIED, HIGH, MEDIUM, LOW = "Verified", "High", "Medium", "Low"


def _s(v) -> str:
    t = "" if v is None else str(v).strip()
    return "" if t.lower() in ("", "null", "nan", "none", "<na>") else t


def _dt(series) -> pd.Series:
    d = pd.to_datetime(series, errors="coerce", utc=True)
    try:
        return d.dt.tz_localize(None)
    except (TypeError, AttributeError):
        return pd.to_datetime(series, errors="coerce")


# --------------------------------------------------------------------------- #
# reference lookups
# --------------------------------------------------------------------------- #
def _reference(loader):
    atypes, assets = loader.asset_types(), loader.asset_master()
    issue, items = loader.issue_types(), loader.maintenance_items()
    aptm, beds = loader.apartment_master(), loader.beds_master_uuid()

    tname = dict(zip(atypes["id"].map(_s), atypes["name"].map(_s))) if not atypes.empty else {}
    cyc = dict(zip(atypes["id"].map(_s), pd.to_numeric(atypes.get("maintenance_cycle_months"), errors="coerce"))) if not atypes.empty else {}
    life = dict(zip(atypes["id"].map(_s), pd.to_numeric(atypes.get("expected_life_months"), errors="coerce"))) if not atypes.empty else {}
    iname = dict(zip(issue["id"].map(_s), issue["name"].map(_s))) if not issue.empty else {}
    apt_code = dict(zip(aptm["id"].map(_s), aptm["apartment_code"].map(_s))) if not aptm.empty else {}
    bed_code = dict(zip(beds["id"].map(_s), beds["bed_code"].map(_s))) if (not beds.empty and "bed_code" in beds.columns) else {}

    issue_to_types: Dict[str, set] = {}
    if not items.empty:
        for _, r in items.iterrows():
            it, at = _s(r.get("issue_type_id")), tname.get(_s(r.get("asset_type_id")), "")
            if it and at:
                issue_to_types.setdefault(it, set()).add(at)

    meta = {}
    if not assets.empty:
        pur = _dt(assets.get("purchase_date"))
        war = _dt(assets.get("warranty_expiry"))
        for i, aid in enumerate(assets["id"].map(_s)):
            if not aid:
                continue
            tid = _s(assets.iloc[i].get("asset_type_id"))
            meta[aid] = {
                "asset_code": _s(assets.iloc[i].get("asset_code")),
                "brand": _s(assets.iloc[i].get("brand")), "model": _s(assets.iloc[i].get("model")),
                "asset_type": tname.get(tid, ""), "purchase_date": pur.iloc[i],
                "warranty_expiry": war.iloc[i],
                "expected_life_months": life.get(tid, np.nan),
                "maintenance_cycle_months": cyc.get(tid, np.nan),
            }
    return {"meta": meta, "issue_to_types": issue_to_types, "iname": iname,
            "apt_code": apt_code, "bed_code": bed_code}


def _allocations(loader):
    al = loader.asset_allocations().copy()
    if al.empty:
        return {}, {}, {}, 0
    al["_ad"] = _dt(al.get("allocated_date"))
    al["_apt"] = al["apartment_id"].map(_s)
    al["_bed"] = al["bed_id"].map(_s)
    al["_aid"] = al["asset_id"].map(_s)
    allocated = al[al["_aid"] != ""]["_aid"].nunique()
    latest = al.sort_values("_ad").drop_duplicates("_aid", keep="last")
    per_apt = latest[latest["_apt"] != ""].groupby("_apt")["_aid"].apply(list).to_dict()
    per_bed = latest[latest["_bed"] != ""].groupby("_bed")["_aid"].apply(list).to_dict()
    cur_room = {r["_aid"]: {"apt": r["_apt"], "bed": r["_bed"]} for _, r in latest.iterrows()}
    first_alloc = al[al["_ad"].notna()].groupby("_aid")["_ad"].min().to_dict()
    return {"per_apt": per_apt, "per_bed": per_bed}, cur_room, first_alloc, int(allocated)


# --------------------------------------------------------------------------- #
# step 1-4: ticket -> asset mapping
# --------------------------------------------------------------------------- #
def _map_tickets(loader, ref, rooms):
    mt = loader.maintenance_tickets().copy()
    if mt.empty:
        return mt
    meta, i2t = ref["meta"], ref["issue_to_types"]
    per_apt, per_bed = rooms["per_apt"], rooms["per_bed"]
    rows = []
    for _, t in mt.iterrows():
        aid, apt, bed = _s(t.get("asset_id")), _s(t.get("apartment_id")), _s(t.get("bed_id"))
        it = _s(t.get("issue_type_id"))
        if aid:
            asset, conf = aid, VERIFIED
        else:
            types = i2t.get(it, set())
            pool = per_bed.get(bed) or per_apt.get(apt) or []
            cand = [x for x in pool if meta.get(x, {}).get("asset_type", "") in types] if types else []
            if len(cand) == 1:
                asset, conf = cand[0], HIGH
            elif 2 <= len(cand) <= 3:
                asset, conf = cand[0], MEDIUM
            else:
                asset, conf = "", LOW
        rows.append({
            "asset_id": asset, "confidence": conf,
            "apartment_id": apt, "apartment_code": ref["apt_code"].get(apt, apt),
            "issue_type": ref["iname"].get(it, ""), "created_at": _dt(pd.Series([t.get("created_at")])).iloc[0],
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# step 5-7: age + fresh predictive scoring
# --------------------------------------------------------------------------- #
def _score(hist, meta, eff_date, as_of, avg_cost):
    age_days = (as_of - eff_date.normalize()).days if pd.notna(eff_date) else np.nan
    age_m = age_days / _DM if pd.notna(age_days) else np.nan
    life = pd.to_numeric(pd.Series([meta.get("expected_life_months")]), errors="coerce").iloc[0]
    cyc = pd.to_numeric(pd.Series([meta.get("maintenance_cycle_months")]), errors="coerce").iloc[0]
    age_ratio = age_m / life if (pd.notna(age_m) and pd.notna(life) and life > 0) else np.nan

    tc, recent, repeat = hist["ticket_count"], hist["recent_30d"], hist["repeat_count"]
    since = hist["days_since_last"]
    cyc_days = cyc * _DM if pd.notna(cyc) else np.nan
    overdue = bool(pd.notna(cyc_days) and pd.notna(since) and since > cyc_days)

    # --- fresh health score (0-100) ---
    penalty = 0.0
    penalty += min(35 * age_ratio, 35) if pd.notna(age_ratio) else 0
    penalty += min(8 * tc, 32)
    penalty += min(12 * recent, 24)
    penalty += 15 if overdue else 0
    penalty += min(6 * repeat, 18)
    health = round(max(0.0, min(100.0, 100 - penalty)), 1)

    # --- fresh risk thresholds ---
    if health < 35 or (repeat >= 3 and recent >= 2):
        risk = "Critical"
    elif health < 55:
        risk = "High"
    elif health < 75:
        risk = "Medium"
    else:
        risk = "Low"

    # --- failure probability (Poisson; exposure floored to damp re-ticket bursts) ---
    first = hist["first_ticket"]
    exposure = max((as_of - first.normalize()).days, 30) if pd.notna(first) else 30
    lam = min(tc / exposure, 0.2)
    prob30 = round(100 * (1 - np.exp(-lam * 30)), 1)

    # --- maintenance due / replacement ---
    maint_due = round(cyc_days - since, 0) if (pd.notna(cyc_days) and pd.notna(since)) else np.nan
    end_of_life = bool(pd.notna(age_ratio) and age_ratio >= 1.0)
    replace = bool(end_of_life or (tc >= 6 and repeat >= 2) or health < 25)

    # --- failure trend ---
    d = hist["dates"]
    if len(d) >= 3:
        mid = d.iloc[len(d) // 2]
        de = max((mid - d.min()).days, 1); dl = max((d.max() - mid).days, 1)
        re, rl = (d < mid).sum() / de, (d >= mid).sum() / dl
        ratio = rl / re if re > 0 else (2 if rl > 0 else 1)
        trend = ("Rapidly Degrading" if ratio > 2.5 else "Degrading" if ratio > 1.3
                 else "Improving" if ratio < 0.6 else "Stable")
    else:
        trend = "Stable"

    exp_cost_1y = round(lam * 365 * avg_cost, 0)

    if replace:
        rec = "Replace Asset"
    elif risk in ("Critical", "High") and (recent >= 1 or overdue):
        rec = "Repair Immediately"
    elif overdue or risk == "Medium" or trend in ("Degrading", "Rapidly Degrading"):
        rec = "Schedule Maintenance"
    elif risk != "Low":
        rec = "Monitor Closely"
    else:
        rec = "No Action Needed"

    bits = []
    if tc:
        bits.append(f"{tc} tickets" + (f", {recent} in 30d" if recent else ""))
    if overdue:
        bits.append("past maintenance cycle")
    if pd.notna(age_ratio):
        bits.append(f"age {int(100*age_ratio)}% of expected life")
    reason = f"{risk} risk — " + ("; ".join(bits) if bits else "limited history") + "."

    return {
        "age_days": age_days, "age_months": round(age_m, 1) if pd.notna(age_m) else np.nan,
        "age_ratio": round(age_ratio, 2) if pd.notna(age_ratio) else np.nan,
        "expected_life_months": life, "maintenance_cycle_months": cyc,
        "health_score": health, "risk_level": risk,
        "failure_prob_30d": prob30, "failure_trend": trend,
        "maintenance_due_days": maint_due, "maintenance_overdue": overdue,
        "replacement_recommended": replace, "expected_cost_1y": exp_cost_1y,
        "recommendation": rec, "reason": reason,
    }


def _alerts(r) -> str:
    a = []
    if r["maintenance_overdue"]:
        a.append("Maintenance Overdue")
    if r["risk_level"] in ("High", "Critical") and r["maintenance_overdue"]:
        a.append("Immediate Maintenance Required")
    if r["replacement_recommended"]:
        a.append("Replacement Recommended")
    if r["ticket_count"] >= 4:
        a.append("High Failure Frequency")
    if r["repeat_count"] >= 2:
        a.append("Repeated Same Issue")
    w = r.get("warranty_expiry")
    if pd.notna(w) and w < pd.Timestamp.today():
        a.append("Warranty Expired")
    return ", ".join(a)


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
def build(loader, as_of: Optional[pd.Timestamp] = None) -> Dict[str, object]:
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    ref = _reference(loader)
    rooms, cur_room, first_alloc, allocated = _allocations(loader)
    mapped = _map_tickets(loader, ref, rooms)
    if mapped.empty:
        return {"assets": pd.DataFrame(), "mapped": mapped, "kpis": {}, "as_of": as_of}

    # avg ticket cost (closure_cost sparse -> resolutions fallback -> neutral)
    avg_cost = 0.0
    for acc, col in [("maintenance_tickets", "closure_cost"), ("ticket_resolutions", "total_cost")]:
        df = getattr(loader, acc)()
        if not df.empty and col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(v):
                avg_cost = float(v.mean()); break
    avg_cost = avg_cost if avg_cost > 0 else 500.0

    scored = mapped[mapped["asset_id"].isin([""]) == False]  # noqa: E712
    scored = scored[scored["confidence"].isin([VERIFIED, HIGH])]
    rows = []
    for aid, g in scored.groupby("asset_id"):
        m = ref["meta"].get(aid, {})
        d = g["created_at"].dropna().sort_values()
        gaps = d.diff().dropna().dt.days
        iv = g["issue_type"].value_counts()
        last = d.max() if len(d) else pd.NaT
        first = d.min() if len(d) else pd.NaT
        hist = {
            "ticket_count": len(g), "recent_30d": int((d >= as_of - pd.Timedelta(days=30)).sum()) if len(d) else 0,
            "repeat_count": int((iv >= 2).sum()), "first_ticket": first, "last_ticket": last,
            "days_since_last": (as_of - last.normalize()).days if pd.notna(last) else np.nan,
            "avg_interval": round(gaps.mean(), 1) if len(gaps) else np.nan, "dates": d,
        }
        # age: purchase_date -> earliest allocation -> none
        pdate, fa = m.get("purchase_date"), first_alloc.get(aid, pd.NaT)
        if pd.notna(pdate):
            eff, src = pdate, "Purchase Date"
        elif pd.notna(fa):
            eff, src = fa, "Allocation Date"
        else:
            eff, src = pd.NaT, "Failure History"
        room = cur_room.get(aid, {})
        sc = _score(hist, m, eff, as_of, avg_cost)
        rec = {
            "asset_id": aid, "asset_code": m.get("asset_code", ""), "asset_type": m.get("asset_type", ""),
            "brand": m.get("brand", ""), "model": m.get("model", ""),
            "apartment_code": ref["apt_code"].get(room.get("apt", ""), ""),
            "bed_code": ref["bed_code"].get(room.get("bed", ""), ""),
            "purchase_date": m.get("purchase_date", pd.NaT), "warranty_expiry": m.get("warranty_expiry", pd.NaT),
            "age_source": src, "age_estimated_from": ("" if pd.isna(eff) else str(eff)[:10]),
            "ticket_count": hist["ticket_count"], "recent_30d": hist["recent_30d"],
            "repeat_count": hist["repeat_count"], "days_since_last": hist["days_since_last"],
            "first_ticket": first, "last_ticket": last, **sc,
        }
        rec["alerts"] = _alerts(rec)
        rows.append(rec)

    A = pd.DataFrame(rows)
    if not A.empty:
        A = A.sort_values("health_score").reset_index(drop=True)

    conf = mapped["confidence"].value_counts().to_dict()
    reliable = conf.get(VERIFIED, 0) + conf.get(HIGH, 0)
    total_tickets = int(len(mapped))
    kpis = {
        "total_assets": len(ref["meta"]), "allocated_assets": allocated,
        "total_tickets": total_tickets, "tickets_mapped": int(reliable),
        "tickets_mapped_incl_medium": int(reliable + conf.get(MEDIUM, 0)),
        "coverage_pct": round(100 * reliable / total_tickets, 1) if total_tickets else 0.0,
        "analyzed_assets": int(len(A)),
        "assets_with_purchase_date": int((A["age_source"] == "Purchase Date").sum()) if not A.empty else 0,
        "assets_allocation_estimate": int((A["age_source"] == "Allocation Date").sum()) if not A.empty else 0,
        "assets_failure_only": int((A["age_source"] == "Failure History").sum()) if not A.empty else 0,
        "healthy": int((A["risk_level"] == "Low").sum()) if not A.empty else 0,
        "medium_risk": int((A["risk_level"] == "Medium").sum()) if not A.empty else 0,
        "high_risk": int((A["risk_level"] == "High").sum()) if not A.empty else 0,
        "critical": int((A["risk_level"] == "Critical").sum()) if not A.empty else 0,
        "due_maintenance": int((A["maintenance_overdue"]).sum()) if not A.empty else 0,
        "due_replacement": int((A["replacement_recommended"]).sum()) if not A.empty else 0,
        "repeat_failures": int((A["repeat_count"] >= 1).sum()) if not A.empty else 0,
        "confidence": {k: int(conf.get(k, 0)) for k in (VERIFIED, HIGH, MEDIUM, LOW)},
    }
    return {"assets": A, "mapped": mapped, "kpis": kpis, "as_of": as_of}


def _rl(score):
    return ("Critical" if score >= 75 else "High" if score >= 50 else "Medium" if score >= 25 else "Low")


def room_risk(built) -> pd.DataFrame:
    A, mapped = built["assets"], built["mapped"]
    if A.empty:
        return pd.DataFrame()
    tks = mapped[mapped["asset_id"] != ""].groupby("apartment_code").size().rename("tickets")
    g = A.groupby("apartment_code").agg(assets=("asset_id", "count"), avg_health=("health_score", "mean"),
                                        high_risk_assets=("risk_level", lambda s: int(s.isin(["High", "Critical"]).sum()))).reset_index()
    g = g.merge(tks.reset_index(), on="apartment_code", how="left")
    g["tickets"] = g["tickets"].fillna(0).astype(int); g["avg_health"] = g["avg_health"].round(1)
    g["room_risk_score"] = ((100 - g["avg_health"]) * 0.5 + g["high_risk_assets"] * 12 + g["tickets"].clip(upper=20) * 1.5).round(1).clip(upper=100)
    g["room_risk_level"] = g["room_risk_score"].map(_rl)
    return g[g["apartment_code"] != ""].sort_values("room_risk_score", ascending=False)


def type_risk(built) -> pd.DataFrame:
    A = built["assets"]
    if A.empty:
        return pd.DataFrame()
    g = A.groupby("asset_type").agg(assets=("asset_id", "count"), tickets=("ticket_count", "sum"),
                                    avg_health=("health_score", "mean"), avg_prob30=("failure_prob_30d", "mean"),
                                    high_risk_assets=("risk_level", lambda s: int(s.isin(["High", "Critical"]).sum()))).reset_index()
    g["avg_health"] = g["avg_health"].round(1); g["avg_prob30"] = g["avg_prob30"].round(1)
    g["type_risk_score"] = ((100 - g["avg_health"]) * 0.5 + g["avg_prob30"] * 0.5).round(1).clip(upper=100)
    g["type_risk_level"] = g["type_risk_score"].map(_rl)
    return g[g["asset_type"] != ""].sort_values("type_risk_score", ascending=False)


def charts(built) -> Dict[str, pd.DataFrame]:
    A, mapped = built["assets"], built["mapped"]
    out = {}
    if A.empty:
        return out
    hb = pd.cut(A["health_score"], [-1, 35, 55, 75, 100], labels=["0-35", "35-55", "55-75", "75-100"])
    out["health_distribution"] = hb.value_counts().reindex(["0-35", "35-55", "55-75", "75-100"]).rename_axis("health").reset_index(name="assets")
    out["failure_by_type"] = A.groupby("asset_type")["ticket_count"].sum().sort_values(ascending=False).head(20).rename("tickets").reset_index()
    att = mapped[mapped["asset_id"] != ""]
    mm = att.assign(month=att["created_at"].dt.to_period("M").astype(str))
    out["monthly_trend"] = mm[mm["month"] != "NaT"].groupby("month").size().rename("tickets").reset_index()
    dm = pd.to_numeric(A["maintenance_due_days"], errors="coerce")
    tb = pd.cut(dm, [-10**9, 0, 7, 30, 90, 10**9], labels=["overdue", "<=7d", "8-30d", "31-90d", ">90d"])
    out["maintenance_timeline"] = tb.value_counts().reindex(["overdue", "<=7d", "8-30d", "31-90d", ">90d"]).rename_axis("window").reset_index(name="assets")
    out["replacement_by_type"] = A[A["replacement_recommended"]].groupby("asset_type").size().sort_values(ascending=False).rename("assets").reset_index()
    out["top_failing_assets"] = A.sort_values("ticket_count", ascending=False).head(15)[["asset_code", "asset_type", "apartment_code", "ticket_count", "risk_level"]]
    return out


def exports(built) -> Dict[str, pd.DataFrame]:
    A = built["assets"]
    if A.empty:
        return {}
    base = ["asset_code", "asset_type", "brand", "model", "apartment_code", "bed_code"]
    return {
        "asset_health_report": A.drop(columns=["asset_id", "first_ticket", "last_ticket"], errors="ignore"),
        "maintenance_schedule": A[A["maintenance_due_days"].notna()].sort_values("maintenance_due_days")[
            base + ["maintenance_due_days", "maintenance_overdue", "maintenance_cycle_months", "recommendation"]],
        "replacement_plan": A[A["replacement_recommended"]][
            base + ["age_months", "age_ratio", "ticket_count", "health_score", "reason"]],
    }


# ==========================================================================  #
# EXTENSION — full-history business coverage (additive; code above unchanged).  #
# Room + asset-type intelligence use ALL tickets (every ticket carries          #
# apartment_id + issue_type); asset-level scoring stays reliable.               #
# ==========================================================================  #
def _avg_ticket_cost(loader) -> float:
    for acc, col in [("maintenance_tickets", "closure_cost"), ("ticket_resolutions", "total_cost")]:
        df = getattr(loader, acc)()
        if not df.empty and col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(v):
                return float(v.mean())
    return 500.0


def coverage(loader, built) -> Dict[str, object]:
    """Ticket participation across the whole maintenance history."""
    mt = loader.maintenance_tickets()
    ref = _reference(loader)
    total = int(len(mt))
    apt = mt["apartment_id"].map(_s) if "apartment_id" in mt.columns else pd.Series([], dtype=str)
    it = mt["issue_type_id"].map(_s) if "issue_type_id" in mt.columns else pd.Series([], dtype=str)
    mappable = it.map(lambda x: x in ref["issue_to_types"])
    processed = int(((apt != "") | mappable).sum())
    k = built["kpis"]
    conf = k["confidence"]
    return {
        "total_tickets": total,
        "tickets_processed": processed,
        "tickets_waiting_mapping": total - processed,
        "asset_pinned_reliable": conf["Verified"] + conf["High"],
        "asset_pinned_incl_medium": conf["Verified"] + conf["High"] + conf["Medium"],
        "total_assets": k["total_assets"],
        "allocated_assets": k["allocated_assets"],
        "assets_with_purchase_date": k["assets_with_purchase_date"],
        "assets_using_allocation_date": k["assets_allocation_estimate"],
        "coverage_pct": round(100 * processed / total, 1) if total else 0.0,
        "asset_coverage_pct": round(100 * (conf["Verified"] + conf["High"]) / total, 1) if total else 0.0,
    }


def room_intelligence(loader, built, as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Per-apartment intelligence from ALL tickets (not just asset-pinned)."""
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    mapped = built["mapped"]
    if mapped.empty:
        return pd.DataFrame()
    avg_cost = _avg_ticket_cost(loader)
    m = mapped[mapped["apartment_code"].astype(str) != ""].copy()
    rows = []
    for apt, g in m.groupby("apartment_code"):
        d = g["created_at"].dropna().sort_values()
        recent30 = int((d >= as_of - pd.Timedelta(days=30)).sum())
        recent90 = int((d >= as_of - pd.Timedelta(days=90)).sum())
        trend = "Stable"
        if len(d) >= 4:
            mid = d.iloc[len(d) // 2]
            de = max((mid - d.min()).days, 1)
            dl = max((d.max() - mid).days, 1)
            re, rl = (d < mid).sum() / de, (d >= mid).sum() / dl
            ratio = rl / re if re > 0 else 2
            trend = "Increasing" if ratio > 1.3 else ("Decreasing" if ratio < 0.7 else "Stable")
        iv = g["issue_type"].value_counts()
        repeated = int((iv >= 2).sum())
        distinct_types = int(g["issue_type"].replace("", np.nan).dropna().nunique())
        assets_failing = int(g.loc[g["asset_id"] != "", "asset_id"].nunique())
        total = len(g)
        est_cost = round(total * avg_cost, 0)
        score = round(min(100, total * 2.0 + repeated * 6 + recent30 * 4
                          + (15 if trend == "Increasing" else 0) + assets_failing * 2), 1)
        rows.append({
            "apartment_code": apt, "total_tickets": total, "recent_30d": recent30, "recent_90d": recent90,
            "demand_trend": trend, "repeated_failures": repeated, "distinct_issue_types": distinct_types,
            "assets_failing": assets_failing, "est_maintenance_cost": est_cost,
            "room_risk_score": score, "room_risk_level": _rl(score),
            "preventive_inspection": bool(trend == "Increasing" and total >= 10),
            "multiple_asset_failures": bool(assets_failing >= 2 or distinct_types >= 3),
        })
    return pd.DataFrame(rows).sort_values("room_risk_score", ascending=False).reset_index(drop=True)


def asset_type_intelligence(loader, built, as_of: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """Per-asset-type intelligence: failure frequency from ALL tickets (issue->type),
    age/interval from scored assets of that type."""
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    ref = _reference(loader)
    i2t = ref["issue_to_types"]
    mt = loader.maintenance_tickets()
    freq: Dict[str, int] = {}
    if not mt.empty:
        for it in mt["issue_type_id"].map(_s):
            for at in i2t.get(it, set()):
                freq[at] = freq.get(at, 0) + 1
    _, _, first_alloc, _ = _allocations(loader)
    meta = ref["meta"]
    age_by_type: Dict[str, list] = {}
    life_by_type: Dict[str, float] = {}
    count_by_type: Dict[str, int] = {}
    for aid, mm in meta.items():
        at = mm.get("asset_type", "")
        if not at:
            continue
        count_by_type[at] = count_by_type.get(at, 0) + 1
        pdate = mm.get("purchase_date")
        fa = first_alloc.get(aid, pd.NaT)
        eff = pdate if pd.notna(pdate) else fa
        if pd.notna(eff):
            age_by_type.setdefault(at, []).append((as_of - eff.normalize()).days / _DM)
        life_by_type[at] = mm.get("expected_life_months", np.nan)
    A = built["assets"]
    tk = A.groupby("asset_type").agg(avg_tickets=("ticket_count", "mean"),
                                     avg_health=("health_score", "mean")) if not A.empty else pd.DataFrame()
    # avg maintenance interval per type from asset-pinned ticket gaps
    interval_by_type: Dict[str, float] = {}
    if not A.empty:
        type_by_asset = dict(zip(A["asset_id"], A["asset_type"]))
        att = built["mapped"]
        att = att[att["asset_id"] != ""]
        gaps_by_type: Dict[str, list] = {}
        for aid, g in att.groupby("asset_id"):
            dd = g["created_at"].dropna().sort_values()
            if len(dd) >= 2:
                at2 = type_by_asset.get(aid)
                if at2:
                    gaps_by_type.setdefault(at2, []).extend(dd.diff().dropna().dt.days.tolist())
        interval_by_type = {t: round(float(np.mean(v)), 1) for t, v in gaps_by_type.items() if v}
    rows = []
    for at in set(list(freq.keys()) + list(count_by_type.keys())):
        ages = age_by_type.get(at, [])
        avg_age = round(float(np.mean(ages)), 1) if ages else np.nan
        life = pd.to_numeric(pd.Series([life_by_type.get(at)]), errors="coerce").iloc[0]
        age_ratio = round(avg_age / life, 2) if (pd.notna(avg_age) and pd.notna(life) and life > 0) else np.nan
        row = tk.loc[at] if (not tk.empty and at in tk.index) else None
        avg_health = round(float(row["avg_health"]), 1) if row is not None else np.nan
        rows.append({
            "asset_type": at, "assets": count_by_type.get(at, 0), "failure_frequency": freq.get(at, 0),
            "avg_age_months": avg_age,
            "avg_tickets_per_asset": round(float(row["avg_tickets"]), 2) if row is not None else 0.0,
            "avg_maintenance_interval_days": interval_by_type.get(at, np.nan),
            "avg_health": avg_health, "age_ratio": age_ratio,
            "approaching_end_of_life": bool(pd.notna(age_ratio) and age_ratio >= 0.8),
            "high_risk": bool((pd.notna(avg_health) and avg_health < 55) or freq.get(at, 0) >= 100),
        })
    return pd.DataFrame(rows).sort_values("failure_frequency", ascending=False).reset_index(drop=True)


def executive(loader, built, room_intel=None, type_intel=None, as_of: Optional[pd.Timestamp] = None) -> Dict[str, object]:
    """Executive business dashboard."""
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    A = built["assets"]
    ri = room_intel if room_intel is not None else room_intelligence(loader, built, as_of)
    ti = type_intel if type_intel is not None else asset_type_intelligence(loader, built, as_of)

    likely = A[A["failure_prob_30d"] >= 60].sort_values("failure_prob_30d", ascending=False) if not A.empty else A
    preventive = A[(A["maintenance_overdue"]) | (pd.to_numeric(A["maintenance_due_days"], errors="coerce") <= 30)] if not A.empty else A
    replace = A[A["replacement_recommended"]] if not A.empty else A

    pv = {}
    if not A.empty:
        for src in ("Purchase Date", "Allocation Date", "Failure History"):
            s = A[A["age_source"] == src]
            pv[src] = {"assets": int(len(s)),
                       "avg_age_months": round(float(pd.to_numeric(s["age_months"], errors="coerce").mean()), 1) if (len(s) and s["age_months"].notna().any()) else None}

    d = built["mapped"]["created_at"].dropna()
    last30 = int((d >= as_of - pd.Timedelta(days=30)).sum())
    prev30 = int(((d >= as_of - pd.Timedelta(days=60)) & (d < as_of - pd.Timedelta(days=30))).sum())
    forecast = last30 if prev30 == 0 else int(round(last30 * (last30 / prev30) ** 0.5))

    insights = []
    if not ri.empty:
        insights.append(f"Top risky apartment: {ri.iloc[0]['apartment_code']} "
                        f"({ri.iloc[0]['total_tickets']} tickets, {ri.iloc[0]['room_risk_level']} risk).")
        inc = ri[ri["demand_trend"] == "Increasing"]
        if len(inc):
            insights.append(f"{len(inc)} apartment(s) show increasing maintenance demand.")
    if not ti.empty:
        insights.append(f"Top failing asset category: {ti.iloc[0]['asset_type']} "
                        f"({int(ti.iloc[0]['failure_frequency'])} issue-tickets).")
        eol = ti[ti["approaching_end_of_life"]]
        if len(eol):
            insights.append(f"{len(eol)} asset type(s) approaching end of life: {', '.join(eol['asset_type'].head(3))}.")
    insights.append(f"{len(likely)} asset(s) likely to fail within 30 days (>=60% probability).")
    insights.append(f"Maintenance workload forecast next 30 days: ~{forecast} tickets (last 30d = {last30}).")
    if pv.get("Allocation Date", {}).get("assets"):
        insights.append(f"{pv['Allocation Date']['assets']} assets aged from allocation date "
                        f"(no purchase date) - true age may be understated.")

    return {
        "top_risky_apartments": ri.head(10) if not ri.empty else pd.DataFrame(),
        "top_risky_asset_types": ti.head(10) if not ti.empty else pd.DataFrame(),
        "likely_fail_30d": likely[["asset_code", "asset_type", "apartment_code", "failure_prob_30d", "risk_level", "recommendation"]] if not A.empty else pd.DataFrame(),
        "preventive": preventive[["asset_code", "asset_type", "apartment_code", "maintenance_due_days", "recommendation"]] if not A.empty else pd.DataFrame(),
        "replacement": replace[["asset_code", "asset_type", "apartment_code", "age_months", "age_ratio", "reason"]] if not A.empty else pd.DataFrame(),
        "purchase_vs_allocation": pv,
        "workload_forecast_30d": int(forecast), "workload_last_30d": int(last30),
        "insights": insights,
    }


# ==========================================================================  #
# EXTENSION 2 — complete business maintenance-intelligence views (additive).   #
# Reuses build()/mapped/assets; ticket-centric mapping unchanged. No ML.        #
# ==========================================================================  #
def _reliability_level(s):
    return ("High" if s >= 75 else "Medium" if s >= 55 else "Low" if s >= 35 else "Poor")


def _lifecycle_stage(age_ratio, replace):
    if replace or (pd.notna(age_ratio) and age_ratio >= 1.0):
        return "Replace"
    if pd.isna(age_ratio):
        return "Unknown"
    if age_ratio < 0.25:
        return "New"
    if age_ratio < 0.60:
        return "Mid Life"
    if age_ratio < 0.85:
        return "Aging"
    return "Near End of Life"


def business_views(loader, built, as_of=None) -> Dict[str, object]:
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    A = built["assets"].copy()
    mapped = built["mapped"]
    if A.empty:
        return {}

    # per-asset avg failure interval (from asset-pinned ticket gaps)
    interval = {}
    att = mapped[mapped["asset_id"] != ""]
    for aid, g in att.groupby("asset_id"):
        d = g["created_at"].dropna().sort_values()
        if len(d) >= 2:
            interval[aid] = round(float(d.diff().dropna().dt.days.mean()), 1)
    A["avg_interval_days"] = A["asset_id"].map(interval)

    # ---- 1. Asset Reliability Score (0-100) ----
    def rel(r):
        p = min(r["ticket_count"] * 7, 35) + min(r["repeat_count"] * 10, 25) + min(r["recent_30d"] * 8, 16)
        ar = r["age_ratio"]
        p += min(ar * 20, 20) if pd.notna(ar) else 0
        p += 12 if r["maintenance_overdue"] else 0
        return round(max(0.0, 100 - p), 1)
    A["reliability_score"] = A.apply(rel, axis=1)
    A["reliability_level"] = A["reliability_score"].map(_reliability_level)

    # ---- 10. lifecycle stage ----
    A["lifecycle_stage"] = A.apply(lambda r: _lifecycle_stage(r["age_ratio"], r["replacement_recommended"]), axis=1)

    # ---- 2. Apartment Health Score ----
    all_by_apt = mapped[mapped["apartment_code"].astype(str) != ""].groupby("apartment_code")
    apt_rows = []
    for apt, ag in A.groupby("apartment_code"):
        if not str(apt).strip():
            continue
        tickets = int(all_by_apt.size().get(apt, 0)) if apt in all_by_apt.size().index else int(ag["ticket_count"].sum())
        repeat = int(ag["repeat_count"].sum())
        high = int(ag["risk_level"].isin(["High", "Critical"]).sum())
        overdue = int(ag["maintenance_overdue"].sum())
        avg_age = round(float(pd.to_numeric(ag["age_months"], errors="coerce").mean()), 1) if ag["age_months"].notna().any() else np.nan
        pen = min(tickets * 1.5, 30) + min(repeat * 6, 24) + min(high * 8, 24) + min(overdue * 6, 12)
        pen += min((avg_age / 60) * 10, 10) if pd.notna(avg_age) else 0
        hs = round(max(0.0, 100 - pen), 1)
        apt_rows.append({"apartment_code": apt, "total_tickets": tickets, "repeat_failures": repeat,
                         "high_risk_assets": high, "overdue_assets": overdue, "avg_asset_age_months": avg_age,
                         "health_score": hs, "health_level": _reliability_level(hs)})
    apartment_health = pd.DataFrame(apt_rows).sort_values("health_score").reset_index(drop=True)

    # ---- 3. Asset-Type Health ----
    ref = _reference(loader)
    i2t = ref["issue_to_types"]
    mt = loader.maintenance_tickets()
    freq = {}
    for it in mt["issue_type_id"].map(_s):
        for at in i2t.get(it, set()):
            freq[at] = freq.get(at, 0) + 1
    catalog = {}
    for mm in ref["meta"].values():
        at = mm.get("asset_type", "")
        if at:
            catalog[at] = catalog.get(at, 0) + 1
    th_rows = []
    for at in set(list(catalog) + list(freq)):
        sub = A[A["asset_type"] == at]
        total_assets = catalog.get(at, 0)
        failures = freq.get(at, 0)
        th_rows.append({
            "asset_type": at, "total_assets": total_assets, "total_failures": failures,
            "failure_rate_per_asset": round(failures / total_assets, 2) if total_assets else np.nan,
            "avg_age_months": round(float(pd.to_numeric(sub["age_months"], errors="coerce").mean()), 1) if (len(sub) and sub["age_months"].notna().any()) else np.nan,
            "high_risk_count": int(sub["risk_level"].isin(["High", "Critical"]).sum()),
            "maintenance_due_count": int(sub["maintenance_overdue"].sum()),
            "replacement_count": int(sub["replacement_recommended"].sum()),
        })
    asset_type_health = pd.DataFrame(th_rows).sort_values("total_failures", ascending=False).reset_index(drop=True)

    # ---- 4. Maintenance Calendar ----
    due = pd.to_numeric(A["maintenance_due_days"], errors="coerce")
    cols_cal = ["asset_code", "asset_type", "apartment_code", "maintenance_due_days", "risk_level", "recommendation"]
    calendar = {
        "today": A[due <= 0][cols_cal],
        "this_week": A[(due > 0) & (due <= 7)][cols_cal],
        "this_month": A[(due > 7) & (due <= 30)][cols_cal],
        "next_month": A[(due > 30) & (due <= 60)][cols_cal],
    }

    # ---- 6. Failure Hotspots ----
    apt_density = mapped[mapped["apartment_code"] != ""].groupby("apartment_code").size().sort_values(ascending=False).head(20).rename("tickets").reset_index()
    aptm = loader.apartment_master()
    floor_map = dict(zip(aptm["apartment_code"].map(_s), pd.to_numeric(aptm.get("floor_number"), errors="coerce"))) if not aptm.empty else {}
    fd = mapped[mapped["apartment_code"] != ""].assign(floor=mapped["apartment_code"].map(lambda x: floor_map.get(_s(x))))
    floor_density = fd.dropna(subset=["floor"]).groupby("floor").size().rename("tickets").reset_index() if not fd.empty else pd.DataFrame()
    type_failures = asset_type_health[["asset_type", "total_failures"]].head(20)
    mm2 = mapped.assign(month=mapped["created_at"].dt.to_period("M").astype(str))
    monthly = mm2[mm2["month"] != "NaT"].groupby("month").size().rename("tickets").reset_index()
    hotspots = {"apartment_density": apt_density, "floor_density": floor_density,
                "type_failures": type_failures, "monthly_trend": monthly}

    # ---- 7. Asset Rankings ----
    base = ["asset_code", "asset_type", "apartment_code"]
    rankings = {
        "most_reliable": A.sort_values("reliability_score", ascending=False).head(10)[base + ["reliability_score", "ticket_count"]],
        "least_reliable": A.sort_values("reliability_score").head(10)[base + ["reliability_score", "ticket_count", "repeat_count"]],
        "most_repaired": A.sort_values("ticket_count", ascending=False).head(10)[base + ["ticket_count", "repeat_count", "avg_interval_days"]],
        "near_end_of_life": A[A["age_ratio"].notna()].sort_values("age_ratio", ascending=False).head(10)[base + ["age_months", "age_ratio", "lifecycle_stage"]],
        "highest_risk": A.sort_values(["risk_level", "health_score"], key=lambda s: s.map({"Critical": 0, "High": 1, "Medium": 2, "Low": 3}) if s.name == "risk_level" else s).head(10)[base + ["risk_level", "health_score", "failure_prob_30d"]],
    }

    # ---- 8. Repeat Failure Analysis ----
    rep = A[(A["repeat_count"] >= 1) | (A["ticket_count"] >= 2)].copy()
    rep["recommendation"] = rep.apply(
        lambda r: ("Replace — chronic repeat failures" if r["repeat_count"] >= 2 and (pd.notna(r["avg_interval_days"]) and r["avg_interval_days"] <= 30)
                   else "Inspect root cause — recurring issue" if r["repeat_count"] >= 1
                   else "Monitor — multiple tickets"), axis=1)
    repeat_analysis = rep.sort_values(["repeat_count", "ticket_count"], ascending=False)[
        base + ["ticket_count", "repeat_count", "avg_interval_days", "risk_level", "recommendation"]]

    # ---- 9. Preventive Maintenance Queue (prioritized + why) ----
    def why(r):
        bits = []
        if r["maintenance_overdue"]:
            bits.append("maintenance overdue")
        if r["repeat_count"] >= 2:
            bits.append(f"{r['repeat_count']} repeat failures")
        if r["recent_30d"] >= 1:
            bits.append(f"{r['recent_30d']} recent failure(s)")
        if pd.notna(r["age_ratio"]) and r["age_ratio"] >= 0.85:
            bits.append(f"age {int(100*r['age_ratio'])}% of life")
        if r["failure_prob_30d"] >= 60:
            bits.append(f"{r['failure_prob_30d']}% 30d failure prob")
        return "; ".join(bits) if bits else "elevated risk"
    q = A[A["risk_level"] != "Low"].copy()
    q["priority"] = q["risk_level"]
    q["why"] = q.apply(why, axis=1)
    preventive_queue = q.sort_values("health_score")[
        base + ["priority", "health_score", "failure_prob_30d", "maintenance_due_days", "recommendation", "why"]]

    # ---- 10. lifecycle summary ----
    lifecycle_counts = A["lifecycle_stage"].value_counts().reindex(
        ["New", "Mid Life", "Aging", "Near End of Life", "Replace", "Unknown"]).fillna(0).astype(int).rename_axis("stage").reset_index(name="assets")

    # ---- 5. Executive KPIs ----
    cov = coverage(loader, built)
    exec_kpis = {
        **{kk: cov[kk] for kk in ["total_assets", "allocated_assets", "total_tickets",
                                  "assets_with_purchase_date", "assets_using_allocation_date", "coverage_pct"]},
        "healthy": int((A["risk_level"] == "Low").sum()),
        "medium_risk": int((A["risk_level"] == "Medium").sum()),
        "high_risk": int((A["risk_level"] == "High").sum()),
        "critical_risk": int((A["risk_level"] == "Critical").sum()),
        "maintenance_due": int(A["maintenance_overdue"].sum()),
        "replacement_due": int(A["replacement_recommended"].sum()),
        "top_problem_apartment": apartment_health.iloc[0]["apartment_code"] if not apartment_health.empty else "—",
        "top_problem_asset_type": asset_type_health.iloc[0]["asset_type"] if not asset_type_health.empty else "—",
    }

    return {
        "assets": A,                       # now includes reliability + lifecycle columns
        "apartment_health": apartment_health,
        "asset_type_health": asset_type_health,
        "calendar": calendar,
        "hotspots": hotspots,
        "rankings": rankings,
        "repeat_analysis": repeat_analysis,
        "preventive_queue": preventive_queue,
        "lifecycle_counts": lifecycle_counts,
        "exec_kpis": exec_kpis,
    }


# ==========================================================================  #
# EXTENSION 3 — management/business planning views (additive; nothing above    #
# is changed). Ticket-centric mapping unchanged. No ML. Read-only.             #
# ==========================================================================  #
def _replacement_cost_map(loader) -> Dict[str, float]:
    at = loader.asset_types()
    if at.empty or "name" not in at.columns:
        return {}
    return dict(zip(at["name"].map(_s), pd.to_numeric(at.get("replacement_cost_estimate"), errors="coerce")))


def budget_forecast(loader, built, bv, as_of=None) -> Dict[str, object]:
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    d = built["mapped"]["created_at"].dropna()
    last30 = int((d >= as_of - pd.Timedelta(days=30)).sum())
    last90 = int((d >= as_of - pd.Timedelta(days=90)).sum())
    monthly = (last90 / 3.0) if last90 else float(last30)
    avg_cost = _avg_ticket_cost(loader)
    windows = []
    for label, mult in [("Next 30 days", 1), ("Next 90 days", 3), ("Next 6 months", 6), ("Next 1 year", 12)]:
        jobs = int(round(monthly * mult))
        windows.append({"window": label, "expected_jobs": jobs, "estimated_budget": round(jobs * avg_cost, 0)})
    th = bv.get("asset_type_health")
    top_types = th[["asset_type", "total_failures"]].head(6) if th is not None and not th.empty else pd.DataFrame()
    return {"windows": pd.DataFrame(windows), "top_contributing_types": top_types,
            "avg_ticket_cost": round(avg_cost, 0), "monthly_rate": round(monthly, 1)}


def replacement_planner(loader, built, as_of=None) -> Dict[str, object]:
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    A = built["assets"].copy()
    if A.empty:
        return {"summary": pd.DataFrame(), "detail": pd.DataFrame()}
    rc = _replacement_cost_map(loader)
    A["replacement_cost"] = A["asset_type"].map(rc)
    life = pd.to_numeric(A["expected_life_months"], errors="coerce")
    age = pd.to_numeric(A["age_months"], errors="coerce")
    A["months_to_eol"] = (life - age).round(1)

    def bucket(r):
        m = r["months_to_eol"]
        if r["replacement_recommended"] or (pd.notna(m) and m <= 0) or \
           (r["risk_level"] == "Critical" and pd.notna(r["age_ratio"]) and r["age_ratio"] >= 1):
            return "Replace Immediately"
        if pd.notna(m):
            if m <= 3:
                return "Replace within 3 months"
            if m <= 6:
                return "Replace within 6 months"
            if m <= 12:
                return "Replace within 1 year"
        return ""
    A["replacement_window"] = A.apply(bucket, axis=1)
    plan = A[A["replacement_window"] != ""].copy()
    prio = {"Replace Immediately": "Critical", "Replace within 3 months": "High",
            "Replace within 6 months": "Medium", "Replace within 1 year": "Low"}
    plan["priority"] = plan["replacement_window"].map(prio)
    order = ["Replace Immediately", "Replace within 3 months", "Replace within 6 months", "Replace within 1 year"]
    summary = (plan.groupby("replacement_window")
               .agg(assets=("asset_id", "count"),
                    est_replacement_cost=("replacement_cost", lambda s: round(pd.to_numeric(s, errors="coerce").sum(), 0)))
               .reindex(order).dropna(how="all").reset_index())
    summary["priority"] = summary["replacement_window"].map(prio)
    detail = plan.sort_values("months_to_eol")[
        ["asset_code", "asset_type", "apartment_code", "replacement_window", "priority",
         "months_to_eol", "replacement_cost", "risk_level", "reason"]]
    return {"summary": summary, "detail": detail}


def workload_forecast(built, as_of=None) -> Dict[str, object]:
    as_of = pd.Timestamp(as_of).normalize() if as_of is not None else pd.Timestamp.today().normalize()
    mapped = built["mapped"]
    d = mapped["created_at"]
    last7 = int((d >= as_of - pd.Timedelta(days=7)).sum())
    last30 = int((d >= as_of - pd.Timedelta(days=30)).sum())
    prev30 = int(((d >= as_of - pd.Timedelta(days=60)) & (d < as_of - pd.Timedelta(days=30))).sum())
    trend = (last30 / prev30) if prev30 else 1.0
    windows = pd.DataFrame([
        {"window": "This week", "expected_tickets": last7},
        {"window": "Next week", "expected_tickets": int(round(last7 * min(max(trend, 0.5), 2)))},
        {"window": "This month", "expected_tickets": last30},
        {"window": "Next month", "expected_tickets": int(round(last30 * min(max(trend, 0.5), 2)))},
    ])
    recent = mapped[d >= as_of - pd.Timedelta(days=30)]
    by_apt = (recent[recent["apartment_code"] != ""].groupby("apartment_code").size()
              .sort_values(ascending=False).head(15).rename("expected_next_month").reset_index())
    by_type = (recent[recent["issue_type"] != ""].groupby("issue_type").size()
               .sort_values(ascending=False).head(15).rename("expected_next_month").reset_index())
    return {"windows": windows, "by_apartment": by_apt, "by_type": by_type}


def asset_roi(loader, built, bv) -> pd.DataFrame:
    A = bv.get("assets", built["assets"]) if bv else built["assets"]
    th = bv.get("asset_type_health")
    if A.empty or th is None or th.empty:
        return pd.DataFrame()
    life = pd.to_numeric(A["expected_life_months"], errors="coerce")
    age = pd.to_numeric(A["age_months"], errors="coerce")
    A2 = A.assign(_rul=(life - age))
    g = A2.groupby("asset_type").agg(reliability=("reliability_score", "mean"),
                                     avg_rul_months=("_rul", "mean"),
                                     avg_tickets=("ticket_count", "mean")).reset_index()
    out = th.merge(g, on="asset_type", how="left")
    out["reliability"] = out["reliability"].round(1)
    out["avg_rul_months"] = out["avg_rul_months"].round(1)
    out["avg_tickets"] = out["avg_tickets"].round(2)
    out = out.rename(columns={"total_assets": "purchase_count", "total_failures": "failure_count",
                              "failure_rate_per_asset": "maintenance_frequency"})
    out["poor_value"] = ((pd.to_numeric(out["maintenance_frequency"], errors="coerce") >= 1.5)
                         & (pd.to_numeric(out["reliability"], errors="coerce") < 70))
    cols = ["asset_type", "purchase_count", "failure_count", "maintenance_frequency",
            "avg_rul_months", "reliability", "poor_value"]
    return out[cols].sort_values("maintenance_frequency", ascending=False).reset_index(drop=True)


def apartment_report_card(built, bv) -> pd.DataFrame:
    ah = bv.get("apartment_health")
    if ah is None or ah.empty:
        return pd.DataFrame()
    A = built["assets"]
    acount = A.groupby("apartment_code")["asset_id"].count().rename("asset_count")
    prev = A.assign(_p=(A["maintenance_overdue"] | (pd.to_numeric(A["maintenance_due_days"], errors="coerce") <= 30))) \
            .groupby("apartment_code")["_p"].sum().rename("preventive_due")
    out = ah.merge(acount.reset_index(), on="apartment_code", how="left") \
            .merge(prev.reset_index(), on="apartment_code", how="left")
    out["asset_count"] = out["asset_count"].fillna(0).astype(int)
    out["preventive_due"] = out["preventive_due"].fillna(0).astype(int)
    out = out.sort_values("health_score", ascending=False).reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out[["rank", "apartment_code", "health_score", "health_level", "asset_count",
                "total_tickets", "high_risk_assets", "repeat_failures", "preventive_due"]]


def asset_performance(built, bv=None) -> pd.DataFrame:
    A = bv.get("assets", built["assets"]) if bv else built["assets"]
    if A.empty:
        return pd.DataFrame()
    g = A.groupby("asset_type").agg(
        assets=("asset_id", "count"),
        reliability=("reliability_score", "mean"),
        avg_failures=("ticket_count", "mean"),
        avg_age_months=("age_months", "mean"),
        high_risk=("risk_level", lambda s: int(s.isin(["High", "Critical"]).sum()))).reset_index()
    g["reliability"] = g["reliability"].round(1)
    g["avg_failures"] = g["avg_failures"].round(2)
    g["avg_age_months"] = g["avg_age_months"].round(1)
    g["high_risk_pct"] = (100 * g["high_risk"] / g["assets"]).round(1)
    # maintenance frequency = avg failures per asset per year of observed age
    g["maintenance_freq_per_year"] = (g["avg_failures"] / (g["avg_age_months"] / 12).clip(lower=0.25)).round(2)
    return g[["asset_type", "assets", "reliability", "avg_failures", "avg_age_months",
              "maintenance_freq_per_year", "high_risk_pct"]].sort_values("reliability").reset_index(drop=True)


def business_recommendations(built, bv, planner, report_card, roi) -> pd.DataFrame:
    recs = []
    A = built["assets"]
    # replace
    imm = planner["detail"][planner["detail"]["priority"] == "Critical"] if not planner["detail"].empty else pd.DataFrame()
    if len(imm):
        cost = round(pd.to_numeric(imm["replacement_cost"], errors="coerce").sum(), 0)
        recs.append({"recommendation": f"Replace {len(imm)} end-of-life asset(s)",
                     "why": "Past expected life / critical risk with chronic failures.",
                     "expected_impact": f"Cuts recurring failures; est. replacement ~₹{cost:,.0f}.",
                     "priority": "Critical"})
    # preventive
    overdue = int(A["maintenance_overdue"].sum()) if not A.empty else 0
    highrisk = int(A["risk_level"].isin(["High", "Critical"]).sum()) if not A.empty else 0
    if highrisk:
        recs.append({"recommendation": "Increase preventive maintenance",
                     "why": f"{highrisk} asset(s) at High/Critical risk ({overdue} overdue).",
                     "expected_impact": "Fewer emergency repairs; higher asset reliability.",
                     "priority": "High"})
    # monitor apartments
    if not report_card.empty:
        worst = report_card.sort_values("health_score").head(3)["apartment_code"].tolist()
        recs.append({"recommendation": f"Monitor apartments {', '.join(worst)}",
                     "why": "Lowest apartment health scores (high tickets/repeat failures).",
                     "expected_impact": "Targeted attention reduces ticket density in worst rooms.",
                     "priority": "High"})
    # review asset categories (poor ROI)
    if not roi.empty:
        poor = roi[roi["poor_value"]]["asset_type"].head(3).tolist()
        if poor:
            recs.append({"recommendation": f"Review asset categories: {', '.join(poor)}",
                         "why": "High maintenance frequency with low reliability (poor value).",
                         "expected_impact": "Vendor/spec review can lower lifecycle cost.",
                         "priority": "Medium"})
    # investigate repeats
    rep = bv.get("repeat_analysis")
    if rep is not None and len(rep):
        recs.append({"recommendation": f"Investigate {len(rep)} assets with repeated failures",
                     "why": "Recurring same-issue tickets suggest root-cause not fixed.",
                     "expected_impact": "Root-cause fixes remove repeat tickets.",
                     "priority": "Medium"})
    return pd.DataFrame(recs)


def asset_timeline_events(loader, built, asset_id) -> Dict[str, object]:
    _, _, first_alloc, _ = _allocations(loader)
    A = built["assets"]
    row = A[A["asset_id"] == asset_id]
    tks = built["mapped"]
    tks = tks[tks["asset_id"] == asset_id][["created_at", "issue_type", "confidence"]].sort_values("created_at")
    r = row.iloc[0] if not row.empty else None
    nxt = None
    if r is not None and pd.notna(r.get("maintenance_due_days")):
        nxt = (pd.Timestamp.today().normalize() + pd.Timedelta(days=float(r["maintenance_due_days"]))).date()
    return {
        "allocation_date": (str(first_alloc.get(asset_id))[:10] if first_alloc.get(asset_id) is not None and pd.notna(first_alloc.get(asset_id)) else None),
        "purchase_date": (str(r["purchase_date"])[:10] if (r is not None and pd.notna(r.get("purchase_date"))) else None),
        "last_maintenance": (str(r["last_ticket"])[:10] if (r is not None and pd.notna(r.get("last_ticket"))) else None),
        "next_predicted": str(nxt) if nxt is not None else None,
        "tickets": tks,
    }


# ==========================================================================  #
# EXTENSION 4 — SLA, Brand analysis, Purchase recommendation (additive).       #
# Auto-hide when the required columns are absent. No ML. Read-only.            #
# ==========================================================================  #
_CLOSED_SET = {"closed", "resolved", "completed", "done"}


def sla_dashboard(loader, built) -> Dict[str, object]:
    """SLA metrics from maintenance_tickets. Returns {} when timestamps absent."""
    mt = loader.maintenance_tickets().copy()
    if mt.empty or "created_at" not in mt.columns or "resolved_at" not in mt.columns:
        return {}
    ref = _reference(loader)
    i2t = ref["issue_to_types"]
    created = _dt(mt.get("created_at"))
    resolved = _dt(mt.get("resolved_at"))
    sla = _dt(mt.get("sla_deadline")) if "sla_deadline" in mt.columns else pd.Series(pd.NaT, index=mt.index)
    status = mt.get("status", pd.Series("", index=mt.index)).map(lambda v: _s(v).lower())
    res_h = (resolved - created).dt.total_seconds() / 3600.0
    res_h = res_h.where(res_h >= 0)
    is_open = ~status.isin(_CLOSED_SET)
    both = resolved.notna() & sla.notna()
    met = both & (resolved <= sla)
    viol = both & (resolved > sla)

    df = pd.DataFrame({
        "apartment_code": mt.get("apartment_id", "").map(lambda x: ref["apt_code"].get(_s(x), _s(x))),
        "issue_type": mt.get("issue_type_id", "").map(lambda x: ref["iname"].get(_s(x), "")),
        "asset_type": mt.get("issue_type_id", "").map(lambda x: (sorted(i2t.get(_s(x), set()))[0] if i2t.get(_s(x)) else "Unknown")),
        "technician": mt.get("assigned_to", "").map(_s) if "assigned_to" in mt.columns else "",
        "res_h": res_h, "open": is_open, "sla_both": both, "sla_met": met,
    })
    rr = res_h.dropna()
    n_both = int(both.sum())
    overall = {
        "total_tickets": int(len(mt)), "open_tickets": int(is_open.sum()),
        "closed_tickets": int((~is_open).sum()),
        "avg_resolution_hours": round(float(rr.mean()), 1) if len(rr) else None,
        "median_resolution_hours": round(float(rr.median()), 1) if len(rr) else None,
        "fastest_resolution_hours": round(float(rr.min()), 1) if len(rr) else None,
        "slowest_resolution_hours": round(float(rr.max()), 1) if len(rr) else None,
        "sla_met_pct": round(100 * met.sum() / n_both, 1) if n_both else None,
        "sla_violated_pct": round(100 * viol.sum() / n_both, 1) if n_both else None,
        "sla_measured_on": n_both, "resolution_measured_on": int(len(rr)),
    }

    def grp(col):
        g = df[df[col].astype(str) != ""].groupby(col)
        out = g.agg(tickets=("open", "size"),
                    avg_resolution_hours=("res_h", lambda s: round(float(s.mean()), 1) if s.notna().any() else np.nan),
                    sla_met_pct=("sla_met", lambda s: round(100 * s.sum() / max(df.loc[s.index, "sla_both"].sum(), 1), 1))).reset_index()
        return out.sort_values("tickets", ascending=False)

    tech = pd.DataFrame()
    if "assigned_to" in mt.columns and (df["technician"].astype(str) != "").any():
        tg = df[df["technician"] != ""].groupby("technician")
        tech = tg.agg(assigned=("open", "size"),
                      closed=("open", lambda s: int((~s).sum())),
                      avg_resolution_hours=("res_h", lambda s: round(float(s.mean()), 1) if s.notna().any() else np.nan),
                      sla_met_pct=("sla_met", lambda s: round(100 * s.sum() / max(df.loc[s.index, "sla_both"].sum(), 1), 1))).reset_index()
        tech["technician"] = tech["technician"].str[:8] + "…"   # id (no name table) — shortened
        tech = tech.sort_values("assigned", ascending=False)

    return {"overall": overall, "by_apartment": grp("apartment_code"),
            "by_issue_type": grp("issue_type"), "by_asset_type": grp("asset_type"),
            "technician": tech, "has_technician": not tech.empty}


def brand_analysis(built, bv) -> Dict[str, object]:
    """Per-brand reliability from assets. Returns {} when brand column absent/empty."""
    A = bv.get("assets", built["assets"]) if bv else built["assets"]
    if A.empty or "brand" not in A.columns:
        return {}
    b = A[A["brand"].astype(str).str.strip().replace({"nan": "", "None": ""}) != ""].copy()
    if b.empty:
        return {}
    g = b.groupby("brand").agg(
        total_assets=("asset_id", "count"),
        total_failures=("ticket_count", "sum"),
        avg_age_months=("age_months", "mean"),
        avg_health=("health_score", "mean"),
        avg_reliability=("reliability_score", "mean"),
        high_risk_assets=("risk_level", lambda s: int(s.isin(["High", "Critical"]).sum())),
        replacement_count=("replacement_recommended", "sum")).reset_index()
    g["failure_rate"] = (g["total_failures"] / g["total_assets"]).round(2)
    for c in ("avg_age_months", "avg_health", "avg_reliability"):
        g[c] = g[c].round(1)
    g["replacement_count"] = g["replacement_count"].astype(int)
    most_reliable = g.sort_values("avg_reliability", ascending=False).head(10)
    worst = g.sort_values(["failure_rate", "avg_reliability"], ascending=[False, True]).head(10)

    # recommendations: within an asset_type, compare brands by failure rate
    recs = []
    bt = b.groupby(["asset_type", "brand"]).agg(assets=("asset_id", "count"),
                                                failures=("ticket_count", "sum")).reset_index()
    bt["rate"] = bt["failures"] / bt["assets"]
    for at, sub in bt.groupby("asset_type"):
        sub = sub[sub["assets"] >= 2].sort_values("rate", ascending=False)
        if len(sub) >= 2 and sub.iloc[-1]["rate"] > 0:
            hi, lo = sub.iloc[0], sub.iloc[-1]
            ratio = hi["rate"] / lo["rate"] if lo["rate"] else np.inf
            if ratio >= 1.5:
                recs.append(f"{hi['brand']} {at} has {ratio:.1f}× higher failure rate than {lo['brand']}.")
    return {"by_brand": g.sort_values("avg_reliability", ascending=False),
            "most_reliable": most_reliable, "worst_performing": worst, "recommendations": recs}


def purchase_recommendation(loader, built, bv) -> pd.DataFrame:
    """Per asset-type purchase guidance (best brand when several exist)."""
    A = bv.get("assets", built["assets"]) if bv else built["assets"]
    if A.empty:
        return pd.DataFrame()
    life = pd.to_numeric(A["expected_life_months"], errors="coerce")
    age = pd.to_numeric(A["age_months"], errors="coerce")
    A2 = A.assign(_rul=(life - age))
    rows = []
    for at, sub in A2.groupby("asset_type"):
        if not str(at).strip():
            continue
        brands = sub[sub["brand"].astype(str).str.strip().replace({"nan": "", "None": ""}) != ""]
        if brands["brand"].nunique() >= 2:
            bg = brands.groupby("brand").agg(rel=("reliability_score", "mean"),
                                             assets=("asset_id", "count")).reset_index()
            best = bg.sort_values("rel", ascending=False).iloc[0]["brand"]
            cur_brand = f"{brands['brand'].value_counts().idxmax()} (best: {best})"
        elif brands["brand"].nunique() == 1:
            cur_brand = brands["brand"].iloc[0]
        else:
            cur_brand = "—"
        rel = round(float(sub["reliability_score"].mean()), 1)
        fr = round(float(sub["ticket_count"].mean()), 2)
        rul = round(float(sub["_rul"].mean()), 1) if sub["_rul"].notna().any() else np.nan
        freq = round(float((sub["ticket_count"] / (age.loc[sub.index] / 12).clip(lower=0.25)).mean()), 2) if age.loc[sub.index].notna().any() else np.nan
        if rel >= 75:
            rec, reason, benefit, prio = "Continue Purchasing", "High reliability, low failure load.", "Stable lifecycle cost.", "Low"
        elif rel >= 55:
            rec, reason, benefit, prio = "Monitor", "Moderate reliability.", "Watch failure trend before scaling.", "Medium"
        elif rel >= 35:
            rec, reason, benefit, prio = "Reduce Purchase", "Below-par reliability / frequent failures.", "Lower maintenance spend.", "High"
        else:
            rec, reason, benefit, prio = "Avoid Purchase", "Poor reliability, high failure rate.", "Avoid recurring failure cost.", "Critical"
        rows.append({"asset_type": at, "current_brand": cur_brand, "failure_rate": fr,
                     "avg_reliability": rel, "avg_maintenance_frequency_per_year": freq,
                     "avg_remaining_useful_life_months": rul, "recommendation": rec,
                     "reason": reason, "expected_benefit": benefit, "priority": prio})
    order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    return pd.DataFrame(rows).sort_values("priority", key=lambda s: s.map(order)).reset_index(drop=True)
