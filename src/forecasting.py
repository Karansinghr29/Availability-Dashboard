"""
forecasting.py — ADDITIVE time-series forecasting layer for maintenance workload.

Purpose
-------
Forecast NEXT MONTH's maintenance ticket volume at three levels:
  * Portfolio   (all tickets)
  * Apartment   (per apartment_code)
  * Asset Type  (per asset type, via the issue -> asset-type bridge)

This does NOT replace the rule engine (asset_engine.py) or its Poisson/health
scores. It is a separate analytics module the dashboard reads on demand.

Timeline
--------
Uses the UNIFIED event date = created_at else resolved_at else closed_at, giving
~18 months of history (created_at alone is only ~30% populated / 4 months).
Tickets are counted per calendar month into a gap-filled monthly series.

Models compared (per series, chronological hold-out):
  * Seasonal Naive   — value from 12 months ago (falls back to last value)
  * Holt-Winters ETS — additive trend (+ additive seasonality when >= 24 months)
  * SARIMA           — SARIMAX; seasonal only when >= 24 months, else ARIMA(1,1,1)
  * Prophet          — if installed; monthly, yearly seasonality auto

Selection: lowest MAPE on the hold-out (MAE tie-break). The winner is refit on
the FULL series and used to forecast the next month. Everything is wrapped in
try/except so a model that fails on a short series simply drops out.

NOTE ON DATA LENGTH (honest caveat surfaced to the UI):
18-19 monthly points is enough for simple/seasonal-naive/ETS and marginal for
SARIMA/Prophet yearly seasonality (which prefer >= 24). Short per-entity series
(< ~8 months) are forecast by naive/mean only and flagged low-confidence.
"""

from __future__ import annotations

import warnings
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_MIN_TRAIN = 6          # minimum months to attempt a real model
_TEST_MONTHS = 3        # chronological hold-out length (falls back to 2/1 for short series)
_SEASON = 12


# --------------------------------------------------------------------------- #
# 1. Build the unified monthly series
# --------------------------------------------------------------------------- #
def _s(v) -> str:
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v).strip()


def _event_dates(mt: pd.DataFrame) -> pd.Series:
    """created_at else resolved_at else closed_at, tz-naive."""
    def col(c):
        return pd.to_datetime(mt[c], errors="coerce", utc=True).dt.tz_localize(None) if c in mt.columns else pd.Series(pd.NaT, index=mt.index)
    return col("created_at").fillna(col("resolved_at")).fillna(col("closed_at"))


def _monthly(dates: pd.Series) -> pd.Series:
    """Gap-filled monthly count series (Timestamp index at month start)."""
    d = dates.dropna()
    if d.empty:
        return pd.Series(dtype=float)
    m = d.dt.to_period("M").value_counts().sort_index()
    idx = pd.period_range(m.index.min(), m.index.max(), freq="M")
    return m.reindex(idx, fill_value=0).astype(float).rename_axis("month").rename("tickets").to_timestamp()


def build_series(loader) -> Dict[str, object]:
    """Return {'portfolio': Series, 'by_apartment': {code: Series}, 'by_type': {type: Series}}."""
    mt = loader.maintenance_tickets().copy()
    if mt.empty:
        return {"portfolio": pd.Series(dtype=float), "by_apartment": {}, "by_type": {}}
    mt["_ev"] = _event_dates(mt)

    portfolio = _monthly(mt["_ev"])

    # apartment level (apartment resolved via bed_id -> beds_master, else apartment_id)
    beds, aptm = loader.beds_master_uuid(), loader.apartment_master()
    bapt = dict(zip(beds["id"].map(_s), beds["apartment_id"].map(_s))) if (not beds.empty and "apartment_id" in beds.columns) else {}
    aptcode = dict(zip(aptm["id"].map(_s), aptm["apartment_code"].map(_s))) if not aptm.empty else {}
    bid = mt["bed_id"].map(_s) if "bed_id" in mt.columns else pd.Series("", index=mt.index)
    apt_via_bed = bid.map(lambda b: aptcode.get(bapt.get(b, ""), ""))
    apt_direct = mt["apartment_id"].map(lambda x: aptcode.get(_s(x), _s(x))) if "apartment_id" in mt.columns else pd.Series("", index=mt.index)
    mt["_apt"] = [a if a else d for a, d in zip(apt_via_bed, apt_direct)]
    by_apt = {ap: _monthly(g["_ev"]) for ap, g in mt[mt["_apt"].astype(bool)].groupby("_apt")}

    # asset-type level (issue_type_id -> asset types via maintenance_items)
    items, atypes = loader.maintenance_items(), loader.asset_types()
    tname = dict(zip(atypes["id"].map(_s), atypes["name"].map(_s))) if not atypes.empty else {}
    i2t: Dict[str, set] = {}
    if not items.empty:
        for _, r in items.iterrows():
            it, at = _s(r.get("issue_type_id")), tname.get(_s(r.get("asset_type_id")), "")
            if it and at:
                i2t.setdefault(it, set()).add(at)
    rows = []
    for ev, it in zip(mt["_ev"], mt.get("issue_type_id", pd.Series("", index=mt.index)).map(_s)):
        for at in i2t.get(it, set()):
            rows.append((at, ev))
    td = pd.DataFrame(rows, columns=["at", "ev"])
    by_type = {at: _monthly(g["ev"]) for at, g in td.groupby("at")} if not td.empty else {}

    return {"portfolio": portfolio, "by_apartment": by_apt, "by_type": by_type}


# --------------------------------------------------------------------------- #
# 2. Models (each returns a 1..h step forecast array, or None on failure)
# --------------------------------------------------------------------------- #
def _fc_seasonal_naive(train: pd.Series, h: int):
    if len(train) >= _SEASON:
        base = train.iloc[-_SEASON:].values
        return np.array([base[i % _SEASON] for i in range(h)])
    return np.repeat(train.iloc[-1], h)  # fallback: last value


def _fc_holt_winters(train: pd.Series, h: int):
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    seasonal = "add" if len(train) >= 2 * _SEASON else None
    sp = _SEASON if seasonal else None
    trend = "add" if len(train) >= 4 else None
    m = ExponentialSmoothing(train, trend=trend, seasonal=seasonal, seasonal_periods=sp,
                             initialization_method="estimated").fit()
    return np.asarray(m.forecast(h))


def _fc_sarima(train: pd.Series, h: int):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    if len(train) >= 2 * _SEASON:
        order, sorder = (1, 1, 1), (0, 1, 1, _SEASON)
    else:
        order, sorder = (1, 1, 1), (0, 0, 0, 0)   # plain ARIMA when too short for yearly season
    m = SARIMAX(train, order=order, seasonal_order=sorder,
                enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    return np.asarray(m.forecast(h))


def _fc_prophet(train: pd.Series, h: int):
    try:
        from prophet import Prophet
    except Exception:
        return None
    dfp = pd.DataFrame({"ds": train.index, "y": train.values})
    m = Prophet(yearly_seasonality=(len(train) >= _SEASON), weekly_seasonality=False,
                daily_seasonality=False)
    m.fit(dfp)
    future = m.make_future_dataframe(periods=h, freq="MS")
    return np.asarray(m.predict(future)["yhat"].iloc[-h:].values)


_MODELS = {
    "Seasonal Naive": _fc_seasonal_naive,
    "Holt-Winters": _fc_holt_winters,
    "SARIMA": _fc_sarima,
    "Prophet": _fc_prophet,
}


# --------------------------------------------------------------------------- #
# 3. Evaluate (chronological split) + select best + forecast next month
# --------------------------------------------------------------------------- #
def _mae(a, f):
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(f))))


def _mape(a, f):
    a = np.asarray(a, dtype=float); f = np.asarray(f, dtype=float)
    mask = a != 0
    if not mask.any():
        return None
    return float(np.mean(np.abs((a[mask] - f[mask]) / a[mask])) * 100)


def forecast_series(series: pd.Series, label: str = "", use_prophet: bool = True) -> Optional[Dict[str, object]]:
    """Chronological hold-out eval of all models, pick best by MAPE (MAE tie-break),
    refit winner on full series, forecast next month.

    use_prophet: Prophet's cmdstanpy backend costs ~15s/fit in this environment, so it is
    compared on the Portfolio series but skipped per-entity for dashboard responsiveness
    (Seasonal Naive / Holt-Winters / SARIMA remain, all sub-second). Still 'compared where
    it is available'."""
    models = _MODELS if use_prophet else {k: v for k, v in _MODELS.items() if k != "Prophet"}
    s = series.dropna().astype(float)
    n = len(s)
    if n < _MIN_TRAIN:
        # too short for evaluation -> naive point forecast, flagged low confidence
        nxt = float(s.iloc[-1]) if n else 0.0
        return {
            "label": label, "n_months": n, "best_model": "Naive (insufficient history)",
            "metrics": {}, "forecast_next": round(nxt, 1),
            "forecast_lower": round(nxt, 1), "forecast_upper": round(nxt, 1), "confidence_pct": None,
            "next_period": (s.index.max() + pd.offsets.MonthBegin(1)) if n else None,
            "history": s, "confidence": "Low",
        }

    test_h = min(_TEST_MONTHS, max(1, n - _MIN_TRAIN))
    train, test = s.iloc[:-test_h], s.iloc[-test_h:]
    metrics = {}
    for name, fn in models.items():
        try:
            pred = fn(train, test_h)
            if pred is None or len(pred) != test_h or not np.all(np.isfinite(pred)):
                continue
            pred = np.clip(pred, 0, None)   # counts can't be negative
            metrics[name] = {"MAE": round(_mae(test.values, pred), 2),
                             "MAPE": (round(_mape(test.values, pred), 1) if _mape(test.values, pred) is not None else None)}
        except Exception:
            continue
    if not metrics:
        nxt = float(s.iloc[-1])
        return {"label": label, "n_months": n, "best_model": "Naive (models failed)",
                "metrics": {}, "forecast_next": round(nxt, 1),
                "forecast_lower": round(nxt, 1), "forecast_upper": round(nxt, 1), "confidence_pct": None,
                "next_period": s.index.max() + pd.offsets.MonthBegin(1),
                "history": s, "confidence": "Low"}

    def _key(m):
        mp = metrics[m]["MAPE"]
        return (mp if mp is not None else 1e9, metrics[m]["MAE"])
    best = min(metrics, key=_key)

    # refit winner on FULL series -> next-month forecast
    try:
        full_pred = _MODELS[best](s, 1)
        nxt = float(np.clip(full_pred[-1], 0, None))
    except Exception:
        nxt = float(s.iloc[-1])

    # ---- prediction interval for the winning model (native where available) ---- #
    lo, hi = _interval_for(best, s, nxt)

    best_mape = metrics[best]["MAPE"]
    conf = "High" if (best_mape is not None and best_mape <= 15 and n >= _SEASON) \
        else "Moderate" if (best_mape is not None and best_mape <= 30) else "Low"
    return {
        "label": label, "n_months": n, "best_model": best, "metrics": metrics,
        "forecast_next": round(nxt, 1),
        "forecast_lower": round(lo, 1), "forecast_upper": round(hi, 1), "confidence_pct": 95,
        "next_period": s.index.max() + pd.offsets.MonthBegin(1),
        "history": s, "confidence": conf,
        "test_months": test_h,
    }


# --------------------------------------------------------------------------- #
# 4. Prediction intervals (native statsmodels / Prophet, else residual-based)
# --------------------------------------------------------------------------- #
def _interval_for(best: str, s: pd.Series, point: float, alpha: float = 0.05):
    """95% interval for the +1 forecast. Uses SARIMAX conf_int / Prophet yhat bounds
    when the winner is SARIMA/Prophet, else a residual-std normal band. Never negative."""
    z = 1.96
    try:
        if best == "SARIMA":
            from statsmodels.tsa.statespace.sarimax import SARIMAX
            order, sorder = ((1, 1, 1), (0, 1, 1, _SEASON)) if len(s) >= 2 * _SEASON else ((1, 1, 1), (0, 0, 0, 0))
            r = SARIMAX(s, order=order, seasonal_order=sorder,
                        enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            ci = r.get_forecast(1).conf_int(alpha=alpha)
            return float(np.clip(ci.iloc[-1, 0], 0, None)), float(ci.iloc[-1, 1])
        if best == "Prophet":
            from prophet import Prophet
            dfp = pd.DataFrame({"ds": s.index, "y": s.values})
            m = Prophet(interval_width=1 - alpha, yearly_seasonality=(len(s) >= _SEASON),
                        weekly_seasonality=False, daily_seasonality=False).fit(dfp)
            pr = m.predict(m.make_future_dataframe(periods=1, freq="MS")).iloc[-1]
            return float(np.clip(pr["yhat_lower"], 0, None)), float(pr["yhat_upper"])
        if best == "Holt-Winters":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            seasonal = "add" if len(s) >= 2 * _SEASON else None
            trend = "add" if len(s) >= 4 else None
            r = ExponentialSmoothing(s, trend=trend, seasonal=seasonal,
                                     seasonal_periods=_SEASON if seasonal else None,
                                     initialization_method="estimated").fit()
            sigma = float(np.nanstd(r.resid))
        else:  # Seasonal Naive / fallback -> seasonal (or first) difference volatility
            diff = s - s.shift(_SEASON if len(s) >= _SEASON else 1)
            sigma = float(diff.dropna().std())
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = float(s.std())
        return max(0.0, point - z * sigma), point + z * sigma
    except Exception:
        sig = float(s.std()) if np.isfinite(s.std()) else 0.0
        return max(0.0, point - z * sig), point + z * sig


# --------------------------------------------------------------------------- #
# 5. Forecast-vs-Actual backtest (walk-forward, one-step)
# --------------------------------------------------------------------------- #
def backtest(series: pd.Series, model_name: str, last: int = 12) -> pd.DataFrame:
    """Walk-forward one-step backtest of the CHOSEN production model over the most recent
    `last` completed months: at each origin, train on the past, forecast that month, compare
    to actual. Automatically extends as new months arrive."""
    s = series.dropna().astype(float)
    n = len(s)
    cols = ["month", "predicted", "actual", "abs_error", "pct_error", "model"]
    if n < _MIN_TRAIN + 1:
        return pd.DataFrame(columns=cols)
    fn = _MODELS.get(model_name, _fc_seasonal_naive)
    start = max(_MIN_TRAIN, n - last)
    rows = []
    for t in range(start, n):
        train, actual = s.iloc[:t], float(s.iloc[t])
        try:
            pred = float(np.clip(fn(train, 1)[-1], 0, None))
        except Exception:
            pred = float(train.iloc[-1])
        ae = abs(actual - pred)
        pe = round(100 * ae / actual, 1) if actual else None
        rows.append({"month": s.index[t], "predicted": round(pred, 1), "actual": round(actual, 1),
                     "abs_error": round(ae, 1), "pct_error": pe, "model": model_name})
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# 6. Cost basis + business-friendly explainability
# --------------------------------------------------------------------------- #
def avg_ticket_cost(loader):
    """(avg_cost, is_real_closure_cost). closure_cost is null in this dataset, so this
    falls back to ticket_resolutions.total_cost mean, else a 500 default (flagged)."""
    for acc, col in [("maintenance_tickets", "closure_cost"), ("ticket_resolutions", "total_cost")]:
        try:
            df = getattr(loader, acc)()
        except Exception:
            continue
        if df is not None and not df.empty and col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(v):
                return float(v.mean()), (col == "closure_cost")
    return 500.0, False


def explain_forecast(series: Dict[str, object], k: int = 3) -> Dict[str, object]:
    """Business-friendly reasons for the portfolio direction, from the historical inputs
    (no SHAP / black box). Compares the recent k months vs the previous k months."""
    port = series.get("portfolio")
    out = {"headline": "", "bullets": []}
    if port is None or len(port.dropna()) < 2 * k:
        return out
    p = port.dropna().astype(float)
    recent, prev = p.iloc[-k:].mean(), p.iloc[-2 * k:-k].mean()
    if prev > 0:
        chg = (recent - prev) / prev * 100
        out["headline"] = (f"Portfolio workload is trending "
                           f"{'up' if chg >= 0 else 'down'} {abs(chg):.0f}% "
                           f"(last {k} mo avg {recent:.0f} vs prior {k} mo {prev:.0f}).")

    def _delta(sr):
        sr = sr.dropna().astype(float)
        if len(sr) < 2 * k:
            return None, None, None
        r, q = sr.iloc[-k:].sum(), sr.iloc[-2 * k:-k].sum()
        pct = ((r - q) / q * 100) if q > 0 else (100.0 if r > 0 else 0.0)
        return pct, r, q

    # asset-type movers (collapse types that share the same % — the issue->type bridge
    # maps generic issues to several types, giving them identical series)
    movers = []
    for at, sr in series.get("by_type", {}).items():
        pct, r, _ = _delta(sr)
        if pct is not None and abs(pct) >= 5 and r >= 3:
            movers.append((at, pct))
    movers.sort(key=lambda x: -abs(x[1]))
    seen_pct = {}
    for at, pct in movers:
        key = round(pct)
        seen_pct.setdefault(key, []).append(at)
    for key in sorted(seen_pct, key=lambda k: -abs(k))[:4]:
        types = seen_pct[key]
        label = types[0] if len(types) == 1 else f"{', '.join(types[:3])}" + (f" (+{len(types) - 3} more)" if len(types) > 3 else "")
        out["bullets"].append(f"{label} complaints {'increased' if key >= 0 else 'decreased'} {abs(key):.0f}%")

    # top apartment contribution (recent k months share)
    apt_recent = {ap: sr.dropna().iloc[-k:].sum() for ap, sr in series.get("by_apartment", {}).items()}
    tot = sum(apt_recent.values())
    if tot > 0:
        top_ap, top_v = max(apt_recent.items(), key=lambda kv: kv[1])
        out["bullets"].append(f"Apartment {top_ap} contributes {100 * top_v / tot:.0f}% of recent tickets")

    # seasonal contribution: target month historical mean vs overall mean
    nxt_month = (p.index.max() + pd.offsets.MonthBegin(1)).month
    same = p[p.index.month == nxt_month]
    if len(same) >= 1 and p.mean() > 0:
        seff = (same.mean() - p.mean()) / p.mean() * 100
        mname = pd.Timestamp(2000, nxt_month, 1).strftime("%B")
        out["bullets"].append(f"Seasonal trend for {mname}: {seff:+.0f}% vs the overall monthly average")
    return out


def run_forecasts(loader, top_apartments: int = 15, prophet_entities: bool = False) -> Dict[str, object]:
    """Full forecasting run: portfolio + asset types + top-N apartments by volume.

    Prophet is always compared on the Portfolio series; per-entity it is off by default
    (prophet_entities=False) because its cmdstanpy fit is ~15s/series — enabling it for
    ~24 entities would make the page take minutes. The three fast models still compete."""
    series = build_series(loader)
    out = {"portfolio": None, "by_type": [], "by_apartment": [], "as_of": None,
           "prophet_scope": "portfolio only" if not prophet_entities else "all levels"}

    # cost basis + business explainability (additive)
    out["avg_ticket_cost"], out["cost_is_real"] = avg_ticket_cost(loader)
    out["explain"] = explain_forecast(series)

    p = series["portfolio"]
    if not p.empty:
        out["as_of"] = str(p.index.max().date())
        out["portfolio"] = forecast_series(p, "Portfolio", use_prophet=True)
        # forecast-vs-actual backtest of the chosen portfolio model (last 12 months)
        out["portfolio_backtest"] = backtest(p, out["portfolio"]["best_model"], last=12)

    for at, s in sorted(series["by_type"].items(), key=lambda kv: -kv[1].sum()):
        r = forecast_series(s, at, use_prophet=prophet_entities)
        if r:
            out["by_type"].append(r)

    apts = sorted(series["by_apartment"].items(), key=lambda kv: -kv[1].sum())[:top_apartments]
    for ap, s in apts:
        r = forecast_series(s, ap, use_prophet=prophet_entities)
        if r:
            out["by_apartment"].append(r)
    return out
