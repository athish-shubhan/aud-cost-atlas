"""Spread forecasting: next-hour average spread (with intervals) and blowout probability.

Decisions are made every 15 minutes. Features use only data up to the decision time
plus the *published* event schedule. Evaluation is walk-forward by calendar month:
train on every earlier month, test on the next one.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .calendar import minutes_to_next
from .data import typical_spread

QUANTILES = (0.1, 0.5, 0.9)
FEATURES = ["hour_sin", "hour_cos", "dow", "min_to_event", "event_within_60", "spr_15", "spr_60",
            "quotes_15", "rv_15", "rv_60", "typical", "rollover_next_hour"]


def dataset(bars: pd.DataFrame, cal: pd.DataFrame) -> pd.DataFrame:
    s = bars["spread_bps"]
    full = pd.date_range(bars.index[0].ceil("15min"), bars.index[-1], freq="15min", tz="UTC")
    full = full[full.isin(bars.index)]
    # trailing aggregates over the minutes *before* each decision time
    roll = lambda col, w, f: getattr(bars[col].rolling(f"{w}min", closed="left"), f)()
    feats = pd.DataFrame(index=full)
    feats["spr_15"] = roll("spread_bps", 15, "mean").reindex(full)
    feats["spr_60"] = roll("spread_bps", 60, "mean").reindex(full)
    feats["quotes_15"] = np.log1p(roll("quotes", 15, "sum").reindex(full))
    feats["rv_15"] = np.sqrt((bars["rv_bps"] ** 2).rolling("15min", closed="left").sum()).reindex(full)
    feats["rv_60"] = np.sqrt((bars["rv_bps"] ** 2).rolling("60min", closed="left").sum()).reindex(full)
    syd = full.tz_convert(ZoneInfo("Australia/Sydney"))
    feats["hour_sin"] = np.sin(2 * np.pi * (syd.hour + syd.minute / 60) / 24)
    feats["hour_cos"] = np.cos(2 * np.pi * (syd.hour + syd.minute / 60) / 24)
    feats["dow"] = syd.dayofweek
    feats["min_to_event"] = minutes_to_next(full, cal).to_numpy()
    feats["event_within_60"] = (feats["min_to_event"] < 60).astype(int)
    typ = typical_spread(bars)
    feats["typical"] = typ.reindex(full).to_numpy()
    nyc = full.tz_convert(ZoneInfo("America/New_York"))
    feats["rollover_next_hour"] = ((nyc.hour == 16) | ((nyc.hour == 17) & (nyc.minute < 15))).astype(int)
    # targets (forward-looking, labels only)
    feats["y_next_hour"] = _forward_mean(s, full, 60)
    mx15 = _forward_max(s, full, 15)
    feats["y_blowout_15"] = pd.Series(mx15 > 2 * feats["typical"], index=feats.index).astype(float).where(feats["typical"].notna())
    feats["slot"] = (full.hour * 4 + full.minute // 15)
    feats["month"] = full.tz_convert("UTC").strftime("%Y-%m")
    return feats.dropna()


def _ns(idx: pd.DatetimeIndex) -> np.ndarray:
    return idx.tz_convert(None).to_numpy().astype("datetime64[ns]")


def _forward_mean(s: pd.Series, at: pd.DatetimeIndex, minutes: int) -> np.ndarray:
    """Mean of s over [at, at + minutes) for each decision time."""
    t = _ns(pd.DatetimeIndex(s.index))
    c = np.r_[0.0, np.cumsum(s.to_numpy())]
    a = np.searchsorted(t, _ns(at), side="left")
    b = np.searchsorted(t, _ns(at + pd.Timedelta(minutes=minutes)), side="left")
    n = b - a
    return np.where(n > 0, (c[b] - c[a]) / np.maximum(n, 1), np.nan)


def _forward_max(s: pd.Series, at: pd.DatetimeIndex, minutes: int) -> np.ndarray:
    """Max of s over [at, at + minutes) for each decision time."""
    t = _ns(pd.DatetimeIndex(s.index))
    v = s.to_numpy()
    a = np.searchsorted(t, _ns(at), side="left")
    b = np.searchsorted(t, _ns(at + pd.Timedelta(minutes=minutes)), side="left")
    return np.array([v[i:j].max() if j > i else np.nan for i, j in zip(a, b, strict=True)])


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    d = y - q
    return float(np.mean(np.maximum(alpha * d, (alpha - 1) * d)))


def walk_forward(ds: pd.DataFrame, min_train_months: int = 3, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-row OOS predictions, per-month metrics) for baseline and XGBoost."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier, XGBRegressor

    months = sorted(ds["month"].unique())
    preds = []
    for m in months[min_train_months:]:
        tr, te = ds[ds["month"] < m], ds[ds["month"] == m]
        out = te[["slot", "min_to_event", "y_next_hour", "y_blowout_15"]].copy()
        # seasonal baseline: quantiles of the target by 15-minute slot in training data
        qs = tr.groupby("slot")["y_next_hour"].quantile(list(QUANTILES)).unstack()
        for a in QUANTILES:
            out[f"base_q{a}"] = qs[a].reindex(te["slot"]).to_numpy()
        out["base_p"] = tr.groupby("slot")["y_blowout_15"].mean().reindex(te["slot"]).fillna(
            tr["y_blowout_15"].mean()).to_numpy()
        reg = XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(QUANTILES),
                           n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                           tree_method="hist", random_state=seed, n_jobs=2)
        reg.fit(tr[FEATURES], tr["y_next_hour"])
        p = np.sort(reg.predict(te[FEATURES]), axis=1)
        for i, a in enumerate(QUANTILES):
            out[f"xgb_q{a}"] = p[:, i]
        lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)).fit(tr[FEATURES], tr["y_blowout_15"])
        out["logit_p"] = lr.predict_proba(te[FEATURES])[:, 1]
        clf = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                            tree_method="hist", random_state=seed, n_jobs=2, eval_metric="logloss")
        clf.fit(tr[FEATURES], tr["y_blowout_15"].astype(int))
        out["xgb_p"] = clf.predict_proba(te[FEATURES])[:, 1]
        out["month"] = m
        preds.append(out)
    pr = pd.concat(preds)
    rows = []
    for (m, near), g in pr.groupby(["month", pr["min_to_event"] < 60]):
        y = g["y_next_hour"].to_numpy()
        r = {"month": m, "near_event": bool(near), "n": len(g)}
        for model in ("base", "xgb"):
            r[f"{model}_pinball"] = np.mean([pinball(y, g[f"{model}_q{a}"].to_numpy(), a) for a in QUANTILES])
            r[f"{model}_coverage80"] = float(np.mean((y >= g[f"{model}_q0.1"]) & (y <= g[f"{model}_q0.9"])))
        yb = g["y_blowout_15"].to_numpy()
        for model in ("base", "logit", "xgb"):
            r[f"{model}_brier"] = float(np.mean((g[f"{model}_p"] - yb) ** 2))
        rows.append(r)
    return pr, pd.DataFrame(rows)
