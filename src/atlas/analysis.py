"""Cost heatmap and event study."""

from __future__ import annotations

from typing import cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from skeptic.boot import block_bootstrap_by_group
from skeptic.multiple import adjust

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def heatmap(bars: pd.DataFrame, tz: str = "Australia/Sydney", n_boot: int = 300,
            seed: int = 0) -> pd.DataFrame:
    """Mean quoted spread by local hour and weekday, with a 95% interval from resampling days.

    Weekdays are Sydney-local, so Saturday early morning in Sydney is Friday afternoon in
    New York and is included. Returns one row per (weekday, hour) with mean, lower, upper
    and the number of days.
    """
    loc = pd.DatetimeIndex(bars.index).tz_convert(ZoneInfo(tz))
    df = pd.DataFrame({"spread": bars["spread_bps"].to_numpy(), "dow": loc.dayofweek,
                       "hour": loc.hour, "day": loc.normalize()})
    daily = df.groupby(["dow", "hour", "day"])["spread"].mean().reset_index()
    rng = np.random.default_rng(seed)
    rows = []
    for (d, h), g in daily.groupby(["dow", "hour"]):
        v = g["spread"].to_numpy()
        boots = rng.choice(v, size=(n_boot, v.size), replace=True).mean(axis=1)
        lo, hi = np.quantile(boots, [0.025, 0.975])
        rows.append({"weekday": WEEKDAYS[cast(int, d)], "dow": d, "hour": h, "mean_bps": v.mean(), "lo": lo, "hi": hi,
                     "days": v.size})
    return pd.DataFrame(rows)


def rank_windows(hm: pd.DataFrame, k: int = 5, min_days: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    h = hm[hm["days"] >= min_days]
    cols = ["weekday", "hour", "mean_bps", "lo", "hi", "days"]
    return h.nsmallest(k, "mean_bps")[cols], h.nlargest(k, "mean_bps")[cols]


def event_study(bars: pd.DataFrame, events: pd.DataFrame, window: int = 30, n_boot: int = 1000,
                seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spread path from -window to +window minutes around each event, in event time.

    Each event's path is divided by a control path: the mean spread at the same UTC clock
    minutes on non-event days with the same weekday (within +-8 weeks). So 1.0 means "no
    different from a normal day at this time", and the time-of-day pattern cancels out.

    Returns (per-minute summary by event type with bootstrap bands over events,
             per-event peak excess with a p-value).
    """
    s = bars["spread_bps"]
    ev_days = set(events["time_utc"].dt.normalize())
    paths, peaks = [], []
    offsets = np.arange(-window, window + 1)
    for ev, t in zip(events["event"], events["time_utc"], strict=True):
        t = t.floor("min")
        idx = t + pd.to_timedelta(offsets, unit="m")
        path = s.reindex(idx).to_numpy()
        if np.isnan(path).mean() > 0.2:
            continue
        ctrl = []
        for wk in range(-8, 9):
            if wk == 0:
                continue
            day = t + pd.Timedelta(weeks=wk)
            if day.normalize() in ev_days:
                continue
            ctrl.append(s.reindex(day + pd.to_timedelta(offsets, unit="m")).to_numpy())
        if len(ctrl) < 3:
            continue
        ctrl_path = np.nanmean(np.vstack(ctrl), axis=0)
        ratio = path / ctrl_path
        paths.append(pd.DataFrame({"event": ev, "time_utc": t, "minute": offsets, "ratio": ratio,
                                   "excess_bps": path - ctrl_path}))
        post = (offsets >= 0) & (offsets < 5)
        ctrl_post = np.array([np.nanmean(c[post]) for c in ctrl])
        obs = np.nanmean(path[post])
        # permutation-style p-value: share of control days at least as wide as the event day
        p = (1 + np.sum(ctrl_post >= obs)) / (1 + ctrl_post.size)
        peaks.append({"event": ev, "time_utc": t, "post5_bps": obs, "control_bps": np.nanmean(ctrl_post),
                      "excess_bps": obs - np.nanmean(ctrl_post), "p_value": p})
    if not paths:
        return pd.DataFrame(), pd.DataFrame()
    allp = pd.concat(paths, ignore_index=True)
    rows = []
    for (ev, mnt), g in allp.groupby(["event", "minute"]):
        v = g["ratio"].dropna().to_numpy()
        if v.size < 2:
            continue
        b = block_bootstrap_by_group(v, np.arange(v.size), np.mean, n_boot=n_boot, seed=seed)
        rows.append({"event": ev, "minute": mnt, "ratio_mean": b.estimate, "lo": b.lower, "hi": b.upper,
                     "n_events": v.size, "excess_bps": g["excess_bps"].mean()})
    return pd.DataFrame(rows), pd.DataFrame(peaks)


def event_effects(peaks: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Per event type: mean excess spread in the first 5 minutes, bootstrap CI over events,
    one-sided t-test p-value, and Holm adjustment across event types."""
    from scipy import stats as sps

    rows = []
    for ev, g in peaks.groupby("event"):
        v = g["excess_bps"].dropna().to_numpy()
        if v.size < 3:
            continue
        b = block_bootstrap_by_group(v, np.arange(v.size), np.mean, n_boot=n_boot, seed=seed)
        t = sps.ttest_1samp(v, 0.0, alternative="greater")
        rows.append({"event": ev, "n_events": v.size, "excess_bps": b.estimate, "lo": b.lower,
                     "hi": b.upper, "ratio": float(np.mean(g["post5_bps"] / g["control_bps"])),
                     "p_value": float(t.pvalue)})
    out = pd.DataFrame(rows)
    if len(out):
        out["p_holm"] = adjust(out["p_value"].to_numpy(), "holm")
    return out
