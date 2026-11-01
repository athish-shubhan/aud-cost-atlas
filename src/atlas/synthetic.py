"""Synthetic quote generator with *known, planted* effects.

This is not market data. It exists so that the whole pipeline (cleaning, sessions,
event study, heatmap, forecasting, regimes) can be run end to end and checked:
the event study should recover the planted event effects, and the heatmap should
recover the planted session pattern. Every chart made from it is labelled SYNTHETIC.

Planted structure (all multiplicative on a base spread):
- time of day: tight in the London/New York overlap, wider in the Asian session,
  widest after the New York close; a spike around the 17:00 New York rollover;
- events: a widening that starts 2 minutes before release, peaks at release and
  decays with a 4-minute half-life, sized per event type (see ``EVENT_MULT``);
- volatility clusters, plus a few injected bad ticks (crossed, zero, outliers)
  for the cleaner to catch.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.signal import lfilter

from .sessions import fx_week_mask

BASE = {"AUDUSD": (0.66, 0.45), "AUDJPY": (99.0, 0.9), "EURUSD": (1.10, 0.25)}  # (price, base spread bps)
EVENT_MULT = {"RBA cash rate decision": 3.0, "US Employment Situation (payrolls)": 2.5,
              "US CPI": 2.0, "FOMC statement": 2.0, "ABS Labour Force": 1.8, "ABS CPI": 1.8}
EVENT_PAIRS = {"RBA cash rate decision": ("AUDUSD", "AUDJPY"), "ABS Labour Force": ("AUDUSD", "AUDJPY"),
               "ABS CPI": ("AUDUSD", "AUDJPY")}


def tod_multiplier(idx: pd.DatetimeIndex) -> np.ndarray:
    ny = idx.tz_convert(ZoneInfo("America/New_York"))
    ldn = idx.tz_convert(ZoneInfo("Europe/London"))
    syd = idx.tz_convert(ZoneInfo("Australia/Sydney"))
    ny_m = ny.hour * 60 + ny.minute
    ldn_h, syd_h = ldn.hour, syd.hour
    m = np.full(len(idx), 1.6)                                    # default: thin hours
    m = np.where((syd_h >= 9) & (syd_h < 17), 1.25, m)            # Sydney day
    m = np.where((ldn_h >= 8) & (ldn_h < 17), 1.0, m)             # London
    m = np.where((ldn_h >= 13) & (ldn_h < 17) & (ny.hour >= 8), 0.8, m)  # London/NY overlap
    m = np.where((ny_m >= 16 * 60 + 55) & (ny_m < 17 * 60 + 15), 4.0, m)  # rollover
    return m


def event_multiplier(idx: pd.DatetimeIndex, events: pd.DataFrame, pair: str) -> np.ndarray:
    m = np.ones(len(idx))
    for ev, when in zip(events["event"], events["time_utc"], strict=True):
        pairs = EVENT_PAIRS.get(ev)
        if pairs is not None and pair not in pairs:
            continue
        k = EVENT_MULT.get(ev, 1.5) - 1.0
        dt = ((idx - when) / pd.Timedelta(minutes=1)).to_numpy()
        pre = (dt >= -2) & (dt < 0)
        post = (dt >= 0) & (dt < 60)
        bump = np.zeros(len(idx))
        bump[pre] = 0.5 * k
        bump[post] = k * 0.5 ** (dt[post] / 4.0)
        m += bump
    return m


def minute_spreads(pair: str, start: str, end: str, events: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, end, freq="1min", tz="UTC", inclusive="left")
    idx = idx[fx_week_mask(idx)]
    price, base = BASE[pair]
    # slowly varying volatility regime (AR(1) in log space) drives both spread and quote rate
    n = len(idx)
    e = rng.standard_normal(n)
    lv = lfilter([0.02], [1.0, -0.999], e)
    vol = np.exp(lv)
    spr = base * tod_multiplier(idx) * event_multiplier(idx, events, pair) * vol ** 0.5
    spr *= np.exp(0.15 * rng.standard_normal(n))
    sigma = 0.6 * vol * tod_multiplier(idx) ** -0.3 * event_multiplier(idx, events, pair)
    ret = sigma * rng.standard_t(4, n) / np.sqrt(2)
    mid = price * np.exp(np.cumsum(ret) / 1e4)
    rate = np.clip(12 / tod_multiplier(idx) * vol, 1, 80)
    return pd.DataFrame({"mid": mid, "spread_bps": spr, "rate": rate}, index=idx)


def ticks_from_minutes(m: pd.DataFrame, seed: int = 0, bad_tick_rate: float = 2e-5) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    n = rng.poisson(m["rate"].to_numpy())
    rep = np.repeat(np.arange(len(m)), n)
    t0 = pd.DatetimeIndex(m.index).tz_convert(None).to_numpy().astype("datetime64[ns]")[rep]
    off = rng.random(rep.size) * 60e9
    ts = t0 + off.astype("timedelta64[ns]")
    order = np.argsort(ts, kind="stable")
    ts, rep = ts[order], rep[order]
    mid = m["mid"].to_numpy()[rep] * np.exp(rng.standard_normal(rep.size) * 0.3 / 1e4)
    spr = m["spread_bps"].to_numpy()[rep] * np.exp(0.1 * rng.standard_normal(rep.size))
    half = mid * spr / 2e4
    q = pd.DataFrame({"bid": mid - half, "ask": mid + half}, index=pd.DatetimeIndex(ts, tz="UTC", name="ts"))
    bad = rng.random(len(q)) < bad_tick_rate
    kind = rng.integers(0, 3, bad.sum())
    b, a = q["bid"].to_numpy().copy(), q["ask"].to_numpy().copy()
    bi = np.flatnonzero(bad)
    b[bi[kind == 0]], a[bi[kind == 0]] = a[bi[kind == 0]], b[bi[kind == 0]]   # crossed
    b[bi[kind == 1]] = 0.0                                                    # zero bid
    a[bi[kind == 2]] *= 1.05                                                  # 500 bps outlier
    q["bid"], q["ask"] = b, a
    return q


def generate(pair: str, start: str, end: str, events: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    return ticks_from_minutes(minute_spreads(pair, start, end, events, seed), seed)
