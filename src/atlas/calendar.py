"""Versioned event calendar. Every row needs an official source URL, or it is rejected."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

DEFAULT = Path(__file__).resolve().parents[2] / "data" / "calendar" / "events.csv"
REQUIRED = ["event", "country", "local_date", "local_time", "tz", "status", "source_url"]
OFFICIAL_DOMAINS = ("rba.gov.au", "abs.gov.au", "federalreserve.gov", "bls.gov")


def load(path: str | Path = DEFAULT) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    missing = [c for c in REQUIRED if c not in df]
    if missing:
        raise ValueError(f"calendar is missing columns {missing}")
    bad = df["source_url"].isna() | ~df["source_url"].fillna("").str.contains("|".join(OFFICIAL_DOMAINS))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} calendar rows lack an official source URL:\n{df[bad].head()}")
    df["time_utc"] = [
        pd.Timestamp(f"{d} {t}").tz_localize(ZoneInfo(tz)).tz_convert("UTC")
        for d, t, tz in zip(df["local_date"], df["local_time"], df["tz"], strict=True)
    ]
    return df.sort_values("time_utc").reset_index(drop=True)


def events_between(cal: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
                   kinds: list[str] | None = None) -> pd.DataFrame:
    m = (cal["time_utc"] >= start) & (cal["time_utc"] < end)
    if kinds:
        m &= cal["event"].isin(kinds)
    return cal[m]


def minutes_to_next(ts_utc: pd.DatetimeIndex, cal: pd.DataFrame, cap: float = 1440.0) -> pd.Series:
    """Minutes until the next calendar event (capped), using only the published schedule."""
    ev = pd.DatetimeIndex(cal["time_utc"]).sort_values().tz_convert(None).to_numpy().astype("datetime64[ns]")
    t = ts_utc.tz_convert(None).to_numpy().astype("datetime64[ns]")
    i = ev.searchsorted(t, side="left")
    nxt = ev[i.clip(max=len(ev) - 1)]
    mins = (nxt - t) / np.timedelta64(1, "m")
    mins[i >= len(ev)] = cap
    return pd.Series(mins.clip(0, cap), index=ts_utc)
