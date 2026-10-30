"""Trading sessions defined by local clock time, so daylight saving moves them correctly.

UTC hours are never hardcoded: Sydney, London and New York each shift by an hour
at their own daylight-saving dates (and Sydney shifts in the opposite season), so
the overlaps between sessions change through the year.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Session:
    name: str
    tz: str
    open: time
    close: time


SESSIONS = (
    Session("Sydney", "Australia/Sydney", time(8, 0), time(17, 0)),
    Session("Tokyo", "Asia/Tokyo", time(9, 0), time(18, 0)),
    Session("London", "Europe/London", time(8, 0), time(17, 0)),
    Session("New York", "America/New_York", time(8, 0), time(17, 0)),
)

ROLLOVER = Session("NY rollover", "America/New_York", time(16, 55), time(17, 15))
"""The daily FX value-date rollover at 17:00 New York, when liquidity thins and spreads widen."""


def _local_minutes(ts_utc: pd.DatetimeIndex, tz: str) -> tuple[np.ndarray, np.ndarray]:
    loc = ts_utc.tz_convert(ZoneInfo(tz))
    return (loc.hour * 60 + loc.minute).to_numpy(), loc.dayofweek.to_numpy()


def in_session(ts_utc: pd.DatetimeIndex, s: Session) -> np.ndarray:
    """True where the local time is within [open, close) on a local weekday."""
    mins, dow = _local_minutes(ts_utc, s.tz)
    o, c = s.open.hour * 60 + s.open.minute, s.close.hour * 60 + s.close.minute
    return (mins >= o) & (mins < c) & (dow < 5)


def session_flags(ts_utc: pd.DatetimeIndex) -> pd.DataFrame:
    df = pd.DataFrame({s.name: in_session(ts_utc, s) for s in SESSIONS}, index=ts_utc)
    df["rollover"] = in_session(ts_utc, ROLLOVER) | _rollover_any_day(ts_utc)
    df["n_open"] = df[[s.name for s in SESSIONS]].sum(axis=1)
    return df


def _rollover_any_day(ts_utc: pd.DatetimeIndex) -> np.ndarray:
    mins, _ = _local_minutes(ts_utc, ROLLOVER.tz)
    return (mins >= 16 * 60 + 55) & (mins < 17 * 60 + 15)


def session_label(ts_utc: pd.DatetimeIndex) -> pd.Series:
    """One label per timestamp: the overlap if two sessions are open, else the single session,
    else 'Off-hours'. Rollover overrides everything."""
    f = session_flags(ts_utc)
    names = [s.name for s in SESSIONS]
    lab = np.full(len(f), "Off-hours", dtype=object)
    for i, a in enumerate(names):
        lab = np.where(f[a] & (f["n_open"] == 1), a, lab)
        for b in names[i + 1:]:
            lab = np.where(f[a] & f[b], f"{a}/{b}", lab)
    lab = np.where(f["rollover"], "NY rollover", lab)
    return pd.Series(lab, index=ts_utc, name="session")


def fx_week_mask(ts_utc: pd.DatetimeIndex) -> np.ndarray:
    """True inside the FX trading week: Sunday 17:00 to Friday 17:00 New York time."""
    loc = ts_utc.tz_convert(ZoneInfo("America/New_York"))
    dow, mins = loc.dayofweek.to_numpy(), (loc.hour * 60 + loc.minute).to_numpy()
    closed = (dow == 5) | ((dow == 4) & (mins >= 17 * 60)) | ((dow == 6) & (mins < 17 * 60))
    return ~closed
