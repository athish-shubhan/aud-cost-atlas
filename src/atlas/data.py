"""Quote loaders, cleaning, and one-minute cost metrics. All timestamps are UTC internally.

Supported sources:
- Dukascopy one-minute bid and ask bars (separate BID and ASK OHLC files), which give two
  quote snapshots per minute (open and close) and are the source used for the published results,
- Dukascopy raw hourly ``.bi5`` tick files (what :mod:`atlas.download` fetches),
- Dukascopy web-export CSVs (separate BID and ASK tick files),
- HistData.com "Generic ASCII" tick files (timestamps in EST *without* daylight saving, UTC-5).

FX is over the counter. These are indicative top-of-book quotes from one venue,
with no trade prints, so everything here measures quote-based cost for small to
mid sizes. Nothing here can estimate market impact for large orders.
"""

from __future__ import annotations

import lzma
import struct
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .sessions import fx_week_mask, session_flags

POINT = {"AUDUSD": 1e-5, "EURUSD": 1e-5, "AUDJPY": 1e-3, "USDJPY": 1e-3}


def decode_bi5(raw: bytes, hour_start_utc: datetime, pair: str) -> pd.DataFrame:
    """Decode one Dukascopy hourly tick file: 20-byte big-endian records
    (ms offset, ask points, bid points, ask volume, bid volume)."""
    if not raw:
        return pd.DataFrame(columns=["bid", "ask"], index=pd.DatetimeIndex([], tz="UTC"))
    data = lzma.decompress(raw)
    n = len(data) // 20
    rec = np.array(list(struct.iter_unpack(">IIIff", data[: n * 20])), dtype=float)
    pt = POINT.get(pair.upper(), 1e-5)
    ts = pd.Timestamp(hour_start_utc).tz_convert("UTC") + pd.to_timedelta(rec[:, 0], unit="ms")
    return pd.DataFrame({"bid": rec[:, 2] * pt, "ask": rec[:, 1] * pt}, index=pd.DatetimeIndex(ts, name="ts"))


def read_dukascopy_export(bid_csv: str | Path, ask_csv: str | Path) -> pd.DataFrame:
    """Merge separate BID and ASK tick exports (first column is the UTC timestamp)."""
    out = []
    for p, name in ((bid_csv, "bid"), (ask_csv, "ask")):
        df = pd.read_csv(p)
        ts = pd.to_datetime(df.iloc[:, 0], utc=True, format="mixed", dayfirst=True)
        px = df["Close"] if "Close" in df else df.iloc[:, 1]
        out.append(pd.Series(px.to_numpy(float), index=ts, name=name).sort_index())
    bid, ask = out
    m = pd.merge_asof(bid.to_frame().reset_index(names="ts"), ask.to_frame().reset_index(names="ts"),
                      on="ts", direction="backward")
    m = pd.concat([m, pd.merge_asof(ask.to_frame().reset_index(names="ts"),
                                    bid.to_frame().reset_index(names="ts"), on="ts", direction="backward")])
    return m.dropna().drop_duplicates().sort_values("ts").set_index("ts")[["bid", "ask"]]


HISTDATA_TZ = timezone(timedelta(hours=-5))


def read_histdata_ticks(path: str | Path) -> pd.DataFrame:
    """HistData Generic ASCII ticks: ``YYYYMMDD HHMMSSfff,bid,ask,volume`` in fixed UTC-5."""
    p = Path(path)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as z:
            name = next(n for n in z.namelist() if n.lower().endswith((".csv", ".txt")))
            with z.open(name) as f:
                df = pd.read_csv(f, header=None, names=["dt", "bid", "ask", "vol"])
    else:
        df = pd.read_csv(p, header=None, names=["dt", "bid", "ask", "vol"])
    ts = pd.to_datetime(df["dt"], format="%Y%m%d %H%M%S%f").dt.tz_localize(HISTDATA_TZ).dt.tz_convert("UTC")
    return pd.DataFrame({"bid": df["bid"].to_numpy(float), "ask": df["ask"].to_numpy(float)},
                        index=pd.DatetimeIndex(ts, name="ts"))


@dataclass(frozen=True)
class CleanReport:
    raw: int
    nonpositive: int
    crossed: int
    duplicates: int
    outliers: int
    outside_fx_week: int
    kept: int
    gaps_over_5min: int

    def as_dict(self) -> dict[str, int]:
        return self.__dict__.copy()


def clean(q: pd.DataFrame, outlier_mult: float = 50.0) -> tuple[pd.DataFrame, CleanReport]:
    """Drop bad quotes and flag what remains. Returns (clean quotes, report)."""
    q = q.sort_index()
    raw = len(q)
    pos = (q["bid"] > 0) & (q["ask"] > 0)
    crossed = q["ask"] < q["bid"]
    dup = (q["bid"].diff() == 0) & (q["ask"].diff() == 0)  # stale repeats carry no new information
    mid = 0.5 * (q["bid"] + q["ask"])
    spr = (q["ask"] - q["bid"]) / mid * 1e4
    med = spr[pos & ~crossed].median()
    outlier = (spr > outlier_mult * med) & pos & ~crossed
    week = pd.Series(fx_week_mask(pd.DatetimeIndex(q.index)), index=q.index)
    keep = pos & ~crossed & ~dup & ~outlier & week
    out = q[keep].copy()
    gaps = out.index.to_series().diff()
    in_week_gap = (gaps > pd.Timedelta("5min")) & pd.Series(fx_week_mask(pd.DatetimeIndex(out.index - gaps.fillna(pd.Timedelta(0)))), index=out.index)
    out["gap_before"] = in_week_gap.to_numpy()
    return out, CleanReport(raw, int((~pos).sum()), int(crossed.sum()), int(dup.sum()), int(outlier.sum()),
                            int((~week).sum()), int(keep.sum()), int(in_week_gap.sum()))


def minute_bars(q: pd.DataFrame) -> pd.DataFrame:
    """One-minute cost metrics from clean quotes.

    spread_bps is time-weighted within the minute (each quote counts for as long as it was live),
    so a burst of quotes does not dominate a quiet minute.
    """
    mid = 0.5 * (q["bid"] + q["ask"])
    spr = (q["ask"] - q["bid"]) / mid * 1e4
    secs = ((q.index - q.index[0]) / pd.Timedelta(seconds=1)).to_numpy()
    dur = np.diff(secs, append=secs[-1])
    dur = np.clip(dur, 0, 60.0)
    lr = np.r_[0.0, np.diff(np.log(mid.to_numpy()))] * 1e4
    df = pd.DataFrame({"w": dur, "ws": dur * spr.to_numpy(), "spr": spr.to_numpy(),
                       "lr2": lr**2, "mid": mid.to_numpy()}, index=q.index)
    g = df.resample("1min", label="left", closed="left")
    bars = pd.DataFrame({
        "spread_bps": g["ws"].sum() / g["w"].sum().replace(0, np.nan),
        "spread_max_bps": g["spr"].max(),
        "quotes": g["spr"].count(),
        "rv_bps": np.sqrt(g["lr2"].sum()),
        "mid": g["mid"].last(),
    })
    bars = bars[bars["quotes"] > 0]
    bars["spread_bps"] = bars["spread_bps"].fillna(bars["spread_max_bps"])
    bars["ret_bps"] = np.log(bars["mid"]).diff() * 1e4
    sig = bars["ret_bps"].rolling(240, min_periods=60).std().shift(1)
    bars["jump"] = (bars["ret_bps"].abs() > 4 * sig).astype(int)
    return bars.join(session_flags(pd.DatetimeIndex(bars.index)))


def typical_spread(bars: pd.DataFrame, days: int = 20) -> pd.Series:
    """Trailing median spread for the same 15-minute slot of the day, over the last ``days``
    days, using only earlier days. Used to define a spread blowout (> 2x typical)."""
    bix = pd.DatetimeIndex(bars.index)
    slot = (bix.hour * 4 + bix.minute // 15).to_numpy()
    day = bix.normalize()
    daily = bars.groupby([day, slot])["spread_bps"].median().unstack()
    trailing = daily.rolling(days, min_periods=5).median().shift(1)
    vals = trailing.stack()
    idx = pd.MultiIndex.from_arrays([day, slot])
    return pd.Series(vals.reindex(idx).to_numpy(), index=bars.index)


def read_m1_bidask(bid_paths: list[str | Path], ask_paths: list[str | Path]) -> pd.DataFrame:
    """Merge one-minute BID and ASK OHLC bars (columns timestamp[ms],open,high,low,close,volume)."""
    rd = lambda ps: pd.concat([pd.read_csv(p) for p in sorted(map(str, ps))], ignore_index=True)
    b, a = rd(bid_paths), rd(ask_paths)
    m = b.merge(a, on="timestamp", suffixes=("_b", "_a")).drop_duplicates("timestamp").sort_values("timestamp")
    idx = pd.DatetimeIndex(pd.to_datetime(m["timestamp"], unit="ms", utc=True), name="ts")
    cols = {f"{k}_{s}": f"{k}_{s}" for k in ("open", "high", "low", "close") for s in ("b", "a")}
    out = m[list(cols)].set_axis(idx)
    out["volume"] = (m["volume_b"] + m["volume_a"]).to_numpy() / 2
    return out


def clean_m1(m: pd.DataFrame, outlier_mult: float = 50.0) -> tuple[pd.DataFrame, CleanReport]:
    """Same rules as :func:`clean`, applied to one-minute bid/ask bars (both snapshots)."""
    raw = len(m)
    pos = (m[["open_b", "close_b", "open_a", "close_a"]] > 0).all(axis=1)
    crossed = (m["close_a"] < m["close_b"]) | (m["open_a"] < m["open_b"])
    dup = pd.Series(False, index=m.index)
    mid = 0.5 * (m["close_b"] + m["close_a"])
    spr = (m["close_a"] - m["close_b"]) / mid * 1e4
    med = spr[pos & ~crossed].median()
    outlier = (spr > outlier_mult * med) & pos & ~crossed
    week = pd.Series(fx_week_mask(pd.DatetimeIndex(m.index)), index=m.index)
    keep = pos & ~crossed & ~outlier & week
    out = m[keep].copy()
    gaps = out.index.to_series().diff()
    prev = pd.DatetimeIndex(out.index - gaps.fillna(pd.Timedelta(0)))
    in_week_gap = (gaps > pd.Timedelta("5min")) & pd.Series(fx_week_mask(prev), index=out.index)
    out["gap_before"] = in_week_gap.to_numpy()
    return out, CleanReport(raw, int((~pos).sum()), int(crossed.sum()), int(dup.sum()), int(outlier.sum()),
                            int((~week).sum()), int(keep.sum()), int(in_week_gap.sum()))


def minute_bars_m1(m: pd.DataFrame) -> pd.DataFrame:
    """Cost metrics from one-minute bid/ask bars, in the same shape as :func:`minute_bars`.

    spread_bps is the mean of the open and close quoted spreads (the two snapshots the bar
    carries); spread_max_bps is the wider of the two. Within-minute blowouts between the
    snapshots are invisible at this resolution, so blowout frequencies are lower bounds.
    rv_bps uses the Parkinson high-low range of the mid, and "quotes" is the bar's volume.
    """
    mid_o = 0.5 * (m["open_b"] + m["open_a"])
    mid_c = 0.5 * (m["close_b"] + m["close_a"])
    s_o = (m["open_a"] - m["open_b"]) / mid_o * 1e4
    s_c = (m["close_a"] - m["close_b"]) / mid_c * 1e4
    hi = 0.5 * (m["high_b"] + m["high_a"])
    lo = 0.5 * (m["low_b"] + m["low_a"])
    bars = pd.DataFrame({
        "spread_bps": 0.5 * (s_o + s_c),
        "spread_max_bps": np.maximum(s_o, s_c),
        "quotes": m["volume"],
        "rv_bps": np.log(hi / lo) * 1e4 / np.sqrt(4 * np.log(2)),
        "mid": mid_c,
    }, index=m.index)
    bars["ret_bps"] = np.log(bars["mid"]).diff() * 1e4
    sig = bars["ret_bps"].rolling(240, min_periods=60).std().shift(1)
    bars["jump"] = (bars["ret_bps"].abs() > 4 * sig).astype(int)
    return bars.join(session_flags(pd.DatetimeIndex(bars.index)))
