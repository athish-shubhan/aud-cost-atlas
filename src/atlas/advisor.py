"""Execution advisor: when over the next two weeks is it cheapest to trade a given notional?

    python -m atlas.advisor --pair AUDUSD --notional 5e6 --urgency 4 --start 2026-11-02

For each candidate window (local hour slots over the next ``days``), the expected cost
of a taker-only slicing schedule is simulated by resampling historical spreads for that
weekday and hour. Windows that overlap a scheduled event (from the verified calendar)
are flagged and ranked last. Cost is quote-based half-spread only: FX quotes carry no
depth, so the advisor cannot model size impact, and says so in its output.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from . import calendar as calmod

TZ = "Australia/Sydney"


def slot_samples(bars: pd.DataFrame, tz: str = TZ) -> dict[tuple[int, int], np.ndarray]:
    loc = pd.DatetimeIndex(bars.index).tz_convert(ZoneInfo(tz))
    df = pd.DataFrame({"s": bars["spread_bps"].to_numpy(), "dow": loc.dayofweek, "hour": loc.hour})
    return {cast(tuple[int, int], k): g["s"].to_numpy() for k, g in df.groupby(["dow", "hour"])}


def advise(bars: pd.DataFrame, cal: pd.DataFrame, start: pd.Timestamp, notional: float, urgency_h: float,
           days: int = 14, slices_per_hour: int = 12, n_sims: int = 2000, seed: int = 0,
           avoid_minutes: int = 30) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    samples = slot_samples(bars)
    tz = ZoneInfo(TZ)
    start_local = start.tz_convert(tz).floor("h")
    rows = []
    hours = int(np.ceil(urgency_h))
    for h in range(days * 24):
        w0 = start_local + pd.Timedelta(hours=h)
        w1 = w0 + pd.Timedelta(hours=hours)
        slots = [((w0 + pd.Timedelta(hours=i)).dayofweek, (w0 + pd.Timedelta(hours=i)).hour) for i in range(hours)]
        if any(s not in samples or samples[s].size < 300 for s in slots):  # FX closed or too little history
            continue
        n_sl = hours * slices_per_hour
        draws = np.concatenate([rng.choice(samples[s], size=(n_sims, slices_per_hour)) for s in slots], axis=1)
        cost = 0.5 * draws.mean(axis=1)  # notional-weighted half-spread of equal slices, bps
        ev = calmod.events_between(cal, (w0 - pd.Timedelta(minutes=avoid_minutes)).tz_convert("UTC"),
                                   (w1 + pd.Timedelta(minutes=avoid_minutes)).tz_convert("UTC"))
        rows.append({"window_start_sydney": w0, "window_end_sydney": w1, "slices": n_sl,
                     "slice_notional": notional / n_sl,
                     "expected_bps": float(cost.mean()), "p05_bps": float(np.quantile(cost, 0.05)),
                     "p95_bps": float(np.quantile(cost, 0.95)),
                     "expected_cost_usd": float(cost.mean() * notional / 1e4),
                     "events_nearby": "; ".join(f"{e} ({t.tz_convert(tz):%a %d %b %H:%M})"
                                                for e, t in zip(ev["event"], ev["time_utc"], strict=True))})
    out = pd.DataFrame(rows)
    out["avoid"] = out["events_nearby"] != ""
    return out.sort_values(["avoid", "expected_bps"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", default="AUDUSD")
    ap.add_argument("--notional", type=float, default=5e6)
    ap.add_argument("--urgency", type=float, default=2.0, help="hours available to complete")
    ap.add_argument("--start", default=None, help="first day to consider (default: today)")
    ap.add_argument("--bars", type=Path, default=None, help="minute bars parquet (default: reports/<pair>_bars.parquet)")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()
    root = Path(__file__).resolve().parents[2]
    bars = pd.read_parquet(a.bars or root / "reports" / f"{a.pair.lower()}_bars.parquet")
    start = pd.Timestamp(a.start, tz="UTC") if a.start else pd.Timestamp.now(tz="UTC")
    res = advise(bars, calmod.load(), start, a.notional, a.urgency)
    cols = ["window_start_sydney", "expected_bps", "p05_bps", "p95_bps", "expected_cost_usd", "events_nearby"]
    print(f"\n{a.pair}, {a.notional:,.0f} over {a.urgency:g}h, taker-only slicing, quote-based cost only.\n")
    print("Cheapest windows (no scheduled events within 30 minutes):")
    print(res[~res["avoid"]].head(a.top)[cols].to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\nScheduled events to avoid in the period:")
    print(res[res["avoid"]].drop_duplicates("events_nearby").head(10)[["window_start_sydney", "events_nearby"]]
          .to_string(index=False))
    print("\nNote: quotes carry no depth, so size impact is not modelled. Larger notionals cost more than shown.")


if __name__ == "__main__":
    main()
