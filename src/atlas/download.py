"""Polite downloader for Dukascopy hourly tick files (run on your own machine).

    python -m atlas.download --m1                     # one-minute bid/ask bars used for the published results
    python -m atlas.download --pair AUDUSD --start 2024-10-01 --end 2026-10-01 --out data/raw   # raw ticks

Dukascopy throttles fast clients, so this fetches one file at a time with a pause
and exponential backoff on errors, and skips files already on disk, so it can be
stopped and resumed. Check Dukascopy's terms of use before downloading.
URL pattern: /datafeed/<PAIR>/<YYYY>/<MM-1>/<DD>/<HH>h_ticks.bi5 (months are zero-based).
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd

from .data import decode_bi5
from .sessions import fx_week_mask

M1_URL = "https://github.com/tradesensedev/dukascopy-data/releases/download/m1-{pair}/{pair}_{year}_{side}.csv.gz"
"""Public mirror of Dukascopy one-minute bid and ask bars (UTC), one file per pair, year and side."""
URL = "https://datafeed.dukascopy.com/datafeed/{pair}/{y:04d}/{m:02d}/{d:02d}/{h:02d}h_ticks.bi5"


def hours(start: datetime, end: datetime) -> list[datetime]:
    idx = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
    idx = idx[fx_week_mask(idx) | fx_week_mask(idx + pd.Timedelta(minutes=59))]
    return [t.to_pydatetime() for t in idx]


def fetch(pair: str, start: datetime, end: datetime, out: Path, pause: float = 0.6, max_tries: int = 6) -> None:
    import requests

    out.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers["User-Agent"] = "aud-cost-atlas research downloader"
    for t in hours(start, end):
        f = out / pair / f"{t:%Y/%m/%d/%H}h_ticks.bi5"
        if f.exists():
            continue
        url = URL.format(pair=pair, y=t.year, m=t.month - 1, d=t.day, h=t.hour)
        for k in range(max_tries):
            try:
                r = s.get(url, timeout=30)
                if r.status_code == 200:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_bytes(r.content)
                    break
                if r.status_code == 404:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_bytes(b"")  # no ticks that hour
                    break
            except requests.RequestException:
                pass
            time.sleep(pause * 2 ** (k + 1))
        time.sleep(pause)


def fetch_m1(pairs: tuple[str, ...], years: tuple[int, ...], out: Path) -> None:
    import requests

    (out / "m1").mkdir(parents=True, exist_ok=True)
    for pair in pairs:
        for year in years:
            for side in ("bid", "ask"):
                f = out / "m1" / f"{pair}_{year}_{side}.csv.gz"
                if f.exists() and f.stat().st_size > 0:
                    continue
                r = requests.get(M1_URL.format(pair=pair, year=year, side=side), timeout=120)
                r.raise_for_status()
                f.write_bytes(r.content)
                print(f"{f.name}: {len(r.content) / 1e6:.1f} MB")


def load_bi5_dir(root: Path, pair: str) -> pd.DataFrame:
    frames = []
    for f in sorted((root / pair).rglob("*h_ticks.bi5")):
        y, m, d = (int(x) for x in f.parts[-4:-1])
        h = int(f.name[:2])
        frames.append(decode_bi5(f.read_bytes(), datetime(y, m, d, h, tzinfo=UTC), pair))
    return pd.concat(frames).sort_index() if frames else pd.DataFrame(columns=["bid", "ask"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--m1", action="store_true", help="fetch the one-minute bid/ask bars used in the README")
    ap.add_argument("--pair", default="AUDUSD")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    a = ap.parse_args()
    if a.m1:
        fetch_m1(("AUDUSD", "AUDJPY", "EURUSD"), (2019, 2024, 2025, 2026), a.out)
        return
    if a.start is None:
        ap.error("--start is required for tick downloads")
    start = datetime.fromisoformat(a.start).replace(tzinfo=UTC)
    end = datetime.fromisoformat(a.end).replace(tzinfo=UTC) if a.end else datetime.now(UTC) - timedelta(days=1)
    fetch(a.pair, start, end, a.out)


if __name__ == "__main__":
    main()
