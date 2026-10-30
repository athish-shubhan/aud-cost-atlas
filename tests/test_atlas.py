import lzma
import struct
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from atlas import calendar as C
from atlas import synthetic
from atlas.analysis import event_effects, event_study, heatmap
from atlas.data import clean, decode_bi5, minute_bars, read_histdata_ticks
from atlas.forecast import dataset
from atlas.sessions import SESSIONS, fx_week_mask, in_session, session_label

UTC_ = "UTC"


def ts(s: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(s, tz=UTC_)])


# ---------------------------------------------------------------- sessions across DST
@pytest.mark.parametrize("utc,expected", [
    # Sydney opens 08:00 local. AEDT (UTC+11) in January -> 21:00 UTC the previous day.
    ("2026-01-12 21:00", True), ("2026-01-12 20:59", False),
    # After DST ends (first Sunday of April 2026 = 5 April) Sydney is AEST (UTC+10) -> 22:00 UTC.
    ("2026-04-06 21:30", False), ("2026-04-06 22:00", True),
    # DST starts again on 4 October 2026 -> back to 21:00 UTC.
    ("2026-10-05 21:00", True),
])
def test_sydney_open_moves_with_dst(utc, expected):
    syd = next(s for s in SESSIONS if s.name == "Sydney")
    assert in_session(ts(utc), syd)[0] == expected


@pytest.mark.parametrize("utc,expected", [
    # London opens 08:00 local: GMT in winter -> 08:00 UTC; BST from 29 March 2026 -> 07:00 UTC.
    ("2026-03-27 07:30", False), ("2026-03-30 07:30", True),
    # New York opens 08:00 local: EST -> 13:00 UTC; EDT from 8 March 2026 -> 12:00 UTC.
    ("2026-03-06 12:30", False), ("2026-03-09 12:30", True),
])
def test_london_and_new_york_dst(utc, expected):
    name = "London" if utc.endswith("07:30") else "New York"
    s = next(x for x in SESSIONS if x.name == name)
    assert in_session(ts(utc), s)[0] == expected


def test_overlap_shrinks_when_us_switches_before_uk():
    # Between US DST start (8 Mar 2026) and UK BST start (29 Mar 2026) the London/NY
    # overlap begins at 12:00 UTC instead of 13:00 UTC.
    assert session_label(ts("2026-03-16 12:30")).iloc[0] == "London/New York"
    assert session_label(ts("2026-03-02 12:30")).iloc[0] == "London"


def test_fx_week_and_rollover():
    assert not fx_week_mask(ts("2026-10-31 12:00"))[0]   # Saturday
    assert fx_week_mask(ts("2026-11-01 22:30"))[0]        # Sunday 17:30 New York (EST)
    assert session_label(ts("2026-07-15 21:00")).iloc[0] == "NY rollover"  # 17:00 EDT


# ---------------------------------------------------------------- calendar
def test_calendar_loads_and_converts_rba_time():
    cal = C.load()
    rba = cal[(cal["event"] == "RBA cash rate decision") & (cal["local_date"] == "2026-02-03")]
    assert rba["time_utc"].iloc[0] == pd.Timestamp("2026-02-03 03:30", tz="UTC")  # 14:30 AEDT
    rba = cal[(cal["event"] == "RBA cash rate decision") & (cal["local_date"] == "2026-06-16")]
    assert rba["time_utc"].iloc[0] == pd.Timestamp("2026-06-16 04:30", tz="UTC")  # 14:30 AEST


def test_calendar_rejects_rows_without_official_source(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("event,country,local_date,local_time,tz,status,source_url\n"
                 "RBA,AU,2026-01-01,14:30,Australia/Sydney,released,https://example.com\n")
    with pytest.raises(ValueError, match="official source"):
        C.load(p)


# ---------------------------------------------------------------- loaders and cleaning
def test_bi5_roundtrip():
    recs = [(1000, 66012, 66010, 1.5, 2.0), (61000, 66015, 66011, 1.0, 1.0)]
    raw = lzma.compress(b"".join(struct.pack(">IIIff", *r) for r in recs))
    df = decode_bi5(raw, datetime(2026, 9, 1, 12, tzinfo=UTC), "AUDUSD")
    assert df.index[1] == pd.Timestamp("2026-09-01 12:01:01", tz="UTC")
    assert df["bid"].iloc[0] == pytest.approx(0.66010) and df["ask"].iloc[0] == pytest.approx(0.66012)


def test_histdata_is_fixed_utc_minus_5(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("20260715 120000000,0.66010,0.66012,0\n")  # July: still UTC-5, no DST
    assert read_histdata_ticks(p).index[0] == pd.Timestamp("2026-07-15 17:00", tz="UTC")


def test_clean_drops_bad_ticks_and_weekend():
    idx = pd.DatetimeIndex(["2026-10-28 10:00:00", "2026-10-28 10:00:01", "2026-10-28 10:00:02",
                            "2026-10-28 10:00:03", "2026-10-28 10:00:04", "2026-10-31 10:00:00"], tz="UTC")
    q = pd.DataFrame({"bid": [0.66, 0.66, 0.0, 0.67, 0.66, 0.66],
                      "ask": [0.66003, 0.66003, 0.66003, 0.66, 0.70, 0.66003]}, index=idx)
    out, rep = clean(q)
    assert rep.duplicates == 1 and rep.nonpositive == 1 and rep.crossed == 1
    assert rep.outliers == 1 and rep.outside_fx_week == 1 and len(out) == 1


def test_minute_bars_time_weighting():
    idx = pd.DatetimeIndex(["2026-10-28 10:00:00", "2026-10-28 10:00:50", "2026-10-28 10:01:00"], tz="UTC")
    q = pd.DataFrame({"bid": [1.0, 1.0, 1.0], "ask": [1.0001, 1.0003, 1.0001]}, index=idx)
    b = minute_bars(q)
    # 1 bp for 50s, 3 bp for 10s -> about 1.33 bp time-weighted, not the 2 bp simple mean
    assert b["spread_bps"].iloc[0] == pytest.approx((50 * 1 + 10 * 3) / 60, rel=0.01)


# ---------------------------------------------------------------- analysis on planted data
@pytest.fixture(scope="module")
def planted():
    cal = C.load()
    q = synthetic.generate("AUDUSD", "2025-06-01", "2026-03-01", cal, seed=5)
    q, _ = clean(q)
    return minute_bars(q), cal


def test_event_study_recovers_planted_rba_effect(planted):
    bars, cal = planted
    ev = C.events_between(cal, bars.index[0], bars.index[-1])
    paths, peaks = event_study(bars, ev[ev["event"] == "RBA cash rate decision"])
    eff = event_effects(peaks)
    at0 = paths[paths["minute"] == 0]["ratio_mean"].iloc[0]
    assert 2.0 < at0 < 4.0  # planted peak multiplier is 3.0
    far = paths[paths["minute"] <= -10]["ratio_mean"].mean()
    assert far == pytest.approx(1.0, abs=0.15)  # no effect long before the release
    assert eff["p_value"].iloc[0] < 0.01


def test_heatmap_finds_planted_session_pattern(planted):
    bars, _ = planted
    hm = heatmap(bars, n_boot=50)
    # London/NY overlap (late evening/night Sydney time) is planted tightest
    by_hour = hm.groupby("hour")["mean_bps"].mean()
    assert by_hour.idxmin() in {22, 23, 0, 1, 2, 3}


def test_forecast_features_do_not_peek(planted):
    bars, cal = planted
    full = dataset(bars, cal)
    cut = bars.index[len(bars) // 2]
    part = dataset(bars[bars.index < cut], cal)
    feats = ["spr_15", "spr_60", "quotes_15", "rv_15", "min_to_event"]
    common = part.index.intersection(full.index)
    common = common[common < cut - pd.Timedelta("2h")]
    pd.testing.assert_frame_equal(full.loc[common, feats], part.loc[common, feats])
    assert np.isfinite(full["y_next_hour"]).all()


def test_m1_bidask_loader_and_metrics(tmp_path):
    from atlas.data import clean_m1, minute_bars_m1, read_m1_bidask

    ts = [1782086400000 + 60000 * i for i in range(3)]  # Mon 22 Jun 2026, 00:00 UTC onwards
    rows = "timestamp,open,high,low,close,volume\n"
    (tmp_path / "b.csv").write_text(rows + "".join(f"{t},0.6600,0.6602,0.6599,0.6601,1\n" for t in ts))
    (tmp_path / "a.csv").write_text(rows + "".join(f"{t},0.6601,0.6603,0.6600,0.6602,1\n" for t in ts))
    m = read_m1_bidask([tmp_path / "b.csv"], [tmp_path / "a.csv"])
    q, rep = clean_m1(m)
    b = minute_bars_m1(q)
    assert rep.kept == 3 and len(b) == 3
    assert b["spread_bps"].iloc[0] == pytest.approx(0.0001 / 0.66005 * 1e4, rel=1e-3)
    assert (b["rv_bps"] > 0).all()
