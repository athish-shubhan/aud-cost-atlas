"""AUD Cost Atlas study: every number and chart in the README comes from this script.

    python studies/run_atlas.py                       # real Dukascopy one-minute bid/ask bars (data/raw/m1)
    python studies/run_atlas.py --source synthetic    # pipeline validation on planted effects
    python studies/run_atlas.py --quick               # checked-in 3-month AUDUSD sample, for CI
    python studies/run_atlas.py --source histdata     # HistData tick zips in data/raw
    python studies/run_atlas.py --source bi5          # Dukascopy hourly tick files from atlas-download

Fetch the real one-minute data with ``python -m atlas.download --m1`` (about 100 MB).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm

from atlas import analysis, forecast, regimes, synthetic
from atlas import calendar as calmod
from atlas.advisor import advise
from atlas.data import clean, clean_m1, minute_bars, minute_bars_m1, read_histdata_ticks, read_m1_bidask
from atlas.download import load_bi5_dir
from atlas.sessions import session_label
from skeptic.multiple import adjust

ROOT = Path(__file__).resolve().parents[1]
BLUE, LIGHT, GREY, RED, GREEN, BROWN = "#1f5fa8", "#9fbbe0", "#8a8a8a", "#b23a3a", "#2e7d4f", "#8a5a00"
PERIOD = ("2024-01-01", "2026-10-27")
PAIRS = ("AUDUSD", "AUDJPY", "EURUSD")
SYN = {"AUDUSD": ("2024-10-01", "2026-10-01"), "AUDJPY": ("2026-04-01", "2026-10-01"),
       "EURUSD": ("2026-04-01", "2026-10-01")}
EVENT_TYPES = ["RBA cash rate decision", "ABS CPI", "ABS Labour Force", "US Employment Situation (payrolls)",
               "US CPI", "FOMC statement"]
EVENT_COLORS = dict(zip(EVENT_TYPES, [RED, "#e08a3c", "#a05aa0", BLUE, GREEN, BROWN], strict=True))
SHORT = {"RBA cash rate decision": "RBA decision", "ABS CPI": "ABS CPI", "ABS Labour Force": "ABS labour force",
         "US Employment Situation (payrolls)": "US payrolls", "US CPI": "US CPI", "FOMC statement": "FOMC statement"}


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", lw=0.8)
    ax.set_axisbelow(True)


def stamp(fig: plt.Figure, text: str, color: str = GREY) -> None:
    """Source line below the axes (bbox_inches="tight" keeps it in the saved image)."""
    fig.text(0.99, -0.04, text, ha="right", va="top", fontsize=7, color=color)


def load_bars(pair: str, source: str, raw: Path, cal: pd.DataFrame, quick: bool) -> tuple[pd.DataFrame, dict]:
    if source == "m1":
        d = ROOT / "sample" if quick else raw / "m1"
        bids, asks = sorted(d.glob(f"{pair}_*bid*.csv.gz")), sorted(d.glob(f"{pair}_*ask*.csv.gz"))
        if not bids:
            return pd.DataFrame(), {}
        m = read_m1_bidask(bids, asks)
        if not quick:
            m = m.loc[PERIOD[0]:PERIOD[1]]
        q, rep = clean_m1(m)
        return minute_bars_m1(q), rep.as_dict()
    if source == "synthetic":
        ticks = synthetic.generate(pair, *SYN[pair], cal, seed={"AUDUSD": 1, "AUDJPY": 2, "EURUSD": 3}[pair])
    elif source == "histdata":
        files = sorted(raw.glob(f"*{pair}*T*.zip")) + sorted(raw.glob(f"*{pair}*.csv"))
        ticks = pd.concat([read_histdata_ticks(f) for f in files]).sort_index() if files else pd.DataFrame()
    elif source == "bi5":
        ticks = load_bi5_dir(raw, pair)
    else:
        raise SystemExit(f"unknown source {source}")
    if ticks.empty:
        return pd.DataFrame(), {}
    q, rep = clean(ticks)
    return minute_bars(q), rep.as_dict()


def heatmap_chart(hms: dict[str, pd.DataFrame], out: Path, note: str) -> None:
    fig, axes = plt.subplots(len(hms), 1, figsize=(10, 2.7 * len(hms)), squeeze=False)
    for ax, (pair, hm) in zip(axes[:, 0], hms.items(), strict=True):
        grid = hm[hm["days"] >= 10].pivot(index="weekday", columns="hour", values="mean_bps")
        grid = grid.reindex([d for d in analysis.WEEKDAYS if d in grid.index])
        v = grid.to_numpy()
        im = ax.imshow(v, aspect="auto", cmap="Blues", norm=LogNorm(vmin=np.nanmin(v), vmax=np.nanmax(v)))
        ax.set_yticks(range(len(grid)), grid.index)
        ax.set_xticks(range(0, 24, 2), [f"{h:02d}" for h in range(0, 24, 2)])
        ax.set_title(f"{pair}: mean quoted spread by Sydney hour and weekday (log colour scale)", fontsize=9.5)
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        ticks = [t for t in (0.5, 0.7, 1, 1.5, 2, 3, 5, 7, 10, 15) if np.nanmin(v) <= t <= np.nanmax(v)]
        cb.set_ticks(ticks, labels=[f"{t:g}" for t in ticks])
        cb.minorticks_off()
        cb.set_label("bps")
    axes[-1, 0].set_xlabel("hour of day, Sydney time")
    stamp(fig, note, RED if "SYNTHETIC" in note else GREY)
    fig.tight_layout()
    fig.savefig(out / "heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def event_chart(paths_all: dict[str, pd.DataFrame], pairs: list[str], out: Path, note: str) -> None:
    fig, axes = plt.subplots(1, len(pairs), figsize=(5.4 * len(pairs), 3.6), sharey=True, squeeze=False)
    for ax, pair in zip(axes[0], pairs, strict=True):
        paths = paths_all[pair]
        for t in EVENT_TYPES:
            d = paths[paths["event"] == t]
            if d.empty:
                continue
            ax.plot(d["minute"], d["ratio_mean"], color=EVENT_COLORS[t], lw=1.3,
                    label=f"{SHORT[t]} (n={int(d['n_events'].max())})")
            ax.fill_between(d["minute"], d["lo"], d["hi"], color=EVENT_COLORS[t], alpha=0.12, lw=0)
        ax.axhline(1, color=GREY, lw=0.8)
        ax.axvline(0, color=GREY, lw=0.8, ls=":")
        ax.set_xlabel("minutes from release")
        ax.set_title(pair, fontsize=10)
        ax.legend(frameon=False, fontsize=7.5)
        style(ax)
    axes[0, 0].set_ylabel("spread / same clock time on normal days")
    fig.suptitle("Quoted spread around scheduled releases (95% bootstrap bands over events)", fontsize=10)
    stamp(fig, note, RED if "SYNTHETIC" in note else GREY)
    fig.savefig(out / "event_study.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def forecast_section(bars: pd.DataFrame, cal: pd.DataFrame, out: Path, note: str) -> dict:
    ds = forecast.dataset(bars, cal)
    preds, met = forecast.walk_forward(ds)
    met.to_csv(out / "forecast_metrics.csv", index=False)
    summ: dict = {}
    for near, g in met.groupby("near_event"):
        w = g["n"] / g["n"].sum()
        summ["near_event" if near else "other"] = {
            k: float(np.sum(w * g[k])) for k in ["base_pinball", "xgb_pinball", "base_coverage80", "xgb_coverage80",
                                                 "base_brier", "logit_brier", "xgb_brier"]} | {"n": int(g["n"].sum())}
    summ["blowout_rate"] = float(preds["y_blowout_15"].mean())
    summ["months_tested"] = int(met["month"].nunique())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    bins = np.linspace(0, 1, 11)
    for col, c, lab in (("base_p", GREY, "seasonal baseline"), ("logit_p", LIGHT, "logistic"), ("xgb_p", BLUE, "XGBoost")):
        k = np.clip(np.digitize(preds[col], bins) - 1, 0, 9)
        cdf = pd.DataFrame({"p": preds[col], "y": preds["y_blowout_15"], "k": k}).groupby("k").agg(
            p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        cdf = cdf[cdf.n >= 30]
        axes[0].plot(cdf.p, cdf.y, "o-", ms=3, color=c, label=lab)
    axes[0].plot([0, 1], [0, 1], ":", color=GREY, lw=0.8)
    axes[0].set_xlabel("predicted probability")
    axes[0].set_ylabel("observed frequency")
    axes[0].set_title("Spread blowout (> 2x typical) in the next 15 min", fontsize=9.5)
    axes[0].legend(frameon=False, fontsize=8)
    style(axes[0])
    labels = [k for k in ("other", "near_event") if k in summ]
    x = np.arange(len(labels))
    for i, (m, c) in enumerate((("base", GREY), ("xgb", BLUE))):
        axes[1].bar(x + (i - 0.5) * 0.35, [summ[k][f"{m}_coverage80"] * 100 for k in labels],
                    width=0.35, color=c, label="seasonal baseline" if m == "base" else "XGBoost quantile")
    axes[1].axhline(80, color=RED, lw=0.8, ls="--")
    axes[1].set_xticks(x, ["normal times", "within 60 min of a release"][: len(labels)])
    axes[1].set_ylabel("80% interval coverage (%)")
    axes[1].set_ylim(0, 115)
    axes[1].set_title("Next-hour average spread: interval coverage (target 80%)", fontsize=9.5)
    axes[1].legend(frameon=False, fontsize=8, loc="upper center", ncol=2)
    style(axes[1])
    stamp(fig, note, RED if "SYNTHETIC" in note else GREY)
    fig.savefig(out / "forecast_calibration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return summ


def case_study(raw: Path, out: Path) -> dict:
    """3 January 2019: the AUD flash move in thin liquidity just after the Asian open."""
    res: dict = {}
    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)
    for pair, c in (("AUDJPY", RED), ("AUDUSD", BLUE)):
        bids = sorted((raw / "m1").glob(f"{pair}_2019_bid*.csv.gz"))
        if not bids:
            continue
        m = read_m1_bidask(bids, sorted((raw / "m1").glob(f"{pair}_2019_ask*.csv.gz")))
        b = minute_bars_m1(m)
        win = b.loc["2019-01-02 20:00":"2019-01-03 02:00"]
        if win.empty:
            continue
        pre = win.loc[:"2019-01-02 21:00", "mid"].iloc[-1]
        low_t = win["mid"].idxmin()
        ctrl = []
        for d in range(7, 12):  # same clock times on the following week's weekdays
            day = pd.Timestamp(f"2019-01-{d:02d} 20:00", tz="UTC")
            seg = b.loc[day - pd.Timedelta(days=1) + pd.Timedelta(hours=0):day + pd.Timedelta(hours=6)]
            ctrl.append(seg.loc[seg.index.time == low_t.time(), "spread_bps"])
        ctrl_spread = float(pd.concat(ctrl).median()) if ctrl else float("nan")
        res[pair] = {"pre_move_mid": float(pre), "low_time_utc": str(low_t), "low_mid": float(win["mid"].min()),
                     "drop_pct": float((win["mid"].min() / pre - 1) * 100),
                     "max_spread_bps": float(win["spread_max_bps"].max()),
                     "max_spread_time_utc": str(win["spread_max_bps"].idxmax()),
                     "normal_spread_same_minute_bps": ctrl_spread,
                     "recovered_by_0200_pct_of_drop": float((win["mid"].iloc[-1] - win["mid"].min())
                                                            / (pre - win["mid"].min()) * 100)}
        rel = (win["mid"] / pre - 1) * 100
        axes[0].plot(win.index, rel, color=c, lw=1, label=pair)
        axes[1].plot(win.index, win["spread_max_bps"], color=c, lw=1, label=pair)
    axes[0].set_ylabel("mid vs 21:00 UTC (%)")
    axes[1].set_ylabel("quoted spread (bps)")
    axes[1].set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title("3 January 2019 flash move: price and quoted spread (Dukascopy, one-minute)", fontsize=10)
    for ax in axes:
        style(ax)
    axes[1].set_xlabel("time (UTC), 2-3 January 2019")
    fig.autofmt_xdate()
    fig.savefig(out / "case_study_2019.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="m1", choices=["m1", "synthetic", "histdata", "bi5"])
    ap.add_argument("--raw", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    syn = a.source == "synthetic"
    out = ROOT / "reports" / ("quick" if a.quick else ("synthetic" if syn else ""))
    out.mkdir(parents=True, exist_ok=True)
    note = ("SYNTHETIC DATA: planted effects, not market findings" if syn
            else "Source: Dukascopy one-minute bid/ask quotes. Quote-based top-of-book cost only.")
    cal = calmod.load()
    res: dict = {"source": a.source, "synthetic": syn, "quick": a.quick, "pairs": {}}
    bars_by: dict[str, pd.DataFrame] = {}
    for pair in (("AUDUSD",) if a.quick else PAIRS if not syn else tuple(SYN)):
        bars, rep = load_bars(pair, a.source, a.raw, cal, a.quick)
        if bars.empty:
            print(f"{pair}: no data, skipped")
            continue
        if not a.quick:
            bars.to_parquet(out / f"{pair.lower()}_bars.parquet")
        bars_by[pair] = bars
        res["pairs"][pair] = {"clean": rep, "start": str(bars.index[0]), "end": str(bars.index[-1]),
                              "minutes": len(bars), "median_spread_bps": float(bars["spread_bps"].median()),
                              "mean_spread_bps": float(bars["spread_bps"].mean())}
        print(pair, rep, flush=True)
    main_pair = "AUDUSD"
    bars = bars_by[main_pair]

    # heatmaps and sessions
    hms = {p: analysis.heatmap(b) for p, b in bars_by.items() if p != "EURUSD"}
    heatmap_chart(hms, out, note)
    res["heatmap"] = {}
    for p, hm in hms.items():
        cheap, dear = analysis.rank_windows(hm)
        res["heatmap"][p] = {"cheapest": cheap.to_dict(orient="records"), "dearest": dear.to_dict(orient="records")}
    res["sessions"] = {}
    for p, b in bars_by.items():
        lab = session_label(pd.DatetimeIndex(b.index)).to_numpy()
        sess = b.groupby(lab)["spread_bps"].agg(["mean", "median", "count"])
        res["sessions"][p] = sess.sort_values("mean").reset_index(names="session").to_dict(orient="records")

    # event study, Holm across pairs x event types
    effects, paths_all = [], {}
    for pair, b in bars_by.items():
        evs = calmod.events_between(cal, b.index[0], b.index[-1])
        evs = evs[evs["status"] == "released"]
        paths, peaks = analysis.event_study(b, evs)
        if paths.empty:
            continue
        eff = analysis.event_effects(peaks).assign(pair=pair)
        effects.append(eff)
        paths_all[pair] = paths
        peaks.assign(pair=pair).to_csv(out / f"event_peaks_{pair.lower()}.csv", index=False)
    eff = pd.concat(effects, ignore_index=True)
    eff["p_holm_all"] = adjust(eff["p_value"].to_numpy(), "holm")
    res["event_effects"] = eff.to_dict(orient="records")
    res["event_peaks_at_release"] = {
        p: {t: {"ratio": float(d.loc[d["minute"] == 0, "ratio_mean"].iloc[0]),
                "lo": float(d.loc[d["minute"] == 0, "lo"].iloc[0]), "hi": float(d.loc[d["minute"] == 0, "hi"].iloc[0]),
                "minutes_above_1_5x": int(((d["minute"] >= 0) & (d["lo"] > 1.5)).sum())}
            for t in EVENT_TYPES if not (d := paths[paths["event"] == t]).empty}
        for p, paths in paths_all.items()}
    event_chart(paths_all, [p for p in ("AUDUSD", "AUDJPY") if p in paths_all], out, note)
    if syn:
        res["planted_recovery"] = [
            {"event": t, "planted_peak_multiplier": synthetic.EVENT_MULT[t],
             "recovered_ratio_at_release": v["ratio"], "ci": [v["lo"], v["hi"]]}
            for t, v in res["event_peaks_at_release"][main_pair].items()]

    # forecasting, regimes, advisor
    res["forecast"] = forecast_section(bars, cal, out, note)
    w = regimes.windows(bars)
    res["regimes"] = regimes.stability(w, w.index[0] + (w.index[-1] - w.index[0]) / 2)
    adv = advise(bars, cal, pd.Timestamp("2026-11-02", tz="UTC"), 5e6, 2.0)
    res["advisor_demo"] = {
        "query": "AUDUSD, $5M, 2 hours, from Mon 2 Nov 2026",
        "cheapest": adv[~adv["avoid"]].head(5).astype(str).to_dict(orient="records"),
        "dearest": adv[~adv["avoid"]].tail(3).astype(str).to_dict(orient="records"),
        "avoid": adv[adv["avoid"]].drop_duplicates("events_nearby").head(6)[["window_start_sydney", "events_nearby"]]
        .astype(str).to_dict(orient="records")}

    if a.source == "m1" and not a.quick:
        res["case_study_2019_01_03"] = case_study(a.raw, out)
    (out / "results.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps({k: res[k] for k in ("forecast", "regimes", "event_peaks_at_release")}, indent=2, default=str))


if __name__ == "__main__":
    main()
