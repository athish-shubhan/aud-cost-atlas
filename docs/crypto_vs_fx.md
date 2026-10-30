# Crypto versus FX: what transfers from ExecLab and what does not

*One-page note. ExecLab studies execution on Binance BTCUSDT and ETHUSDT; this project studies quote-based costs in AUDUSD and AUDJPY.*

| | Crypto spot (Binance) | Spot FX (AUD pairs) |
|---|---|---|
| Trading hours | 24/7, including weekends | 24/5: Sunday 17:00 to Friday 17:00 New York time, with a daily rollover at 17:00 New York |
| Market structure | One central limit order book per venue, visible to everyone | Fragmented over-the-counter market: bank and non-bank dealers, ECNs, single-dealer platforms, internalisation |
| What public data shows | Every trade, with the aggressor side, plus the full order book | Indicative quotes from one venue; no trade prints, and volumes are not comparable across venues |
| Cost of a small order | Fees dominate: Binance's base tier charges 10 bps a side, while the BTCUSDT spread is about 0.001 bps | Spread dominates: no exchange fee. On Dukascopy the median quoted spread is 1.4 bps for AUDUSD (about one pip), 0.7 bps for AUDJPY and 0.26 bps for EURUSD |
| Cost of a large order | Can be estimated from the book (walk the levels) and from reconstructed sweeps | Cannot be estimated from public quotes; needs dealer fills, RFQ responses or last-look outcomes |
| Scheduled event risk | Weaker and not measured in ExecLab: US CPI and FOMC days do move crypto, but there is no domestic central bank or data calendar | Strong: RBA, ABS, US payrolls, CPI and FOMC releases widen AUD spreads at known times |
| Clock | One venue clock (exchange timestamps) | Sessions defined in four local time zones that change with daylight saving on different dates |

## What transfers

- **The measurement discipline.** Implementation shortfall against arrival mid, split into spread, impact, timing and fees, means the same thing in both markets.
- **The validation machinery.** Skeptic's purged walk-forward splits, block bootstrap intervals and multiple-testing control apply unchanged; in FX the event calendar replaces order-flow features as the main source of predictable cost variation.
- **Schedule choice.** TWAP, VWAP, participation and Almgren-Chriss schedules are venue-agnostic; their inputs (volume profile, volatility, impact) have to be re-estimated.
- **Thinking in fill models.** In crypto, a passive order's fate depends on its place in a visible queue. In FX it depends on a dealer's last-look and hold time, so the "pessimistic, base, optimistic" framing still applies, with different mechanics behind it.

## What does not

- **Impact estimates.** ExecLab can reconstruct the cost of market sweeps from public trades. Nothing in public FX quotes supports that, so the Atlas reports quote-based costs only and states that limit in its first paragraph.
- **Order-flow signals.** Signed trade flow is the strongest short-horizon feature in ExecLab. Public FX data has no signed flow at all; a bank sees its own client flow, which is a different and much richer signal.
- **Fees versus spread.** In crypto, choosing between passive and aggressive is mostly a fee question. In FX it is a spread and information-leakage question, which makes the passive versus aggressive trade-off more interesting, not less.
- **Weekends and the rollover.** Crypto has no weekend gap and no daily rollover; FX has both, and the rollover is one of the most predictable spread blowouts in the day.
