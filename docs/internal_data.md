# What I would do with internal data

Public quotes show what it costs to cross a small amount at the top of one venue's book. A bank's own records would answer the questions this study cannot:

1. **Realised slippage by size.** With the desk's own fills and the mid at arrival, the implementation shortfall framework from ExecLab applies directly: spread, impact and timing cost by notional bucket, session and event proximity. That gives the size dimension the Atlas is missing.
2. **RFQ economics.** Request-for-quote logs (quoted spread, hit ratio, time to respond) would show how dealer pricing widens with size and around releases, and whether the public-quote heatmap predicts it.
3. **Internalisation versus external hedging.** Client flow that offsets internally costs nothing to hedge. Modelling the chance that flow nets out within a window would turn the advisor from "when is the market cheap" into "when should we hedge rather than wait".
4. **Last-look and rejection rates.** For passive or streamed liquidity, fill and reject rates by venue are the FX analogue of ExecLab's fill models, and would let the pessimistic, base and optimistic cases be estimated rather than assumed.
5. **Validation.** Every one of these would go through the same Skeptic protocol: walk-forward by month, intervals by block bootstrap, and multiple-testing control across pairs, sessions and events.
