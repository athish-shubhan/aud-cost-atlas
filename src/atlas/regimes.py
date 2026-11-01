"""Liquidity regimes: PCA + Gaussian mixture on 15-minute windows, with a stability test.

Stability: fit on year 1, then fit separately on year 2. If regimes are real, the
year-1 model should describe year 2 nearly as well as year 2's own model, and the
two should label year-2 windows the same way (adjusted Rand index near 1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

COLS = ["log_spread", "log_spread_vol", "log_quotes", "log_rv", "jump_rate"]
NAMES = ["calm", "active", "stressed"]


def windows(bars: pd.DataFrame) -> pd.DataFrame:
    g = bars.resample("15min")
    w = pd.DataFrame({
        "log_spread": np.log(g["spread_bps"].mean()),
        "log_spread_vol": np.log(g["spread_bps"].std() + 1e-3),
        "log_quotes": np.log1p(g["quotes"].sum()),
        "log_rv": np.log(np.sqrt((bars["rv_bps"] ** 2).resample("15min").sum()) + 1e-3),
        "jump_rate": g["jump"].mean(),
    }).dropna()
    return pd.DataFrame(w[np.isfinite(w).all(axis=1)])


@dataclass
class RegimeModel:
    pipe: Pipeline
    gmm: GaussianMixture
    order: np.ndarray  # component index sorted by mean spread: calm, active, stressed

    def label(self, w: pd.DataFrame) -> np.ndarray:
        raw = self.gmm.predict(self.pipe.transform(w[COLS]))
        rank = np.empty_like(self.order)
        rank[self.order] = np.arange(self.order.size)
        return rank[raw]

    def score(self, w: pd.DataFrame) -> float:
        return float(self.gmm.score(self.pipe.transform(w[COLS])))


def fit(w: pd.DataFrame, k: int = 3, seed: int = 0) -> RegimeModel:
    pipe = make_pipeline(StandardScaler(), PCA(n_components=3, random_state=seed)).fit(w[COLS])
    z = pipe.transform(w[COLS])
    gmm = GaussianMixture(k, covariance_type="full", n_init=5, random_state=seed).fit(z)
    lab = gmm.predict(z)
    means = np.array([w["log_spread"].to_numpy()[lab == c].mean() if (lab == c).any() else np.inf
                      for c in range(k)])
    return RegimeModel(pipe, gmm, np.argsort(means))


def stability(w: pd.DataFrame, split: pd.Timestamp, seed: int = 0) -> dict[str, float | list[float]]:
    y1, y2 = w[w.index < split], w[w.index >= split]
    m1, m2 = fit(y1, seed=seed), fit(y2, seed=seed)
    l1, l2 = m1.label(y2), m2.label(y2)
    share = lambda lab: [float(np.mean(lab == i)) for i in range(3)]
    return {"ari_year1_vs_year2_labels": float(adjusted_rand_score(l1, l2)),
            "loglik_year2_under_year1_model": m1.score(y2),
            "loglik_year2_under_own_model": m2.score(y2),
            "regime_share_year2_by_year1_model": share(l1),
            "regime_share_year2_by_own_model": share(l2)}
