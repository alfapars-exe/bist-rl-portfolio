"""Klasik baseline stratejiler — karşılaştırma için."""
from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weight(prices: pd.DataFrame) -> dict:
    """Günlük rebalansla eşit ağırlık."""
    r = prices.pct_change().fillna(0).values
    N = prices.shape[1]
    w = np.ones(N) / N
    rets = (r * w).sum(axis=1)
    nav = np.cumprod(1 + rets)
    return dict(nav=nav, rets=rets, weights=np.tile(w, (len(rets), 1)))


def buy_and_hold_index(prices: pd.DataFrame) -> dict:
    """Eşit ağırlık alıp tut (rebalans yok)."""
    p0 = prices.iloc[0].values
    shares = 1.0 / p0 / prices.shape[1]
    nav = (prices.values * shares).sum(axis=1)
    rets = np.diff(np.log(nav))
    return dict(
        nav=nav / nav[0],
        rets=np.concatenate([[0.0], rets]),
        weights=None,
    )


def mean_variance(prices: pd.DataFrame, lookback: int = 120,
                  rebalance: int = 20, risk_aversion: float = 5.0) -> dict:
    """Kısıtlı long-only Markowitz; yuvarlanan pencere + periyodik rebalans."""
    from scipy.optimize import minimize

    r = prices.pct_change().fillna(0).values
    T, N = r.shape
    navs = [1.0]; rets = []; w_hist = []
    w = np.ones(N) / N
    for t in range(T):
        if t >= lookback and (t - lookback) % rebalance == 0:
            hist = r[t - lookback: t]
            mu = hist.mean(axis=0)
            cov = np.cov(hist.T) + 1e-5 * np.eye(N)

            def obj(w_, mu=mu, cov=cov, ra=risk_aversion):
                return -(w_ @ mu) + 0.5 * ra * w_ @ cov @ w_

            res = minimize(
                obj, np.ones(N) / N,
                bounds=[(0, 0.2)] * N,
                constraints=({"type": "eq", "fun": lambda w_: w_.sum() - 1}),
            )
            w = res.x
        port_r = float((w * r[t]).sum())
        rets.append(port_r)
        navs.append(navs[-1] * (1 + port_r))
        w_hist.append(w.copy())
    return dict(
        nav=np.array(navs[1:]),
        rets=np.array(rets),
        weights=np.array(w_hist),
    )
