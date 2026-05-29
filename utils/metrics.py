"""Backtest metrikleri — CAGR, Sharpe, Sortino, MaxDD, Calmar, Turnover + başarı metriği."""
from __future__ import annotations

import numpy as np

TRADING_DAYS = 252


def cagr(nav: np.ndarray) -> float:
    years = len(nav) / TRADING_DAYS
    return float(nav[-1] ** (1 / max(years, 1e-6)) - 1)


def sharpe(rets: np.ndarray, rf: float = 0.0) -> float:
    mu = np.mean(rets) - rf / TRADING_DAYS
    sd = np.std(rets) + 1e-9
    return float(np.sqrt(TRADING_DAYS) * mu / sd)


def sortino(rets: np.ndarray, rf: float = 0.0) -> float:
    downside = rets[rets < 0]
    sd = np.std(downside) + 1e-9
    mu = np.mean(rets) - rf / TRADING_DAYS
    return float(np.sqrt(TRADING_DAYS) * mu / sd)


def max_drawdown(nav: np.ndarray) -> float:
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    return float(dd.min())


def calmar(nav: np.ndarray) -> float:
    mdd = abs(max_drawdown(nav))
    return cagr(nav) / (mdd + 1e-9)


def turnover(weights: np.ndarray) -> float:
    """Günlük ‖Δw‖₁ ortalaması."""
    d = np.abs(np.diff(weights, axis=0)).sum(axis=1)
    return float(d.mean()) if len(d) else 0.0


def summary(nav: np.ndarray, rets: np.ndarray, weights=None) -> dict:
    out = dict(
        CAGR=cagr(nav),
        Sharpe=sharpe(rets),
        Sortino=sortino(rets),
        MaxDD=max_drawdown(nav),
        Calmar=calmar(nav),
        Volatility=float(np.std(rets) * np.sqrt(TRADING_DAYS)),
        FinalNAV=float(nav[-1]),
    )
    if weights is not None:
        out["Turnover"] = turnover(np.asarray(weights))
    else:
        out["Turnover"] = 0.0
    return out


def success_vs_benchmark(nav_agent: np.ndarray, nav_bench: np.ndarray) -> int:
    """Prompt başarı tanımı: episod sonunda ajan NAV'ı benchmark NAV'ını geçtiyse 1, aksi halde 0."""
    if len(nav_agent) == 0 or len(nav_bench) == 0:
        return 0
    m = min(len(nav_agent), len(nav_bench))
    return int(nav_agent[m - 1] >= nav_bench[m - 1])
