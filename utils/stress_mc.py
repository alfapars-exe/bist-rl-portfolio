"""Monte-Carlo stres — şişman-kuyruklu parametrik + durağan blok bootstrap.

İleri çok-varlık getiri patikalarını iki simülatörle üretip portföy ağırlıklarıyla
terminal-getiri dağılımına ve kuyruk riskine (VaR/CVaR) taşır:
  * Student-t parametrik — çok-değişkenli t (tarihsel ortalama/kovaryans), ağır kuyruk;
  * durağan blok bootstrap — ardışık tarihsel blokları yeniden örnekler; dağılım
    varsayımı olmadan otokorelasyon/volatilite kümelenmesini KORUR (Politis & Romano 1994).

PARS referans ağacından (`Reinforcement Learning Final/stress/montecarlo.py`) kanonik
BIST ağacına port edildi. Env-yerel `np.random.default_rng(seed)` → global RNG'ye dokunmaz.
GÖZLEMSEL / golden-güvenli: mevcut backtest çıktıları üzerinde post-hoc stres analizi.
"""
from __future__ import annotations

from typing import Dict

import numpy as np


def _assets_weights(weights, n_assets):
    """Ağırlık vektörü n+1 (nakit dahil) ise nakit elemanını düşürür → (n,)."""
    w = np.asarray(weights, dtype=float).reshape(-1)
    if w.shape[0] == n_assets + 1:
        w = w[:n_assets]
    return w


def summarize_mc(terminal_returns: np.ndarray, var_alpha: float = 0.05,
                 es_alpha: float = 0.025) -> Dict[str, float]:
    """Terminal getiri dağılımının özet istatistikleri + kuyruk riski (VaR/CVaR)."""
    tr = np.asarray(terminal_returns, dtype=float)
    q_var = np.quantile(tr, var_alpha)
    q_es = np.quantile(tr, es_alpha)
    tail = tr[tr <= q_es]
    return dict(
        mean=float(tr.mean()), median=float(np.median(tr)),
        p05=float(np.quantile(tr, 0.05)), p95=float(np.quantile(tr, 0.95)),
        VaR=float(max(0.0, -q_var)),
        CVaR=float(max(0.0, -(tail.mean() if tail.size else q_es))),
        worst=float(tr.min()), best=float(tr.max()),
        prob_loss=float((tr < 0).mean()),
        prob_loss_20pct=float((tr < -0.20).mean()),
    )


def mc_student_t(asset_returns: np.ndarray, weights: np.ndarray, horizon: int = 252,
                 paths: int = 5000, dof: float = 5.0, seed: int = 42) -> dict:
    """Çok-değişkenli Student-t MC (tarihsel ortalama/kovaryans, ağır kuyruk)."""
    R = np.asarray(asset_returns, dtype=float)
    n = R.shape[1]
    w = _assets_weights(weights, n)
    rng = np.random.default_rng(seed)
    mu = R.mean(0)
    cov = np.cov(R.T) + 1e-8 * np.eye(n)
    L = np.linalg.cholesky(cov)
    scale = np.sqrt((dof - 2) / dof) if dof > 2 else 1.0
    terminal = np.empty(paths)
    for p in range(paths):
        z = rng.standard_normal((horizon, n))
        g = rng.chisquare(dof, size=(horizon, 1)) / dof
        t = z / np.sqrt(g)                               # standart çok-değişkenli-t yenilik
        daily = mu + (t * scale) @ L.T
        port = daily @ w
        terminal[p] = np.prod(1 + port) - 1
    return dict(terminal=terminal, summary=summarize_mc(terminal))


def mc_block_bootstrap(asset_returns: np.ndarray, weights: np.ndarray, horizon: int = 252,
                       paths: int = 5000, block: int = 20, seed: int = 42) -> dict:
    """Tarihsel getirilerin durağan blok bootstrap'ı → terminal dağılım."""
    R = np.asarray(asset_returns, dtype=float)
    T, n = R.shape
    w = _assets_weights(weights, n)
    rng = np.random.default_rng(seed)
    pblock = 1.0 / block
    terminal = np.empty(paths)
    for p in range(paths):
        idx = np.empty(horizon, dtype=int)
        i = rng.integers(0, T)
        for h in range(horizon):
            if h > 0 and rng.random() < pblock:
                i = rng.integers(0, T)                   # yeni blok başlat
            idx[h] = i
            i = (i + 1) % T
        port = R[idx] @ w
        terminal[p] = np.prod(1 + port) - 1
    return dict(terminal=terminal, summary=summarize_mc(terminal))
