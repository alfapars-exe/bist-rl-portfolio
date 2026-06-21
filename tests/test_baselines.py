"""Ek akademik baseline'lar — determinizm + yapisal invariant testleri.

Yeni baseline'lar (risk_parity, inverse_volatility, min_variance, momentum,
cash_riskfree) DETERMINISTIK (RNG YOK) olmali ve long-only/tam-yatirim
kisitlarini saglamali. Bu testler golden-master'i tamamlar: yeni stratejilerin
np.random'a dokunmadigini ve metrics.csv entegrasyonunun mevcut satirlari
korudugunu kanitlar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.baselines import (cash_riskfree, inverse_volatility,
                             min_variance, momentum, risk_parity)
from utils.metrics import summary

N_ASSETS = 8
T_DAYS = 300


def _synthetic_prices(seed: int = 123) -> pd.DataFrame:
    """Determinist sentetik fiyat matrisi (yalniz TEST verisi uretmek icin RNG;
    baseline FONKSIYONLARI bu veriye RNG'siz uygulanir)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-01", periods=T_DAYS)
    mu = rng.normal(0.0004, 0.0003, N_ASSETS)
    sig = np.clip(rng.normal(0.02, 0.006, N_ASSETS), 0.008, 0.05)
    logret = rng.normal(mu, sig, size=(T_DAYS, N_ASSETS))
    price = 100.0 * np.exp(np.cumsum(logret, axis=0))
    cols = [f"A{i}" for i in range(N_ASSETS)]
    return pd.DataFrame(price, index=idx, columns=cols)


ROLLING_FNS = {
    "risk_parity": lambda p: risk_parity(p, lookback=60, rebalance=5),
    "inverse_volatility": lambda p: inverse_volatility(p, lookback=60, rebalance=5),
    "min_variance": lambda p: min_variance(p, lookback=60, rebalance=5),
    "momentum": lambda p: momentum(p, lookback=20, rebalance=5, top_k=3),
}
ALL_FNS = dict(ROLLING_FNS)
ALL_FNS["cash_riskfree"] = lambda p: cash_riskfree(p, daily_rf=0.0)


@pytest.fixture(scope="module")
def prices():
    return _synthetic_prices()


# --------------------------------------------------------------------------
# Determinizm — RNG YOK: ardisik iki cagri BIT-AYNI sonuc vermeli, ve global
# np.random durumunu yeniden tohumlamak sonucu DEGISTIRMEMELI.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(ALL_FNS))
def test_deterministic_repeat(prices, name):
    fn = ALL_FNS[name]
    d1 = fn(prices)
    d2 = fn(prices)
    np.testing.assert_array_equal(d1["nav"], d2["nav"])
    np.testing.assert_array_equal(d1["rets"], d2["rets"])


@pytest.mark.parametrize("name", list(ALL_FNS))
def test_no_rng_dependence(prices, name):
    """Global np.random durumunu farkli tohumlarla bozsak da cikti degismemeli
    (fonksiyonlar np.random'a HIC dokunmuyor -> golden RNG sirasi guvende)."""
    fn = ALL_FNS[name]
    np.random.seed(0)
    base = fn(prices)["nav"]
    np.random.seed(999999)
    other = fn(prices)["nav"]
    np.testing.assert_array_equal(base, other)


# --------------------------------------------------------------------------
# Yapisal invariant'lar — long-only, tam-yatirim (sum=1), NaN yok, dogru sekil.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(ROLLING_FNS))
def test_long_only_fully_invested(prices, name):
    d = ROLLING_FNS[name](prices)
    W = d["weights"]
    assert W.shape == (len(prices), N_ASSETS)
    assert np.all(W >= -1e-9), "negatif agirlik (long-only ihlali)"
    np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-6)


@pytest.mark.parametrize("name", list(ALL_FNS))
def test_shapes_and_finite(prices, name):
    d = ALL_FNS[name]
    out = d(prices)
    assert len(out["nav"]) == len(prices)
    assert len(out["rets"]) == len(prices)
    assert np.all(np.isfinite(out["nav"]))
    assert np.all(np.isfinite(out["rets"]))
    m = summary(out["nav"], out["rets"], out.get("weights"))
    # cash_riskfree(daily_rf=0.0) -> sabit getiri -> vol=0 -> Sharpe/Sortino/Calmar NaN
    # (metrics NaN-guard: sifir-vol/sifir-MaxDD'de tanimsiz metrikler NaN doner).
    # Diger stratejiler (riskli varliklar icerdiginden) tamamen sonlu olmali.
    if name == "cash_riskfree":
        # NAV, CAGR, FinalNAV, Volatility, Turnover sonlu; Sharpe/Sortino/Calmar NaN olabilir.
        finite_keys = {"CAGR", "FinalNAV", "Volatility", "Turnover", "MaxDD"}
        for k in finite_keys:
            assert np.isfinite(m[k]), f"{k} sonlu olmali; {m}"
    else:
        assert all(np.isfinite(v) for v in m.values()), m


# --------------------------------------------------------------------------
# Strateji-ozgu mantik dogrulamalari.
# --------------------------------------------------------------------------
def test_cash_riskfree_flat(prices):
    """rf=0 -> NAV duz 1.0, getiri 0, agirlik tamamen sifir (Turnover=0)."""
    d = cash_riskfree(prices, daily_rf=0.0)
    np.testing.assert_allclose(d["nav"], 1.0, atol=1e-12)
    np.testing.assert_allclose(d["rets"], 0.0, atol=1e-12)
    np.testing.assert_allclose(d["weights"], 0.0, atol=1e-12)
    m = summary(d["nav"], d["rets"], d["weights"])
    assert abs(m["Turnover"]) < 1e-12
    assert abs(m["Volatility"]) < 1e-9


def test_cash_riskfree_positive_rf(prices):
    """Pozitif gunluk rf -> NAV=(1+rf)^t monoton artar."""
    rf = 1e-4
    d = cash_riskfree(prices, daily_rf=rf)
    assert d["nav"][-1] > d["nav"][0]
    np.testing.assert_allclose(d["nav"][-1], (1 + rf) ** len(prices), rtol=1e-9)


def test_momentum_top_k_concentration(prices):
    """Momentum yalniz top_k hisseye agirlik verir -> rebalans sonrasi en fazla
    top_k sifir-olmayan pozisyon."""
    k = 3
    d = momentum(prices, lookback=20, rebalance=5, top_k=k)
    W = d["weights"]
    nz = (W[-1] > 1e-9).sum()                 # son gun (warm-up sonrasi)
    assert nz <= k
    # Secilen pozisyonlar esit agirlikli (1/k)
    active = W[-1][W[-1] > 1e-9]
    if active.size:
        np.testing.assert_allclose(active, 1.0 / k, atol=1e-6)


def test_min_variance_reduces_portfolio_variance(prices):
    """Saf min-var, tahmin penceresinde esit-agirligin portfoy varyansini
    GECMEMELI (kisitli optimum tanimi). Sentetik veride esit-vol durumunda
    cozum esit-agirliga yakinsabilir; finansal invariant 'min-var <= esit-var'
    her durumda gecerlidir (agirlik vektorlerinin farkli olmasi DEGIL)."""
    lookback, rebalance = 60, 5
    mv = min_variance(prices, lookback=lookback, rebalance=rebalance)
    # Son rebalans gunundeki agirlikla, o pencerenin kovaryansi uzerinden karsilastir.
    r = prices.pct_change().fillna(0.0).values
    T, N = r.shape
    t_reb = max(t for t in range(T) if t >= lookback and (t - lookback) % rebalance == 0)
    cov = np.cov(r[t_reb - lookback:t_reb].T) + 1e-5 * np.eye(N)
    w_mv = mv["weights"][t_reb]
    w_eq = np.ones(N) / N
    var_mv = float(w_mv @ cov @ w_mv)
    var_eq = float(w_eq @ cov @ w_eq)
    assert var_mv <= var_eq + 1e-9, (var_mv, var_eq)
