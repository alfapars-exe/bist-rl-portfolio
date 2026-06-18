"""Deflated/Probabilistic Sharpe + CSCV-PBO bilinen-deger testleri (López de Prado).

utils/deflated_sharpe.py GOZLEMSEL (raporlama) — golden'a dokunmaz; bu testler
istatistiksel ozellikleri (monotonluk, sinir, overfitting tespiti) kilitler.
"""
import numpy as np

from utils.deflated_sharpe import (probabilistic_sharpe_ratio, expected_max_sharpe,
                                   deflated_sharpe_ratio, cscv_pbo)


def test_psr_monotonic_and_bounded():
    p_lo = probabilistic_sharpe_ratio(0.05, 252)
    p_hi = probabilistic_sharpe_ratio(0.15, 252)
    assert 0.0 <= p_lo <= 1.0 and 0.0 <= p_hi <= 1.0
    assert p_hi > p_lo                                   # yuksek Sharpe -> yuksek PSR
    # daha cok gozlem -> tahmin hatasi duser -> PSR artar (pozitif SR'de)
    assert probabilistic_sharpe_ratio(0.1, 1000) > probabilistic_sharpe_ratio(0.1, 50)


def test_expected_max_sharpe_increases_with_trials():
    e1 = expected_max_sharpe(0.01, 1)
    e10 = expected_max_sharpe(0.01, 10)
    e100 = expected_max_sharpe(0.01, 100)
    assert e1 == 0.0                                     # tek deneme -> deflasyon yok
    assert e100 > e10 > e1                               # daha cok deneme -> daha yuksek esik


def test_dsr_not_greater_than_psr_under_multiple_trials():
    # DSR, E[max] (>=0) benchmark'ina karsi deflate eder -> DSR <= PSR(benchmark=0)
    sr, n = 0.10, 500
    psr = probabilistic_sharpe_ratio(sr, n)
    dsr = deflated_sharpe_ratio(sr, n, n_trials=50, sr_variance=0.01)
    assert dsr <= psr + 1e-9
    assert 0.0 <= dsr <= 1.0


def test_cscv_pbo_detects_overfitting():
    rng = np.random.default_rng(0)
    T, N = 240, 8
    # SAGLAM: bir config gercekten her yerde ustun -> dusuk PBO
    robust = rng.normal(0.0, 0.01, (T, N))
    robust[:, 0] += 0.004                                # kalici gercek kenar
    pbo_robust = cscv_pbo(robust, n_splits=8)
    assert pbo_robust["pbo"] < 0.25
    # SAF GURULTU: kalicilik yok -> PBO gecerli olasilik araliginda
    pbo_noise = cscv_pbo(rng.normal(0.0, 0.01, (T, N)), n_splits=8)
    assert 0.0 <= pbo_noise["pbo"] <= 1.0
    assert pbo_robust["n_combos"] > 0


def test_cscv_pbo_degenerate_inputs():
    # tek config veya cok kisa seri -> NaN, cokme yok
    out = cscv_pbo(np.zeros((100, 1)), n_splits=10)
    assert np.isnan(out["pbo"]) and out["n_combos"] == 0
