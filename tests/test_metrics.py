"""metrics.py bilinen-deger (known-value) regresyon testleri.

Refactor sirasinda metrik formulleri sessizce degisirse bu testler yakalar.
"""
import numpy as np

from utils.metrics import (cagr, sharpe, sortino, max_drawdown, calmar,
                           turnover, success_vs_benchmark, summary)


def test_sortino_known_value():
    """Kanonik downside deviation: sqrt(mean(min(r,0)^2)) TUM gozlemlerle.
    rets=[0.02, -0.01, 0.03, -0.02] -> dd = sqrt((0.0001+0.0004)/4) = sqrt(0.000125)
    mu = 0.005 -> sortino = sqrt(252)*0.005/0.011180..."""
    rets = np.array([0.02, -0.01, 0.03, -0.02])
    dd = np.sqrt((0.01**2 + 0.02**2) / 4)
    expected = np.sqrt(252) * 0.005 / dd
    assert abs(sortino(rets) - expected) < 1e-9


def test_sortino_uniform_losses_not_inflated():
    """Eski formulun hatasi: her kaybi ayni olan seri (alt-orneklem std=0)
    oranı sisiriyordu. Kanonik formda payda > 0 kalir ve oran sonlu/negatif."""
    rets = np.array([0.01, -0.05, 0.01, -0.05, 0.01, -0.05])
    s = sortino(rets)
    assert np.isfinite(s)
    assert s < 0  # ortalama negatif -> oran negatif olmali


def test_sortino_no_downside_is_inf_not_nan():
    """Hic negatif getiri yokken eski surum NaN donuyordu (np.std bos dizi)."""
    s = sortino(np.array([0.01, 0.02, 0.005]))
    assert not np.isnan(s)
    assert s == float("inf")
    assert sortino(np.zeros(5)) == 0.0


def test_max_drawdown_known():
    nav = np.array([1.0, 1.2, 0.6, 0.9])  # tepe 1.2 -> dip 0.6 => -%50
    assert abs(max_drawdown(nav) - (-0.5)) < 1e-12


def test_max_drawdown_monotonic_is_zero():
    nav = np.array([1.0, 1.1, 1.2, 1.3])
    assert abs(max_drawdown(nav)) < 1e-12


def test_cagr_doubling_over_one_year():
    nav = np.ones(252)
    nav[-1] = 2.0  # tam 1 yil (252 gun) icinde 2x -> CAGR = 100%
    assert abs(cagr(nav) - 1.0) < 1e-9


def test_turnover_known():
    w = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    # |Delta| satir1 = 2, satir2 = 0 -> ortalama 1.0
    assert abs(turnover(w) - 1.0) < 1e-12


def test_sharpe_zero_mean_is_zero():
    rets = np.array([0.01, -0.01, 0.01, -0.01])  # ortalama 0
    assert abs(sharpe(rets)) < 1e-9


def test_calmar_finite_when_no_drawdown():
    nav = np.ones(252)
    nav[-1] = 2.0  # mdd = 0 -> 1e-9 korumasi sayesinde sonlu kalmali
    assert np.isfinite(calmar(nav))


def test_success_vs_benchmark():
    assert success_vs_benchmark(np.array([1.0, 2.0]), np.array([1.0, 1.5])) == 1
    assert success_vs_benchmark(np.array([1.0, 1.0]), np.array([1.0, 2.0])) == 0


def test_summary_has_all_keys():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, size=300)  # hem pozitif hem negatif getiriler
    nav = np.cumprod(1.0 + rets)
    out = summary(nav, rets)
    for key in ["CAGR", "Sharpe", "Sortino", "MaxDD", "Calmar",
                "Volatility", "FinalNAV", "Turnover"]:
        assert key in out
