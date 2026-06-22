"""metrics.py bilinen-deger (known-value) regresyon testleri.

Refactor sirasinda metrik formulleri sessizce degisirse bu testler yakalar.
"""
import numpy as np

from utils.metrics import (cagr, sharpe, sortino, max_drawdown, calmar,
                           turnover, success_vs_benchmark, summary,
                           moving_average, training_diagnostics, real_nav)


def test_real_nav_lira_illusion():
    # Nominal NAV 1->2 (TL bazinda 2x). USD/TRY 10->20 (TL %50 deger kaybetti).
    nav = np.array([1.0, 1.5, 2.0])
    fx = np.array([10.0, 15.0, 20.0])
    rn = real_nav(nav, fx)
    assert np.isclose(rn[0], 1.0)                        # baslangicta 1.0
    # reel = (nav/nav0)/(fx/fx0): t=2 -> (2/1)/(20/10) = 1.0 (reel kazanc YOK — illuzyon)
    assert np.isclose(rn[-1], 1.0)
    assert np.all(rn <= nav + 1e-12)                     # TL deger kaybinda reel <= nominal
    # bos/dejenere giris -> guvenli
    assert real_nav(np.array([]), fx).size == 0


def test_moving_average_basic_and_edge():
    ma = moving_average([1.0, 2.0, 3.0, 4.0], window=2)
    assert np.allclose(ma, [1.5, 2.5, 3.5])
    assert moving_average([], window=3).size == 0          # bos dizi
    assert np.allclose(moving_average([5.0], window=4), [5.0])  # pencere kisilir


def test_training_diagnostics_keys_and_values():
    curve = [{"reward": 1.0, "success": 1, "steps": 100},
             {"reward": -1.0, "success": 0, "steps": 200}]
    rth = [{"bankrupt": False, "drawdown_penalty": 0.0, "tx_cost": 0.001},
           {"bankrupt": True, "drawdown_penalty": 0.5, "tx_cost": 0.003}]
    d = training_diagnostics(curve, rth, ma_window=2)
    assert d["episodes"] == 2
    assert abs(d["success_rate"] - 0.5) < 1e-12
    assert abs(d["avg_steps"] - 150.0) < 1e-12
    assert d["bankrupt_count"] == 1
    assert d["drawdown_penalty_steps"] == 1
    assert abs(d["mean_tx_cost"] - 0.002) < 1e-12


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


def test_sortino_no_downside_is_nan_positive_mean():
    """v10 NaN-guard: hic asagi-yonlu sapma yok VE pozitif ortalama -> Sortino NaN.
    (Onceki surum inf donuyordu; NaN-guard matematiksel 'tanimsiz' semantigini uygular:
    sifir downside dev ile oran tanimlanamaz, sonsuz degil NaN.) Sifir-ortalamada 0.0."""
    s = sortino(np.array([0.01, 0.02, 0.005]))
    # downside_dev = 0, mu > 0 -> NaN (sifir bolen ile tanimlanamaz oran)
    assert np.isnan(s), (
        f"Pozitif sabit getiri -> Sortino NaN beklenir (NaN-guard); bulunan {s}")
    # tum-sifir: mu=0, downside_dev=0 -> mu<=0 -> 0.0
    assert abs(sortino(np.zeros(5))) < 1e-12, "Tum-sifir getiri -> Sortino 0.0"


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


def test_calmar_nan_when_no_drawdown():
    """v10 NaN-guard: MaxDD=0 (dususuz seri) -> Calmar NaN (tanimsiz, 1e-9 korumasi kaldirildi).
    Matematiksel anlam: sifir cekiliste Calmar orani tanimlanamaz (CashRiskFree gibi).
    Cekilis iceren normal seride Calmar sonlu olmali."""
    nav = np.ones(252)
    nav[-1] = 2.0   # MaxDD=0 (monoton artan) -> NaN
    assert np.isnan(calmar(nav)), (
        f"MaxDD=0 -> Calmar NaN beklenir (NaN-guard); bulunan {calmar(nav)}")
    # Cekilis iceren seri -> sonlu
    rets = np.random.default_rng(0).normal(0.0003, 0.012, 252)
    nav2 = np.cumprod(1 + rets)
    assert np.isfinite(calmar(nav2)), "Cekilis var -> Calmar sonlu olmali"


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
