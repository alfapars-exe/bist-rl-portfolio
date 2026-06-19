"""V8 — fiyat gurultusu / slippage testleri (hocanin ACIK sarti, anti-ezber).

Hoca finansal projede gurultuyu sart kostu: "al dediginde tam o fiyattan
alamazsin, yukaridan alirsin... hem gercekci olur HEM EZBERI ONLER." Bu testler
iki invariant'i kilitler:

  (1) Gurultu YALNIZ egitimde (random_start=True) aktif; eval'de (random_start=False)
      KAPALI -> golden eval rollout'u deterministik kalir (1e-6 reproducibility).
  (2) Egitimde gurultu gercekten realize getiriyi degistirir (ezberi onler), ve
      env-yerel rng kullanir (global np.random'a dokunmaz).
"""
import numpy as np
import pandas as pd

from env.portfolio_env import PortfolioEnv
from utils.features import add_features


def _market(seed=0, n=180, k=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0)),
        index=idx, columns=[f"A{i}" for i in range(k)])
    return prices, add_features(prices)


def test_noise_off_in_eval():
    """random_start=False (eval) -> gurultu kapali, _risky_returns gurultusuz & sabit."""
    prices, feats = _market()
    env = PortfolioEnv(prices, feats, horizon="short", random_start=False,
                       seed=0, price_noise_std=0.05)
    assert env._noise_active is False
    r1 = env._risky_returns()
    r2 = env._risky_returns()                       # ayni t -> rng tuketilmez -> birebir ayni
    np.testing.assert_array_equal(r1, r2)
    # analitik (gurultusuz) getiriye esit olmali
    # v10: nakit varlik artik 0.0 degil, cash_daily_rate kazanir (SABIT skaler).
    # float64->float32 donusumu kucuk yuvarlama hatasi (<1e-7) getirebilir; allclose kullan.
    p0, p1 = env.prices[env.t], env.prices[env.t + 1]
    clean = np.concatenate(
        [(p1 - p0) / np.maximum(p0, 1e-9), [env.cash_daily_rate]]
    ).astype(np.float32)
    np.testing.assert_allclose(r1, clean, atol=1e-6,
                               err_msg="Eval risky_returns gurultusuz analitik degerle eslesmeli")


def test_noise_on_in_training():
    """random_start=True (egitim) -> gurultu aktif; ardisik cagrilar farkli (rng tuketir)."""
    prices, feats = _market()
    env = PortfolioEnv(prices, feats, horizon="short", random_start=True,
                       seed=0, price_noise_std=0.05)
    assert env._noise_active is True
    t0 = env.t
    r1 = env._risky_returns()
    r2 = env._risky_returns()
    assert env.t == t0, "_risky_returns t'yi ilerletmemeli"
    assert not np.array_equal(r1, r2), "gurultu aktifken ardisik draw'lar farkli olmali"
    # nakit varligi gurultusuz kalmali (slippage yalniz riskli varliklarda).
    # v10: nakit getirisi 0.0 degil cash_daily_rate (SABIT skaler, gurultusuz).
    dr = env.cash_daily_rate
    assert np.isclose(r1[-1], dr), f"Nakit getirisi cash_daily_rate={dr} olmali; {r1[-1]}"
    assert np.isclose(r2[-1], dr), f"Nakit getirisi cash_daily_rate={dr} olmali; {r2[-1]}"


def test_eval_rollout_invariant_to_noise_setting():
    """KRITIK golden-guvenligi: eval (random_start=False) rollout'u price_noise_std'den
    BAGIMSIZ -> gurultuyu acmak golden eval metriklerini DEGISTIRMEZ (yalniz egitim degisir)."""
    prices, feats = _market()
    e0 = PortfolioEnv(prices, feats, horizon="short", random_start=False,
                      seed=0, price_noise_std=0.0)
    e1 = PortfolioEnv(prices, feats, horizon="short", random_start=False,
                      seed=0, price_noise_std=0.1)
    rng = np.random.default_rng(123)
    navs0, navs1 = [], []
    e0.reset(); e1.reset()
    for _ in range(40):
        a = rng.normal(0, 1, e0.action_dim).astype(np.float32)   # ayni aksiyon dizisi
        _, _, d0, t0, _ = e0.step(a)
        _, _, d1, t1, _ = e1.step(a)
        navs0.append(e0.nav); navs1.append(e1.nav)
        if d0 or t0 or d1 or t1:
            break
    np.testing.assert_allclose(navs0, navs1, atol=1e-12,
                               err_msg="Eval NAV'i gurultu ayarindan etkilendi — golden riskli!")


def test_training_noise_uses_env_local_rng():
    """Gurultu env-yerel self.rng kullanir -> global np.random durumuna dokunmaz."""
    prices, feats = _market()
    env = PortfolioEnv(prices, feats, horizon="short", random_start=True,
                       seed=0, price_noise_std=0.05)
    np.random.seed(777)
    before = np.random.get_state()[1][0]
    for _ in range(10):
        env._risky_returns()
    after = np.random.get_state()[1][0]
    assert before == after, "fiyat gurultusu global np.random'i tuketmemeli (env-yerel rng)"
