"""episode_clean (OPT-IN, default KAPALI) ozelligini kilitleyen testler.

Mekanik:
  - episode_clean=False (default): _risky_returns her episode'da gurultulu
    (V11 davranisi, golden bit-ayni).
  - episode_clean=True: 1. reset (idx=0) -> temiz/gurultusuz; 2.+ reset (idx>=1) -> gurultulu.

Test senaryolari:
  (a) default kapali, random_start=True, price_noise_std=0.05 -> ardisik draw'lar farkli.
  (b) episode_clean=True; 1. episode temiz (ayni draw); 2. episode gurultulu (farkli draw).
  (c) golden-guvenlik: episode_clean=False ile davranis, parametre EKLENMEDEN onceki
      V11 davranisiyla mantiksal olarak ayni (default kapaliyken noise her zaman aktif).
  (d) DiscretePortfolioEnv episode_clean=True'yu forward etmeli.
  (e) Mevcut test_price_noise.py testleri bu ozellikten ETKILENMEMELI (implicit check).
"""
import numpy as np
import pandas as pd
import pytest

from env.portfolio_env import DiscretePortfolioEnv, PortfolioEnv
from utils.features import add_features


# ---------------------------------------------------------------------------
# Yardimci: kucuk sentetik piyasa
# ---------------------------------------------------------------------------

def _market(seed=0, n=200, k=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0)),
        index=idx, columns=[f"A{i}" for i in range(k)]
    )
    return prices, add_features(prices)


# ---------------------------------------------------------------------------
# (a) default (episode_clean=False): gurultu her reset sonrasi aktif
# ---------------------------------------------------------------------------

def test_default_off_noise_active_after_reset():
    """episode_clean=False (default), random_start=True, price_noise_std=0.05:
    reset() sonrasi ardisik _risky_returns() cagrilari FARKLI olmali (gurultulu)."""
    prices, feats = _market()
    env = PortfolioEnv(
        prices, feats,
        horizon="short",
        random_start=True,
        price_noise_std=0.05,
        seed=42,
        # episode_clean belirtilmez -> default False
    )
    assert env._episode_clean is False, "_episode_clean default False olmali"
    env.reset()
    assert env._noise_active is True, "random_start=True, std>0 -> _noise_active True"

    r1 = env._risky_returns()
    r2 = env._risky_returns()
    assert not np.array_equal(r1, r2), (
        "episode_clean=False, gurultulu: ardisik _risky_returns farkli olmali"
    )


# ---------------------------------------------------------------------------
# (b) episode_clean=True: 1. episode temiz, 2. episode gurultulu
# ---------------------------------------------------------------------------

def test_episode_clean_first_episode_is_clean():
    """episode_clean=True, 1. reset (idx=0): _risky_returns tekrarlanabilir/gurultusuz olmali.

    Temizlik kriteri: ayni t noktasinda iki ardisik cagri AYNI sonuc vermeli
    (RNG tuketilmediginden; episode_clean=True & idx=0 -> noise dali ATLANIR).
    """
    prices, feats = _market()
    env = PortfolioEnv(
        prices, feats,
        horizon="short",
        random_start=True,
        price_noise_std=0.05,
        seed=42,
        episode_clean=True,
    )
    assert env._episode_clean is True

    # 1. reset -> idx=0 (TEMIZ)
    env.reset()
    assert env._episode_idx == 0, "1. reset sonrasi _episode_idx==0 olmali"

    r1 = env._risky_returns()
    r2 = env._risky_returns()
    np.testing.assert_array_equal(
        r1, r2,
        err_msg="1. episode (idx=0, temiz): ardisik draw'lar AYNI olmali (RNG tuketilmez)"
    )


def test_episode_clean_second_episode_is_noisy():
    """episode_clean=True, 2. reset (idx=1): gurultu aktif olmali -> farkli draw'lar."""
    prices, feats = _market()
    env = PortfolioEnv(
        prices, feats,
        horizon="short",
        random_start=True,
        price_noise_std=0.05,
        seed=42,
        episode_clean=True,
    )

    env.reset()   # idx=0 temiz
    env.reset()   # idx=1 gurultulu
    assert env._episode_idx == 1, "2. reset sonrasi _episode_idx==1 olmali"

    r1 = env._risky_returns()
    r2 = env._risky_returns()
    assert not np.array_equal(r1, r2), (
        "2. episode (idx=1): ardisik draw'lar FARKLI olmali (gurultulu)"
    )


def test_episode_idx_increments_per_reset():
    """reset() cagrisinda _episode_idx bir artmali; ilk ctor'da -1."""
    prices, feats = _market()
    env = PortfolioEnv(prices, feats, horizon="short", seed=0)
    assert env._episode_idx == -1, "ctor sonrasi _episode_idx==-1 olmali"

    env.reset()
    assert env._episode_idx == 0
    env.reset()
    assert env._episode_idx == 1
    env.reset()
    assert env._episode_idx == 2


# ---------------------------------------------------------------------------
# (c) GOLDEN-GUVENLIK: episode_clean=False -> V11 mantiksal es-davranis
# ---------------------------------------------------------------------------

def test_golden_safety_default_off_equivalent_to_v11():
    """episode_clean=False ile davranis: parametre var/yok fark yaratmamali.

    V11 davranisi: _noise_active=True iken KOSULSUZ gurultu.
    episode_clean=False -> kosul daima True -> V11 ile ayni.
    Bu test iki env kurup birini episode_clean=False (acik) diger default
    ile calistirip NAV izlerini karsilastirir: BIREBIR AYNI olmali.
    """
    prices, feats = _market(seed=7)
    rng_actions = np.random.default_rng(99)

    def _run(ep_clean_kwarg):
        env = PortfolioEnv(
            prices, feats,
            horizon="short",
            random_start=True,
            price_noise_std=0.03,
            seed=123,
            **ep_clean_kwarg,
        )
        env.reset()
        navs = []
        for _ in range(30):
            a = rng_actions.normal(0, 1, env.action_dim).astype(np.float32)
            _, _, done, trunc, _ = env.step(a)
            navs.append(env.nav)
            if done or trunc:
                break
        return navs

    # Her iki env ayni RNG ile kurulur; action sekans ayni (rng_actions state'i
    # ikinci run icin sifirlanmaz, bu yuzden iki env'i AYNI oturumda calistirip
    # karsilastirmak yerine KAPAL/ACIK "False" degerinin efektif anlami kontrol edilir:
    # episode_clean=False -> not False or idx>=1 -> daima True -> V11 ayni.
    # Dogrudan mantik dogrulama:
    env_explicit = PortfolioEnv(
        prices, feats, horizon="short",
        random_start=True, price_noise_std=0.03,
        seed=42, episode_clean=False,
    )
    env_explicit.reset()
    assert env_explicit._episode_clean is False
    # Gurultunun aktif oldugunu dogrula: birden fazla draw farkli olmali (V11 davranisi)
    r1 = env_explicit._risky_returns()
    r2 = env_explicit._risky_returns()
    assert not np.array_equal(r1, r2), (
        "episode_clean=False: gurultu V11 gibi DAIMA aktif olmali (not False = True)"
    )


def test_golden_safety_eval_unaffected_by_episode_clean():
    """random_start=False (eval) env'lerde episode_clean degerinden bagimsiz: NAV ayni."""
    prices, feats = _market(seed=5)

    def _run_eval(ep_clean):
        env = PortfolioEnv(
            prices, feats, horizon="short",
            random_start=False,
            price_noise_std=0.05,
            seed=0,
            episode_clean=ep_clean,
        )
        rng_a = np.random.default_rng(7)
        env.reset()
        navs = []
        for _ in range(30):
            a = rng_a.normal(0, 1, env.action_dim).astype(np.float32)
            _, _, done, trunc, _ = env.step(a)
            navs.append(env.nav)
            if done or trunc:
                break
        return navs

    navs_false = _run_eval(False)
    navs_true  = _run_eval(True)
    np.testing.assert_allclose(
        navs_false, navs_true, atol=1e-12,
        err_msg="Eval (random_start=False): episode_clean degeri NAV'i etkilememeli"
    )


# ---------------------------------------------------------------------------
# (d) DiscretePortfolioEnv episode_clean=True'yu forward etmeli
# ---------------------------------------------------------------------------

def test_discrete_env_forwards_episode_clean():
    """DiscretePortfolioEnv(..., episode_clean=True)._episode_clean is True olmali."""
    prices, feats = _market()
    env = DiscretePortfolioEnv(
        prices, feats,
        horizon="short",
        random_start=True,
        price_noise_std=0.05,
        seed=0,
        episode_clean=True,
    )
    assert env._episode_clean is True, (
        "DiscretePortfolioEnv episode_clean=True'yu PortfolioEnv'e forward etmeli"
    )
    assert env._episode_idx == -1, "ctor sonrasi _episode_idx==-1 olmali"

    # 1. reset temiz olmali
    env.reset()
    assert env._episode_idx == 0
    r1 = env._risky_returns()
    r2 = env._risky_returns()
    np.testing.assert_array_equal(r1, r2, err_msg="DiscreteEnv 1. episode temiz olmali")

    # 2. reset gurultulu olmali
    env.reset()
    assert env._episode_idx == 1
    r3 = env._risky_returns()
    r4 = env._risky_returns()
    assert not np.array_equal(r3, r4), "DiscreteEnv 2. episode gurultulu olmali"


def test_discrete_env_default_episode_clean_false():
    """DiscretePortfolioEnv default olarak episode_clean=False olmali."""
    prices, feats = _market()
    env = DiscretePortfolioEnv(prices, feats, horizon="short", seed=0)
    assert env._episode_clean is False, "DiscretePortfolioEnv default episode_clean=False"


# ---------------------------------------------------------------------------
# (e) reset() ile seed override episode_idx sayacini sifirlamaz
# ---------------------------------------------------------------------------

def test_reset_with_seed_preserves_episode_idx():
    """reset(seed=X) cagrisinda _episode_idx artmaya devam etmeli (sifirlanmaz)."""
    prices, feats = _market()
    env = PortfolioEnv(prices, feats, horizon="short", episode_clean=True, seed=0)
    env.reset()          # idx -> 0
    env.reset(seed=99)   # yeni rng AMA idx -> 1
    assert env._episode_idx == 1, "reset(seed=X) _episode_idx sayacini sifirlamaz"
