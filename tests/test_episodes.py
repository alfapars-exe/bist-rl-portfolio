"""Gurultu-artirimli coklu-episode + NaN-guvenligi testleri (torch'suz, v12-uyumlu).

(1) NaN-guvenligi: data._sanitize_prices NaN'i temizler; env._risky_returns'teki
    np.nan_to_num, env.prices'a (ctor validasyonundan SONRA) enjekte edilen tek bir
    NaN getiriyi 'hareket yok' (0) sayar -> NAV sonlu kalir (eski hata: 0*nan=nan
    NAV'i tum episod boyunca zehirlerdi; HF Space'te gozlemlendi).
(2) Gurultu-artirimli episode mekanizmasi: make_noisy_prices determinizmi +
    evaluate_noise_episodes'in sirayla tam-aralik episode kosturmasi.
Random (torch'suz) ajan; ajan-agnostik act_eval yolu test edilir.
"""
import numpy as np
import pandas as pd

from core.episodes import (
    evaluate_noise_episodes, make_noisy_prices, summarize_episodes,
)
from data import _sanitize_prices
from env.portfolio_env import DiscretePortfolioEnv
from utils.features import TrainScaler, add_features


def _market(seed=0, n=400, k=6):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2015-01-01", periods=n)
    px = pd.DataFrame(100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (n, k)), axis=0)),
                      index=idx, columns=[f"A{i}" for i in range(k)])
    return px


class _RandomAgent:
    def __init__(self, n=6, seed=0):
        self.rng = np.random.default_rng(seed)
        self.n = n

    def act_eval(self, s):
        return int(self.rng.integers(0, self.n))


def _eval_env(px, seed=42, **kw):
    feats = TrainScaler().fit(add_features(px)).transform(add_features(px))
    return DiscretePortfolioEnv(px, feats, horizon="short", adaptive=True,
                                max_steps=10_000, random_start=False, seed=seed, **kw)


# ---------------------------------------------------------------- NaN-guvenligi
def test_sanitize_prices_cleans_nan():
    """data._sanitize_prices: ortadaki NaN ffill/bfill ile temizlenir (NaN kalmaz)."""
    px = _market()
    px_bad = px.copy()
    px_bad.iloc[200, 2] = np.nan
    clean = _sanitize_prices(px_bad)
    assert not clean.isna().any().any(), "_sanitize_prices NaN birakti"


def test_risky_returns_nan_to_num_keeps_nav_finite():
    """Regresyon: env.prices'a (validasyon SONRASI) enjekte edilen tek NaN,
    _risky_returns'te 0'a cevrilir -> getiri vektoru sonlu (eski 0*nan=nan hatasi)."""
    px = _market()
    env = _eval_env(px)
    env.reset()
    env.prices[env.t + 1, 2] = np.nan          # ctor validasyonundan SONRA enjekte
    r = env._risky_returns()
    assert np.isfinite(r).all(), "NaN getiri _risky_returns'te temizlenmedi"


def test_clean_data_risky_returns_unchanged():
    """no-op invariant: temiz veride nan_to_num davranisi degistirmemeli (golden ruhu).
    Riskli kisim = (p1-p0)/max(p0,1e-9); nakit slotu = v10 faiz (cash_period_rate)."""
    px = _market()
    env = _eval_env(px)
    r = env._risky_returns()
    p0, p1 = env.prices[env.t], env.prices[env.t + 1]
    risky = (p1 - p0) / np.maximum(p0, 1e-9)
    sessions = int(env.session_counts[env.t + 1])
    cash = (1.0 + env.cash_daily_rate) ** sessions - 1.0
    expected = np.concatenate([risky, [cash]]).astype(np.float32)
    np.testing.assert_allclose(r, expected, rtol=1e-6, atol=1e-9)


# ------------------------------------------------------------- make_noisy_prices
def test_make_noisy_prices_zero_is_identity():
    """noise_std=0 -> birebir kopya (identity; KORUNUR)."""
    px = _market()
    clone = make_noisy_prices(px, 0.0, seed=1)
    assert np.allclose(clone.values, px.values, atol=1e-6)


def test_make_noisy_prices_changes_path_but_keeps_start():
    """UNIFORM noise: baslangic korunur, yol farkli, degerler sonlu."""
    px = _market()
    noisy = make_noisy_prices(px, 0.02, seed=3)
    assert np.allclose(noisy.values[0], px.values[0])           # baslangic korunur
    assert not np.allclose(noisy.values, px.values)             # yol farkli
    assert np.isfinite(noisy.values).all()


def test_make_noisy_prices_uniform_bounded():
    """UNIFORM gurultu: log-getiri farki [-noise_std, +noise_std] araliginda
    olmali (Gauss'tan farkli: outlier yok, deterministik aralik).
    Log-getiri farki |delta_logret| <= noise_std + kucuk sayisal tolerans."""
    import numpy as np
    px = _market(seed=7)
    noise_std = 0.015
    noisy = make_noisy_prices(px, noise_std, seed=99)
    px_arr = px.to_numpy(dtype=np.float64)
    noisy_arr = noisy.to_numpy(dtype=np.float64)
    logret_orig = np.diff(np.log(np.maximum(px_arr, 1e-9)), axis=0)
    logret_noisy = np.diff(np.log(np.maximum(noisy_arr, 1e-9)), axis=0)
    delta = logret_noisy - logret_orig
    # Maksimum sapma noise_std'yi gecmemeli (kucuk tolerans: kumulatif birikim yok)
    assert np.max(np.abs(delta)) <= noise_std + 1e-9, (
        f"Uniform gurultu siniri asildi: max|delta|={np.max(np.abs(delta)):.6f} > {noise_std}"
    )


def test_make_noisy_prices_features_differ_from_original():
    """Noisy fiyattan uretilen feature'lar orijinalden FARKLI olmali
    (ajan gercekten farkli durum vektoru alir — anti-ezber dogrulama)."""
    from utils.features import add_features
    px = _market(seed=5)
    noisy = make_noisy_prices(px, 0.01, seed=11)
    feats_orig = add_features(px)
    feats_noisy = add_features(noisy)
    # En az bir feature anahtarinda degerler farkli olmali
    any_diff = any(
        not np.allclose(feats_orig[k].values, feats_noisy[k].values)
        for k in feats_orig
    )
    assert any_diff, "Noisy fiyattan hesaplanan feature'lar orijinalle ayni — anti-ezber saglanamadi"


# ------------------------------------------------------- evaluate_noise_episodes
def test_evaluate_noise_episodes_sequential_finite():
    px = _market()
    scaler = TrainScaler().fit(add_features(px))

    def make_env(episode, noise_std, seed):
        npx = make_noisy_prices(px, noise_std, seed=seed)
        feats = scaler.transform(add_features(npx))
        return DiscretePortfolioEnv(npx, feats, horizon="short", adaptive=True,
                                    max_steps=10_000, random_start=False, seed=seed)

    res = evaluate_noise_episodes(make_env, _RandomAgent(seed=0),
                                  n_episodes=4, noise_std=0.01, seed=42)
    assert len(res) == 4
    assert res[0]["noise_std"] == 0.0                          # episode 0 = orijinal
    assert all(r["noise_std"] == 0.01 for r in res[1:])
    assert all(np.isfinite(r["final_nav"]) for r in res)
    finals = {round(r["final_nav"], 6) for r in res}
    assert len(finals) > 1                                     # episode'lar farkli
    summ = summarize_episodes(res)
    assert summ["n_episodes"] == 4 and np.isfinite(summ["final_nav_mean"])


# ----------------------------------------------------------- force_price_noise
def test_force_price_noise_flag_golden_safe():
    px = _market()
    # default: eval'de gurultu KAPALI (golden korunur)
    e_def = _eval_env(px, price_noise_std=0.05)
    assert e_def._noise_active is False
    # force=True: eval'de gurultu ACIK (yeni episode yolu)
    e_force = _eval_env(px, price_noise_std=0.05, force_price_noise=True)
    assert e_force._noise_active is True


# ------------------------------------------- per-episode detay tablosu (trace)
def test_run_trace_loop_light_produces_detail_table():
    """Per-episode adim-adim detay tablosu: _run_trace_loop(light=True) -> trace ->
    step_rows_for_training ekrandaki tabloyu (Gün#/Tarih/Aksiyon/TL/P&L/Komisyon/turnover/
    Tutulan hisse) uretir. Satir sayisi = adim sayisi (tam aralik, sirali); DQN aksiyon
    adi dolu; light modda state/q_values atlanir (bellek)."""
    from ui.services import _compute_test_tl_snaps, _run_trace_loop
    from utils.portfolio_tl import step_rows_for_training
    px = _market()
    feats = TrainScaler().fit(add_features(px)).transform(add_features(px))
    env = DiscretePortfolioEnv(px, feats, horizon="short", adaptive=True,
                               max_steps=10_000, random_start=False, seed=42)
    trace = _run_trace_loop(env, _RandomAgent(seed=0), "DQN", light=True)
    assert len(trace) > 50
    assert trace[0]["state"] is None and trace[0]["q_values"] is None      # light: bellek
    assert {"nav", "weights_after", "prices_t", "reward_terms", "action_name"} <= set(trace[0])
    snaps = _compute_test_tl_snaps(trace, 100_000.0)
    df = step_rows_for_training(
        snaps, [t["date"] for t in trace],
        action_names=[t["action_name"] for t in trace],
        action_indices=[t["action_idx"] for t in trace],
        reward_terms_list=[t["reward_terms"] for t in trace],
        initial_capital=100_000.0)
    assert len(df) == len(trace)                                           # satir = adim
    for col in ["Gün #", "Tarih", "Aksiyon", "Nakit TL", "Portföy TL", "Adım P&L",
                "Kümülatif P&L", "Komisyon TL", "Δ turnover", "Tutulan hisse"]:
        assert col in df.columns
    assert df["Aksiyon"].iloc[0] != ""                                     # DQN aksiyon adi dolu
