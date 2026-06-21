"""Yeni davranis guard testleri (v10 revizyonu).

Bu dosya asagidaki yeni kontratları kilitler:
  1. Nakit faizi  : all-cash portfoy 252 adimda NAV ~ (1+cash_annual_rate) = ~1.40
  2. C1 re-base   : navs_aligned.csv'de NAV[0]=1 ve tum stratejiler ayni uzunlukta
  3. Maliyetli baseline: equal_weight(eta>0) < equal_weight(eta=0) (FinalNAV)
  4. cash_riskfree: default faizle NAV son > 1 (~1.40x); daily_rf=0 ile duz 1.0
  5. Vade gün-limiti: HORIZON_PRESETS her preset min_days/max_days/train_max_steps icerir
  6. metrics NaN-guard: sabit-getiri dizisinde sharpe/calmar NaN; normal dizide sayi
  7. PPO deterministik eval: act_eval ayni state -> ayni aksiyon (mu, sample degil)

Deterministik: seed=42, toy market RNG env-local (global np.random'a dokunmaz).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import HORIZON_PRESETS, EnvConfig, cash_daily_rate as _cash_daily_rate
from env.portfolio_env import PortfolioEnv
from utils.baselines import cash_riskfree, equal_weight
from utils.features import add_features
from utils.metrics import sharpe, sortino, calmar


# ------------------------------------------------------------------ #
# Ortak toy fixture                                                    #
# ------------------------------------------------------------------ #

def _toy_prices(seed: int = 42, n: int = 350, k: int = 4) -> pd.DataFrame:
    """Deterministik kucuk piyasa (yalniz test icin RNG; env/baseline RNG'siz)."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, size=(n, k)), axis=0)),
        index=idx,
        columns=[f"A{i}" for i in range(k)],
    )
    return prices


# ================================================================== #
#  1. Nakit faizi: all-cash portfoy 252 adimda NAV ~ 1.40            #
# ================================================================== #

class TestCashInterest:
    """Env nakit varligi artik gunluk risksiz faiz kazanir (v10)."""

    def test_all_cash_nav_approx_140_after_252_steps(self):
        """Tamamen nakit portfoy 252 adimda NAV = (1+cash_annual_rate) ~ 1.40."""
        prices = _toy_prices()
        feats = add_features(prices)
        daily_rf = _cash_daily_rate(EnvConfig.cash_annual_rate, EnvConfig.trading_days)
        env = PortfolioEnv(
            prices, feats, horizon="short", max_steps=252,
            random_start=False, seed=42,
            cash_daily_rate=daily_rf,
        )
        env.reset()
        N = env.N
        # Nakit logiti cok yuksek -> softmax ~ [0, 0, ..., 1] (nakit)
        cash_action = np.full(N, -10.0, dtype=np.float32)
        cash_action[-1] = 10.0

        for _ in range(252):
            _, _, done, trunc, _ = env.step(cash_action.copy())
            if done or trunc:
                break

        expected = (1.0 + daily_rf) ** 252
        assert abs(env.nav - expected) < 0.01, (
            f"Nakit NAV beklenen ~{expected:.4f}; bulunan {env.nav:.4f}")
        assert env.nav > 1.35, "Nakit portfoy 252 adimda en az 1.35x buyumeli"

    def test_all_cash_deterministic(self):
        """Nakit env deterministik: ayni aksiyon dizisi ayni NAV."""
        prices = _toy_prices()
        feats = add_features(prices)
        daily_rf = _cash_daily_rate(EnvConfig.cash_annual_rate)

        def _run():
            env = PortfolioEnv(prices, feats, horizon="short", max_steps=100,
                               random_start=False, seed=42, cash_daily_rate=daily_rf)
            env.reset()
            N = env.N
            cash_a = np.full(N, -10.0, dtype=np.float32)
            cash_a[-1] = 10.0
            for _ in range(80):
                _, _, done, trunc, _ = env.step(cash_a.copy())
                if done or trunc:
                    break
            return env.nav

        assert _run() == _run(), "Nakit env deterministik olmali"

    def test_cash_rate_zero_gives_flat_nav(self):
        """cash_daily_rate=0 -> nakit portfolio NAV sabit 1.0 (v10 oncesi davranis)."""
        prices = _toy_prices()
        feats = add_features(prices)
        env = PortfolioEnv(prices, feats, horizon="short", max_steps=80,
                           random_start=False, seed=42, cash_daily_rate=0.0)
        env.reset()
        N = env.N
        cash_a = np.full(N, -10.0, dtype=np.float32)
        cash_a[-1] = 10.0
        # tx_cost kucuk olacak ama log_r~0 -> NAV ~1.0
        for _ in range(60):
            _, _, done, trunc, _ = env.step(cash_a.copy())
            if done or trunc:
                break
        # daily_rf=0 -> nakit getirisi 0; NAV sadece log_r(~0) ile buyur
        # ilk adimda tx_cost nedeniyle hafif dusebilir; 1.0'a cok yakin olmali
        assert abs(env.nav - 1.0) < 0.05, (
            f"cash_daily_rate=0 ile NAV ~1.0 beklenir; {env.nav:.6f}")


# ================================================================== #
#  2. C1 re-base: navs_aligned.csv NAV[0]=1                          #
# ================================================================== #

class TestC1ReBase:
    """C1 adil-karsilastirma: navs_aligned dosyasinda tum stratejiler NAV[0]=1."""

    @pytest.fixture(scope="class")
    def navs(self):
        from pathlib import Path
        p = Path(__file__).resolve().parent.parent / "results" / "navs_aligned.csv"
        if not p.exists():
            pytest.skip("results/navs_aligned.csv yok — once main.py calistir")
        return pd.read_csv(p, index_col=0)

    def test_all_navs_start_at_one(self, navs):
        """Tum stratejilerin ilk NAV degeri 1.0 olmali (C1 re-base)."""
        first_row = navs.iloc[0]
        np.testing.assert_allclose(
            first_row.values, 1.0, atol=1e-9,
            err_msg=f"NAV[0] != 1: {dict(first_row)}")

    def test_all_navs_same_length(self, navs):
        """Tum stratejiler ayni satir sayisina sahip olmali (ortak pencere)."""
        col_lengths = {col: navs[col].notna().sum() for col in navs.columns}
        lengths = list(col_lengths.values())
        assert len(set(lengths)) == 1, (
            f"NAV uzunluklari farkli (adil pencere ihlali): {col_lengths}")

    def test_final_nav_equals_window_growth(self, navs):
        """FinalNAV = navs_aligned'in son degeri (pencere buyumesi)."""
        from pathlib import Path
        mp = Path(__file__).resolve().parent.parent / "results" / "metrics.csv"
        if not mp.exists():
            pytest.skip("results/metrics.csv yok")
        metrics = pd.read_csv(mp, index_col=0)
        # navs_aligned sutunlariyla kesisen stratejiler
        common = [c for c in navs.columns if c in metrics.index]
        for strat in common:
            nav_final = float(navs[strat].iloc[-1])
            metrics_final = float(metrics.loc[strat, "FinalNAV"])
            assert abs(nav_final - metrics_final) < 1e-4, (
                f"{strat}: navs_aligned son={nav_final:.6f} vs metrics FinalNAV={metrics_final:.6f}")


# ================================================================== #
#  3. Maliyetli baseline: eta>0 < eta=0                              #
# ================================================================== #

class TestCostlyBaseline:
    """Baseline'lar artik RL ile simetrik islem maliyeti dusuyor (C2)."""

    @pytest.fixture(scope="class")
    def prices(self):
        return _toy_prices(n=300, k=6)

    def test_equal_weight_costly_less_than_free(self, prices):
        """equal_weight(eta=0.001).FinalNAV < equal_weight(eta=0.0).FinalNAV."""
        d_cost = equal_weight(prices, eta=0.001)
        d_free = equal_weight(prices, eta=0.0)
        assert d_cost["nav"][-1] < d_free["nav"][-1], (
            f"Maliyetli baseline >= maliyetsiz: {d_cost['nav'][-1]:.6f} >= {d_free['nav'][-1]:.6f}")

    def test_cost_proportional_to_eta(self, prices):
        """Daha yuksek eta -> daha dusuk FinalNAV (monotonik maliyet)."""
        navs = [equal_weight(prices, eta=e)["nav"][-1] for e in [0.0, 0.001, 0.005, 0.01]]
        for i in range(len(navs) - 1):
            assert navs[i] >= navs[i + 1], (
                f"eta artinca FinalNAV artmamali: eta={[0.0,0.001,0.005,0.01]} -> navs={navs}")

    def test_default_eta_is_medium_preset(self):
        """DEFAULT_ETA == HORIZON_PRESETS['medium']['eta'] (ortak kaynak)."""
        from utils.baselines import DEFAULT_ETA
        assert abs(DEFAULT_ETA - HORIZON_PRESETS["medium"]["eta"]) < 1e-12, (
            f"DEFAULT_ETA={DEFAULT_ETA} != medium eta={HORIZON_PRESETS['medium']['eta']}")


# ================================================================== #
#  4. cash_riskfree: default faizli NAV > 1; daily_rf=0 -> duz 1.0  #
# ================================================================== #

class TestCashRiskFreeBaseline:
    """cash_riskfree baseline v10: default faiz RL ortamiyla simetrik."""

    @pytest.fixture(scope="class")
    def prices(self):
        return _toy_prices(n=300, k=5)

    def test_default_rf_nav_approximately_140_per_year(self, prices):
        """Default daily_rf ~ %40/yil -> 252 gunluk verinin son NAV ~ 1.40 olmali."""
        d = cash_riskfree(prices)
        # 300 gun / 252 ~ 1.19 yil -> NAV = 1.40^1.19 ~ 1.49
        # Kesin deger: (1+dr)^300
        dr = _cash_daily_rate(EnvConfig.cash_annual_rate, EnvConfig.trading_days)
        expected = (1.0 + dr) ** len(prices)
        np.testing.assert_allclose(d["nav"][-1], expected, rtol=1e-9,
                                   err_msg="cash_riskfree default faiz NAV hatali")
        assert d["nav"][-1] > 1.0, "Pozitif faizde NAV > 1 olmali"

    def test_zero_rf_flat_nav(self, prices):
        """daily_rf=0.0 -> NAV tamamen duz 1.0."""
        d = cash_riskfree(prices, daily_rf=0.0)
        np.testing.assert_allclose(d["nav"], 1.0, atol=1e-12,
                                   err_msg="daily_rf=0 ile NAV 1.0 olmali")

    def test_cash_riskfree_monotone_increasing(self, prices):
        """Pozitif faizle her gun NAV bir onceki gunden buyuk ya da esit."""
        d = cash_riskfree(prices)
        nav = d["nav"]
        assert np.all(nav[1:] >= nav[:-1]), "Nakit NAV monoton artmali"

    def test_cash_riskfree_no_market_exposure(self, prices):
        """Nakit portfoy: tum agirliklar sifir (riskli varlik yok)."""
        d = cash_riskfree(prices)
        np.testing.assert_allclose(d["weights"], 0.0, atol=1e-12,
                                   err_msg="cash_riskfree agirliklar sifir olmali")

    def test_cash_riskfree_turnover_zero(self, prices):
        """Rebalans yok -> Turnover = 0."""
        from utils.metrics import summary
        d = cash_riskfree(prices)
        m = summary(d["nav"], d["rets"], d["weights"])
        assert abs(m["Turnover"]) < 1e-12

    def test_cash_riskfree_nan_metrics(self, prices):
        """Sifir vol -> Sharpe/Sortino/Calmar NaN (NaN-guard).
        CashRiskFree sabit getiri: std(rets)=0 ve MaxDD=0."""
        from utils.metrics import summary
        d = cash_riskfree(prices)
        m = summary(d["nav"], d["rets"], d["weights"])
        assert np.isnan(m["Sharpe"]),  f"Sharpe NaN beklenir; {m['Sharpe']}"
        assert np.isnan(m["Sortino"]), f"Sortino NaN beklenir; {m['Sortino']}"
        assert np.isnan(m["Calmar"]),  f"Calmar NaN beklenir; {m['Calmar']}"
        # CAGR, FinalNAV, Volatility sonlu olmali
        assert np.isfinite(m["CAGR"]),     f"CAGR sonlu olmali; {m['CAGR']}"
        assert np.isfinite(m["FinalNAV"]), f"FinalNAV sonlu olmali; {m['FinalNAV']}"


# ================================================================== #
#  5. Vade gün-limiti: HORIZON_PRESETS min/max/train_max_steps       #
# ================================================================== #

class TestHorizonPresets:
    """HORIZON_PRESETS her preset min_days/max_days/train_max_steps icermeli."""

    EXPECTED = {
        "short":  dict(min_days=1,  max_days=30,  train_max_steps=30),
        "medium": dict(min_days=30, max_days=90,  train_max_steps=90),
        "long":   dict(min_days=90, max_days=360, train_max_steps=360),
    }

    def test_all_presets_have_day_limit_keys(self):
        """Her preset min_days, max_days, train_max_steps anahtarlarina sahip olmali."""
        required = {"min_days", "max_days", "train_max_steps"}
        for name, preset in HORIZON_PRESETS.items():
            missing = required - set(preset.keys())
            assert not missing, (
                f"'{name}' preseti eksik anahtarlar: {missing}")

    @pytest.mark.parametrize("preset", ["short", "medium", "long"])
    def test_preset_day_limits_correct(self, preset):
        """Preset deger kanonik tanimlara uygun olmali."""
        p = HORIZON_PRESETS[preset]
        exp = self.EXPECTED[preset]
        for key, val in exp.items():
            assert p[key] == val, (
                f"'{preset}' preset['{key}']: beklenen {val}, bulunan {p[key]}")

    def test_train_max_steps_le_max_days(self):
        """train_max_steps <= max_days (egitim test penceresini asmamali)."""
        for name, p in HORIZON_PRESETS.items():
            assert p["train_max_steps"] <= p["max_days"], (
                f"'{name}': train_max_steps={p['train_max_steps']} > max_days={p['max_days']}")

    def test_min_days_lt_max_days(self):
        """min_days < max_days (pencere kisitlari tutarli)."""
        for name, p in HORIZON_PRESETS.items():
            assert p["min_days"] < p["max_days"], (
                f"'{name}': min_days={p['min_days']} >= max_days={p['max_days']}")


# ================================================================== #
#  6. metrics NaN-guard                                               #
# ================================================================== #

class TestMetricsNanGuard:
    """Sifir-vol/sifir-MaxDD girdi -> Sharpe/Sortino/Calmar NaN; normal -> sayi."""

    def test_sharpe_nan_on_constant_returns(self):
        """Sabit getiri (std=0) -> Sharpe NaN."""
        rets = np.full(300, 0.001)
        assert np.isnan(sharpe(rets)), "Sabit getiri -> Sharpe NaN beklenir"

    def test_sortino_nan_on_constant_returns(self):
        """Sabit pozitif getiri (downside=0) -> Sortino NaN."""
        rets = np.full(300, 0.001)
        assert np.isnan(sortino(rets)), "Sabit getiri -> Sortino NaN beklenir"

    def test_calmar_nan_on_zero_maxdd(self):
        """Monoton artan NAV (MaxDD=0) -> Calmar NaN."""
        nav = np.cumprod(1 + np.full(300, 0.001))
        assert np.isnan(calmar(nav)), "Sifir MaxDD -> Calmar NaN beklenir"

    def test_sharpe_finite_on_normal_returns(self):
        """Normal dagilimlı getiri -> Sharpe sonlu sayi."""
        rets = np.random.default_rng(42).normal(0.0005, 0.015, 300)
        result = sharpe(rets)
        assert np.isfinite(result), f"Normal getiri -> Sharpe sonlu beklenir; {result}"

    def test_calmar_finite_on_drawdown_path(self):
        """Cekilis iceren NAV -> Calmar sonlu sayi."""
        rets = np.random.default_rng(42).normal(0.0003, 0.012, 300)
        nav = np.cumprod(1 + rets)
        result = calmar(nav)
        assert np.isfinite(result), f"Cekilis var -> Calmar sonlu beklenir; {result}"

    def test_nan_only_affects_zero_vol_strategies(self):
        """NaN yalniz sifir-vol stratejide (CashRiskFree); diger stratejiler etkilenmez."""
        # Sabit getiri -> NaN
        flat_rets = np.full(252, _cash_daily_rate(0.40))
        assert np.isnan(sharpe(flat_rets)), "Sabit -> NaN"
        # Karisimlı getiri -> sonlu
        mixed_rets = np.random.default_rng(7).normal(0.001, 0.01, 252)
        assert np.isfinite(sharpe(mixed_rets)), "Normal -> sonlu"


# ================================================================== #
#  7. PPO deterministik eval (mu, sample degil)                      #
# ================================================================== #

class TestPPODeterministicEval:
    """PPO act_eval mu (ortalama) dondurur; tekrar cagrilarda ayni sonuc."""

    @pytest.fixture(scope="class")
    def agent(self):
        from agents.ppo import PPOAgent
        return PPOAgent(state_dim=20, action_dim=5, seed=42)

    def test_same_state_same_action(self, agent):
        """Ayni state -> act_eval her seferinde ayni aksiyon."""
        rng = np.random.default_rng(99)
        s = rng.normal(size=20).astype(np.float32)
        a1 = agent.act_eval(s)
        a2 = agent.act_eval(s)
        np.testing.assert_array_equal(a1, a2,
                                      err_msg="PPO act_eval deterministik olmali (mu)")

    def test_different_states_different_actions(self, agent):
        """Farkli state'ler farkli aksiyon uretmeli (politika bilgi tasir)."""
        rng = np.random.default_rng(0)
        s1 = rng.normal(size=20).astype(np.float32)
        s2 = rng.normal(size=20).astype(np.float32)
        a1 = agent.act_eval(s1)
        a2 = agent.act_eval(s2)
        # Tum elementler esit degil (tesadufen esit olsa cok dusuk olasilik)
        assert not np.allclose(a1, a2), (
            "Farkli state'lerin ayni aksiyon vermesi beklenmiyor")

    def test_act_eval_repeated_calls_stable(self, agent):
        """10 tekrar cagri -> hep ayni sonuc (rastgele state)."""
        s = np.random.default_rng(7).normal(size=20).astype(np.float32)
        actions = [agent.act_eval(s) for _ in range(10)]
        for a in actions[1:]:
            np.testing.assert_array_equal(
                a, actions[0],
                err_msg="PPO act_eval 10 tekrar cagride tutarsiz")

    def test_act_eval_output_shape(self, agent):
        """act_eval ciktisi action_dim boyutunda olmali."""
        s = np.zeros(20, dtype=np.float32)
        a = agent.act_eval(s)
        assert a.shape == (5,), f"Beklenen (5,); bulunan {a.shape}"
        assert a.dtype == np.float32, f"float32 beklenir; {a.dtype}"
