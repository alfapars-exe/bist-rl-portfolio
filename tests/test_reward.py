"""Guard testler — parametrik tur (v9) odul motoru + build_env forwarding + data.py split.

Kapsanan kontratlar:
  (a) Golden-guvenlik: w_gain=0, w_ruin_timing=0 (default) iken
      terms["gain_bonus"]==0.0, terms["ruin_timing_mult"]==1.0 ve
      iflas aninda efektif cezanin flat bankruptcy_penalty'ye esit oldugu dogrulanir.
  (b) Yeni terim aktif: w_gain=0.1 + nav>gain_floor -> gain_bonus lineer artar;
      w_ruin_timing=1.0 + erken iflas (step_frac kucuk) -> ruin_timing_mult > 1.
  (c) build_env param forwarding: reward_overrides={"w_dsr": X} env'in
      RewardEngine'inde w_dsr==X; override yokken RewardConfig.w_dsr default korunur.
      Ayni dogrulama w_cvar icin de tekrarlanir.
  (d) train_test_split: verilen split'e saygili; default DataConfig.train_end.

Determinizm: seed=42; RNG tuketimi stabil (test_env / test_config_wiring stili).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import DataConfig, RewardConfig
from data import train_test_split
from env.reward import AdaptiveRewardShaper, DifferentialSharpe, RewardEngine
from core.factory import build_env
from utils.features import add_features


# ------------------------------------------------------------------ #
# Ortak toy fixture (test_env ve test_config_wiring ile ayni stil)    #
# ------------------------------------------------------------------ #

def _toy_market(seed: int = 0, n: int = 300, k: int = 5):
    """Kucuk deterministik piyasa; env/config testleriyle ayni imza."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    logret = rng.normal(0.0003, 0.012, size=(n, k))
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(logret, axis=0)),
        index=idx,
        columns=[f"A{i}" for i in range(k)],
    )
    return prices, add_features(prices)


def _make_engine(
    *,
    w_gain: float = 0.0,
    gain_floor: float = 1.0,
    w_gain_speed: float = 0.0,
    w_ruin_timing: float = 0.0,
    bankruptcy_penalty: float = 10.0,
) -> RewardEngine:
    """Default shaper + DSR ile minimal RewardEngine; sadece test edilen paramlar degisir."""
    shaper = AdaptiveRewardShaper(
        eta_base=0.0015, lambda_base=0.25, tau_base=0.03,
        vol_target=0.02, turnover_target=0.05, ema_alpha=0.05,
    )
    dsharpe = DifferentialSharpe(eta=0.01, clip=5.0)
    return RewardEngine(
        shaper=shaper, dsharpe=dsharpe,
        w_dsr=0.05,
        bankruptcy_nav=0.01,
        bankruptcy_penalty=bankruptcy_penalty,
        w_gain=w_gain,
        gain_floor=gain_floor,
        w_gain_speed=w_gain_speed,
        w_ruin_timing=w_ruin_timing,
    )


# ================================================================== #
#  (a) Golden-guvenlik: opt-in terimler KAPALI (default)              #
# ================================================================== #

class TestGoldenSafety:
    """w_gain=0, w_ruin_timing=0 iken golden davranis degismemeli."""

    def test_gain_bonus_zero_when_w_gain_zero(self):
        """w_gain=0 iken nav ne olursa olsun gain_bonus==0.0 olmali."""
        engine = _make_engine(w_gain=0.0, gain_floor=1.0)
        # nav > gain_floor olsa bile
        out = engine.compute(
            gross_port_r=0.05, delta_w_l1=0.1,
            nav=2.5, peak=2.5,
            step_count=10, max_steps=100,
        )
        assert abs(out.terms["gain_bonus"]) < 1e-12, (
            f"w_gain=0 iken gain_bonus 0 olmali; bulundu: {out.terms['gain_bonus']}")

    def test_ruin_timing_mult_one_when_w_ruin_zero(self):
        """w_ruin_timing=0 iken ruin_timing_mult==1.0 olmali (flat ceza)."""
        engine = _make_engine(w_ruin_timing=0.0)
        out = engine.compute(
            gross_port_r=-0.99, delta_w_l1=0.0,
            nav=0.5, peak=1.0,
            step_count=1, max_steps=100,   # cok erken adim — timing etkili olmamali
        )
        assert abs(out.terms["ruin_timing_mult"] - 1.0) < 1e-12, (
            f"w_ruin_timing=0 iken ruin_timing_mult 1.0 olmali; "
            f"bulundu: {out.terms['ruin_timing_mult']}")

    def test_bankrupt_penalty_equals_flat_when_w_ruin_zero(self):
        """w_ruin_timing=0 ile iflas aninda bankrupt_penalty == flat bankruptcy_penalty."""
        flat_pen = 10.0
        engine = _make_engine(w_ruin_timing=0.0, bankruptcy_penalty=flat_pen)
        # nav < bankruptcy_nav (0.01) tetikleyecek sekilde
        out = engine.compute(
            gross_port_r=-0.99, delta_w_l1=0.0,
            nav=0.005, peak=1.0,
            step_count=2, max_steps=200,
        )
        assert out.terms["bankrupt"] is True, "Iflas kosulu tetiklenmeli (nav < 0.01)"
        assert abs(out.terms["bankruptcy_penalty"] - flat_pen) < 1e-9, (
            f"Flat ceza beklenen {flat_pen}, bulunan: {out.terms['bankruptcy_penalty']}")

    def test_gain_bonus_zero_nav_below_floor(self):
        """w_gain>0 olsa bile nav <= gain_floor iken gain_bonus==0 olmali."""
        engine = _make_engine(w_gain=0.5, gain_floor=1.0)
        out = engine.compute(
            gross_port_r=0.0, delta_w_l1=0.0,
            nav=0.8, peak=1.0,   # nav < gain_floor
            step_count=5, max_steps=100,
        )
        assert abs(out.terms["gain_bonus"]) < 1e-12, (
            f"nav <= gain_floor iken gain_bonus 0 olmali; "
            f"bulundu: {out.terms['gain_bonus']}")

    def test_both_opt_in_off_no_contribution(self):
        """Iki opt-in de kapali (default) -> total degismiyor; terimler sifir/bir."""
        engine_off = _make_engine(w_gain=0.0, w_ruin_timing=0.0)
        engine_on  = _make_engine(w_gain=0.0, w_ruin_timing=0.0)
        # Ayni parametreler; sadece anlamli iki adim
        kwargs = dict(gross_port_r=0.01, delta_w_l1=0.05,
                      nav=1.2, peak=1.3, step_count=10, max_steps=100)
        out = engine_off.compute(**kwargs)
        # gain_bonus=0 + ruin_timing_mult=1 kesinlikle
        assert abs(out.terms["gain_bonus"]) < 1e-12
        assert abs(out.terms["ruin_timing_mult"] - 1.0) < 1e-12


# ================================================================== #
#  (b) Yeni terim aktif: lineerlik + timing etkisi                    #
# ================================================================== #

class TestNewTermsActive:
    """Opt-in terimler acikken beklenen matematiksel davranis."""

    def test_gain_bonus_linear_in_nav(self):
        """w_gain=0.1, gain_floor=1.0: nav=2.0 -> ~0.1, nav=3.0 -> ~0.2 (w_gain_speed=0)."""
        engine = _make_engine(w_gain=0.1, gain_floor=1.0, w_gain_speed=0.0,
                              w_ruin_timing=0.0)
        # max_steps=1 ve step_count=0 -> step_frac=0; w_gain_speed=0 -> faktor=1
        def _gain(nav_val):
            e = _make_engine(w_gain=0.1, gain_floor=1.0, w_gain_speed=0.0)
            out = e.compute(gross_port_r=0.0, delta_w_l1=0.0,
                            nav=nav_val, peak=nav_val,
                            step_count=0, max_steps=100)
            return out.terms["gain_bonus"]

        # w_gain * (nav - gain_floor) = 0.1 * (2.0 - 1.0) = 0.1
        assert abs(_gain(2.0) - 0.1) < 1e-9, (
            f"nav=2.0 -> gain_bonus beklenen ~0.1, bulunan {_gain(2.0)}")
        # w_gain * (nav - gain_floor) = 0.1 * (3.0 - 1.0) = 0.2
        assert abs(_gain(3.0) - 0.2) < 1e-9, (
            f"nav=3.0 -> gain_bonus beklenen ~0.2, bulunan {_gain(3.0)}")

    def test_gain_bonus_increases_linearly_between_navs(self):
        """nav arttikca gain_bonus orantili artar (lineer kontrat)."""
        navs = [1.0, 1.5, 2.0, 2.5, 3.0]
        bonuses = []
        for nav_val in navs:
            e = _make_engine(w_gain=0.1, gain_floor=1.0, w_gain_speed=0.0)
            out = e.compute(gross_port_r=0.0, delta_w_l1=0.0,
                            nav=nav_val, peak=nav_val,
                            step_count=0, max_steps=100)
            bonuses.append(out.terms["gain_bonus"])
        # Artan monotonluk
        for i in range(len(bonuses) - 1):
            assert bonuses[i + 1] >= bonuses[i], (
                f"Gain bonus monoton artmali: {bonuses}")
        # nav=1.0 -> floor esiginde -> 0
        assert abs(bonuses[0]) < 1e-12

    def test_ruin_timing_mult_greater_than_one_early(self):
        """w_ruin_timing=1.0 + erken adim -> ruin_timing_mult > 1 (erken iflas daha sert)."""
        engine = _make_engine(w_ruin_timing=1.0)
        # step_frac = 1/100 = 0.01 (cok erken)
        out = engine.compute(
            gross_port_r=0.0, delta_w_l1=0.0,
            nav=1.0, peak=1.0,
            step_count=1, max_steps=100,
        )
        mult = out.terms["ruin_timing_mult"]
        assert mult > 1.0, (
            f"Erken adimda (step_frac~0.01) ruin_timing_mult > 1 olmali; "
            f"bulundu: {mult}")

    def test_ruin_timing_mult_approaches_one_late(self):
        """step_frac -> 1.0 (son adim) -> ruin_timing_mult -> 1+w_ruin*(1-1)=1.0."""
        engine = _make_engine(w_ruin_timing=2.0)
        out = engine.compute(
            gross_port_r=0.0, delta_w_l1=0.0,
            nav=1.0, peak=1.0,
            step_count=100, max_steps=100,  # son adim: step_frac=1.0
        )
        mult = out.terms["ruin_timing_mult"]
        assert abs(mult - 1.0) < 1e-9, (
            f"Son adimda ruin_timing_mult==1.0 olmali; bulundu: {mult}")

    def test_ruin_timing_early_bankruptcy_harsher(self):
        """Erken iflas cezasi gec iflasa gore daha sert olmali (w_ruin_timing>0)."""
        flat_pen = 10.0
        w_rt = 1.0
        # Erken iflas: step_count=5, max_steps=200 -> step_frac~0.025
        e_early = _make_engine(w_ruin_timing=w_rt, bankruptcy_penalty=flat_pen)
        out_early = e_early.compute(
            gross_port_r=-0.99, delta_w_l1=0.0,
            nav=0.005, peak=1.0,
            step_count=5, max_steps=200,
        )
        # Gec iflas: step_count=195, max_steps=200 -> step_frac~0.975
        e_late = _make_engine(w_ruin_timing=w_rt, bankruptcy_penalty=flat_pen)
        out_late = e_late.compute(
            gross_port_r=-0.99, delta_w_l1=0.0,
            nav=0.005, peak=1.0,
            step_count=195, max_steps=200,
        )
        assert out_early.terms["bankrupt"] is True
        assert out_late.terms["bankrupt"] is True
        assert out_early.terms["bankruptcy_penalty"] > out_late.terms["bankruptcy_penalty"], (
            "Erken iflas cezasi gec iflasa gore buyuk olmali "
            f"(erken={out_early.terms['bankruptcy_penalty']:.4f}, "
            f"gec={out_late.terms['bankruptcy_penalty']:.4f})")

    def test_gain_speed_factor_increases_early_bonus(self):
        """w_gain_speed>0 -> erken adimda bonus sonraki adima gore yuksek olmali."""
        def _bonus(step_count, max_steps):
            e = _make_engine(w_gain=0.1, gain_floor=1.0, w_gain_speed=1.0)
            out = e.compute(gross_port_r=0.0, delta_w_l1=0.0,
                            nav=2.0, peak=2.0,
                            step_count=step_count, max_steps=max_steps)
            return out.terms["gain_bonus"]

        bonus_early = _bonus(0, 100)    # step_frac=0 -> faktor = 1 + 1*(1-0) = 2
        bonus_late  = _bonus(100, 100)  # step_frac=1 -> faktor = 1 + 1*(1-1) = 1
        assert bonus_early > bonus_late, (
            f"Erken adim bonusu ({bonus_early}) gec adim bonusundan ({bonus_late}) buyuk olmali")


# ================================================================== #
#  (c) build_env param forwarding                                      #
# ================================================================== #

class TestBuildEnvParamForwarding:
    """reward_overrides anahtarlari env'in RewardEngine'ine dogru iletilmeli."""

    def setup_method(self):
        self.prices, self.feats = _toy_market(seed=42, n=200)

    def test_w_dsr_override_applied(self):
        """reward_overrides={"w_dsr": 0.2} -> env.reward.w_dsr == 0.2."""
        env = build_env(
            "SAC", self.prices, self.feats,
            horizon="short", max_steps=60,
            random_start=False, seed=42,
            reward_overrides={"w_dsr": 0.2},
        )
        assert abs(env.reward.w_dsr - 0.2) < 1e-12, (
            f"w_dsr override beklenen 0.2; bulunan {env.reward.w_dsr}")

    def test_w_dsr_default_preserved_without_override(self):
        """reward_overrides yok -> w_dsr == RewardConfig.w_dsr (default korunur)."""
        env = build_env(
            "SAC", self.prices, self.feats,
            horizon="short", max_steps=60,
            random_start=False, seed=42,
        )
        assert abs(env.reward.w_dsr - RewardConfig.w_dsr) < 1e-12, (
            f"Default w_dsr beklenen {RewardConfig.w_dsr}; bulunan {env.reward.w_dsr}")

    def test_cvar_alpha_override_applied(self):
        """reward_overrides={"cvar_alpha": 0.10} -> env.reward._cvar_mult farkli olmali."""
        from scipy.stats import norm as _norm
        alpha = 0.10
        env = build_env(
            "SAC", self.prices, self.feats,
            horizon="short", max_steps=60,
            random_start=False, seed=42,
            reward_overrides={"cvar_alpha": alpha},
        )
        expected_mult = float(_norm.pdf(_norm.ppf(alpha)) / max(alpha, 1e-9))
        assert abs(env.reward._cvar_mult - expected_mult) < 1e-9, (
            f"cvar_alpha=0.10 ile _cvar_mult beklenen {expected_mult:.6f}; "
            f"bulunan {env.reward._cvar_mult:.6f}")

    def test_cvar_alpha_default_preserved_without_override(self):
        """cvar_alpha override yok -> RewardConfig.cvar_alpha default'u yansimali."""
        from scipy.stats import norm as _norm
        alpha = RewardConfig.cvar_alpha
        env = build_env(
            "SAC", self.prices, self.feats,
            horizon="short", max_steps=60,
            random_start=False, seed=42,
        )
        expected_mult = float(_norm.pdf(_norm.ppf(alpha)) / max(alpha, 1e-9))
        assert abs(env.reward._cvar_mult - expected_mult) < 1e-9, (
            f"Default cvar_alpha ile _cvar_mult beklenen {expected_mult:.6f}; "
            f"bulunan {env.reward._cvar_mult:.6f}")

    def test_w_gain_override_applied(self):
        """reward_overrides={"w_gain": 0.3} -> env.reward.w_gain == 0.3."""
        env = build_env(
            "SAC", self.prices, self.feats,
            horizon="short", max_steps=60,
            random_start=False, seed=42,
            reward_overrides={"w_gain": 0.3},
        )
        assert abs(env.reward.w_gain - 0.3) < 1e-12, (
            f"w_gain override beklenen 0.3; bulunan {env.reward.w_gain}")

    def test_w_ruin_timing_override_applied(self):
        """reward_overrides={"w_ruin_timing": 0.5} -> env.reward.w_ruin_timing == 0.5."""
        env = build_env(
            "SAC", self.prices, self.feats,
            horizon="short", max_steps=60,
            random_start=False, seed=42,
            reward_overrides={"w_ruin_timing": 0.5},
        )
        assert abs(env.reward.w_ruin_timing - 0.5) < 1e-12, (
            f"w_ruin_timing override beklenen 0.5; bulunan {env.reward.w_ruin_timing}")

    def test_empty_overrides_uses_all_defaults(self):
        """reward_overrides={} (bos dict) -> tum parametreler config default'larinda."""
        env = build_env(
            "SAC", self.prices, self.feats,
            horizon="short", max_steps=60,
            random_start=False, seed=42,
            reward_overrides={},
        )
        assert abs(env.reward.w_dsr - RewardConfig.w_dsr) < 1e-12
        assert abs(env.reward.w_gain - 0.0) < 1e-12
        assert abs(env.reward.w_ruin_timing - 0.0) < 1e-12


# ================================================================== #
#  (d) train_test_split tarih parametresi                             #
# ================================================================== #

class TestTrainTestSplit:
    """train_test_split(df, split=X) verilen X'e saygili olmali."""

    def setup_method(self):
        idx = pd.bdate_range("2018-01-01", "2024-12-31")
        rng = np.random.default_rng(0)
        self.df = pd.DataFrame(
            rng.normal(0, 1, size=(len(idx), 3)),
            index=idx,
            columns=["A", "B", "C"],
        )

    def test_custom_split_date_respected(self):
        """split='2020-01-01' -> train < 2020-01-01, test >= 2020-01-01."""
        split_date = "2020-01-01"
        train, test = train_test_split(self.df, split=split_date)
        assert len(train) > 0, "Train bolumu bos olmamali"
        assert len(test) > 0, "Test bolumu bos olmamali"
        assert all(train.index < pd.Timestamp(split_date)), (
            "Train indeksi split tarihinden oncede kalmali (< split)")
        assert all(test.index >= pd.Timestamp(split_date)), (
            "Test indeksi split tarihinden itibaren basmali (>= split)")

    def test_train_before_test_no_overlap(self):
        """Train ve test kimeleri cakismamali (sizintisizlik)."""
        train, test = train_test_split(self.df, split="2021-06-01")
        train_set = set(train.index)
        test_set = set(test.index)
        assert train_set.isdisjoint(test_set), (
            f"Train ve test indeksleri cakisiyor! Ortak {len(train_set & test_set)} tarih.")

    def test_default_split_uses_dataconfig_train_end(self):
        """split=None -> DataConfig.train_end ('2022-01-01') kullanilmali."""
        expected_split = DataConfig().train_end
        train_default, test_default = train_test_split(self.df)
        train_explicit, test_explicit = train_test_split(self.df, split=expected_split)
        assert len(train_default) == len(train_explicit), (
            "Default split (None) ile explicit DataConfig.train_end esit olmali")
        assert len(test_default) == len(test_explicit), (
            "Default split (None) ile explicit DataConfig.train_end esit olmali")

    def test_split_preserves_all_rows(self):
        """train satir + test satir = toplam satir (veri kaybi yok)."""
        train, test = train_test_split(self.df, split="2021-01-01")
        assert len(train) + len(test) == len(self.df), (
            "Split sonrasi toplam satir sayisi korunmali")

    def test_early_split_small_train(self):
        """Erken split tarihi -> train kucuk, test buyuk olmali."""
        train, test = train_test_split(self.df, split="2019-01-01")
        assert len(train) < len(test), (
            "Erken split tarihinde train seti test setinden kucuk olmali")

    def test_late_split_large_train(self):
        """Gec split tarihi -> train buyuk, test kucuk olmali."""
        train, test = train_test_split(self.df, split="2024-01-01")
        assert len(train) > len(test), (
            "Gec split tarihinde train seti test setinden buyuk olmali")
