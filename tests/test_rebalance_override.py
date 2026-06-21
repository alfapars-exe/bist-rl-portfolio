"""Rebalans frekansı override (UI parametrik) — opt-in, golden-güvenli regresyon kilidi.

env ctor `rebalance_freq`: None -> vade preset'i (CLI/golden bit-aynı); int -> override
(min 1 korumalı). Hem sürekli (PortfolioEnv) hem ayrık (DiscretePortfolioEnv/DQN) yolu.
UI yolu (_make_env -> build_env -> rebalance_freq) bu davranışa dayanır.
"""
import numpy as np
import pandas as pd

from config import HORIZON_PRESETS
from env.portfolio_env import PortfolioEnv, DiscretePortfolioEnv
from utils.features import add_features


def _toy(seed=0, n=200, k=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    logret = rng.normal(0.0003, 0.012, size=(n, k))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(logret, axis=0)),
                          index=idx, columns=[f"A{i}" for i in range(k)])
    return prices, add_features(prices)


def test_rebalance_none_uses_preset():
    """rebalance_freq verilmezse vade preset'i kullanılır (golden-güvenli default)."""
    prices, feats = _toy()
    for h in ("short", "medium", "long"):
        env = PortfolioEnv(prices, feats, horizon=h, adaptive=True, max_steps=50)
        assert env.rebalance_freq == HORIZON_PRESETS[h]["rebalance"]


def test_rebalance_override_applies_and_min_guard():
    """int override uygulanır; <1 değerler 1'e kenetlenir (boş-bölme/0-mod koruması)."""
    prices, feats = _toy()
    env = PortfolioEnv(prices, feats, horizon="short", adaptive=True,
                       max_steps=50, rebalance_freq=7)
    assert env.rebalance_freq == 7
    env0 = PortfolioEnv(prices, feats, horizon="short", adaptive=True,
                        max_steps=50, rebalance_freq=0)
    assert env0.rebalance_freq == 1


def test_rebalance_override_discrete_dqn():
    """DiscretePortfolioEnv (DQN) override'ı super().__init__ üzerinden threadler."""
    prices, feats = _toy()
    env = DiscretePortfolioEnv(prices, feats, horizon="long", adaptive=True,
                               max_steps=50, rebalance_freq=4)
    assert env.rebalance_freq == 4
    env_def = DiscretePortfolioEnv(prices, feats, horizon="long", adaptive=True,
                                   max_steps=50)
    assert env_def.rebalance_freq == HORIZON_PRESETS["long"]["rebalance"]
