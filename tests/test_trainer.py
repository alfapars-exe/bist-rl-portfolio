"""core.trainer.train generator'lari — telemetri sozlesmesi (Faz 3, H1).

CLI (train.py) ve UI (app.py) bu generator'lari paylasir. Burada her ajan tipi
icin bir iterasyon kosturup yield edilen telemetri dict'inin gerekli alanlari
tasidigini dogruluyoruz (UI/CLI tuketicilerinin bekledigi anahtarlar).
"""
import numpy as np
import pandas as pd
import pytest

from agents import DQNAgent, PPOAgent, SACAgent
from env.portfolio_env import DiscretePortfolioEnv, PortfolioEnv
from utils.features import add_features
from core.trainer import train


def _market(seed=0, n=180, k=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0)),
        index=idx, columns=[f"A{i}" for i in range(k)])
    return prices, add_features(prices)


def test_train_dqn_telemetry():
    prices, feats = _market()
    env = DiscretePortfolioEnv(prices, feats, horizon="short", max_steps=40)
    agent = DQNAgent(env.state_dim, env.n_discrete, seed=0)
    rec = next(train(agent, env, n_iters=1))
    assert rec["algo"] == "DQN"
    for k in ("iter", "episode", "reward", "train_nav", "nav", "gain",
              "eps", "loss", "success", "actions", "agent", "env"):
        assert k in rec


def test_train_ppo_telemetry():
    prices, feats = _market()
    env = PortfolioEnv(prices, feats, horizon="short", max_steps=10_000)
    agent = PPOAgent(env.state_dim, env.action_dim, seed=0)
    rec = next(train(agent, env, n_iters=1, rollout_len=60))
    assert rec["algo"] == "PPO"
    for k in ("update", "p_loss", "v_loss", "ent", "kl", "mean_nav",
              "reward", "loss", "success"):
        assert k in rec


def test_train_sac_telemetry():
    prices, feats = _market()
    env = PortfolioEnv(prices, feats, horizon="short", max_steps=60)
    agent = SACAgent(env.state_dim, env.action_dim, seed=0)
    rec = next(train(agent, env, n_iters=1))
    assert rec["algo"] == "SAC"
    for k in ("episode", "train_nav", "steps", "reward", "loss", "success"):
        assert k in rec


def test_train_unknown_agent_type_raises():
    with pytest.raises(TypeError):
        train(object(), None, n_iters=1)
