"""Regression tests for the v12 N-session accounting contracts."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from agents import DQNAgent, PPOAgent
from core.persistence import load_agent, save_agent
from core.rollout import evaluate
from data import resample_to_step_days
from env.portfolio_env import PortfolioEnv
from utils.features import add_features


def _flat_market(n=100):
    idx = pd.bdate_range("2020-01-01", periods=n)
    return pd.DataFrame(100.0, index=idx, columns=["A", "B"])


def test_step_days_uses_session_end_and_records_partial_period():
    df = pd.DataFrame({"x": range(8)}, index=pd.bdate_range("2024-01-01", periods=8))
    out = resample_to_step_days(df, 3)
    assert out["x"].tolist() == [2, 5, 7]
    assert list(out.attrs["session_counts"]) == [3, 3, 2]


def test_holdings_drift_and_turnover_uses_drifted_weights():
    px = _flat_market()
    probe = PortfolioEnv(px, add_features(px), horizon="medium", max_steps=3,
                         rebalance_freq=1, cash_daily_rate=0.0)
    t = probe.t
    px.iloc[t + 1, 0] = 200.0
    env = PortfolioEnv(px, add_features(px), horizon="medium", max_steps=3,
                       rebalance_freq=1, cash_daily_rate=0.0)
    env.reset()
    equal_logits = np.array([10.0, 10.0, -10.0], dtype=np.float32)
    _, _, _, _, first = env.step(equal_logits)
    np.testing.assert_allclose(env.w[:2], [2 / 3, 1 / 3], atol=1e-6)
    _, _, _, _, second = env.step(equal_logits)
    expected_turn = abs(0.5 - 2 / 3) + abs(0.5 - 1 / 3)
    assert abs(second["turnover"] - expected_turn) < 1e-6
    assert first["target_weights"][0] == first["target_weights"][1]


def test_rollout_dates_are_return_valuation_dates():
    px = _flat_market()
    env = PortfolioEnv(px, add_features(px), max_steps=2, cash_daily_rate=0.0)
    start = env.t

    class Agent:
        def act_eval(self, state):
            return np.zeros(env.action_dim, dtype=np.float32)

    out = evaluate(Agent(), env)
    assert pd.Timestamp(out["dates"][0]) == px.index[start]
    assert out["nav"][0] == 1.0 and out["rets"][0] == 0.0
    assert len(out["target_weights"]) == len(out["nav"])


def test_custom_hidden_checkpoint_roundtrip(tmp_path):
    agent = DQNAgent(10, 3, hidden=(32, 16), seed=3)
    path = save_agent(agent, "DQN", tmp_path / "custom.pt")
    loaded, meta = load_agent(path)
    assert meta["format"] == 2 and not meta["legacy"]
    assert loaded.q.net[0].out_features == 32
    for a, b in zip(agent.q.state_dict().values(), loaded.q.state_dict().values()):
        assert torch.equal(a, b)


def test_ppo_truncation_bootstraps_but_stops_gae_carry():
    agent = PPOAgent(4, 2, gamma=0.9, lam=0.8, seed=1)
    s = np.zeros(4, dtype=np.float32)
    agent.remember(s, np.zeros(2), 1.0, False, 0.5, 0.0,
                   next_v=2.0, boundary=True, discount=0.9, period_length=1)
    adv, ret = agent.compute_gae(0.0)
    assert np.isclose(ret[0], 1.0 + 0.9 * 2.0)
