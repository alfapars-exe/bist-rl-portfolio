"""core.rollout.evaluate ajan-agnostik calismali (Faz 3, H2).

DQN (ayrik) ve PPO/SAC (surekli) — hicbiri icin if/algo dallanmasi olmadan
ayni evaluate cagrilabilmeli ve gecerli bir backtest dict'i donmeli.
"""
import numpy as np
import pandas as pd

from agents import DQNAgent, PPOAgent, SACAgent
from env.portfolio_env import DiscretePortfolioEnv, PortfolioEnv
from utils.features import add_features
from core.rollout import evaluate


def _market(seed=0, n=200, k=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0)),
        index=idx, columns=[f"A{i}" for i in range(k)])
    return prices, add_features(prices)


def test_evaluate_discrete_agent():
    prices, feats = _market()
    env = DiscretePortfolioEnv(prices, feats, horizon="short", max_steps=50)
    agent = DQNAgent(env.state_dim, env.n_discrete, seed=0)
    out = evaluate(agent, env)
    assert {"nav", "rets", "weights", "dates", "reward_terms_history"} <= set(out)
    assert len(out["nav"]) > 0


def test_evaluate_continuous_agents():
    prices, feats = _market()
    for AgentCls in (PPOAgent, SACAgent):
        env = PortfolioEnv(prices, feats, horizon="short", max_steps=50)
        agent = AgentCls(env.state_dim, env.action_dim, seed=0)
        out = evaluate(agent, env)
        assert len(out["nav"]) > 0
        assert out["weights"].shape[1] == env.N
