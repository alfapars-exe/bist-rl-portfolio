"""PortfolioEnv odul dekompozisyonu ve determinizm degismezleri.

Odul: total = log_return - tx_cost - drawdown_penalty - bankruptcy_penalty
Bu ozdeslik her adimda reward_terms icinde tutarli olmali; ayrica env, sabit
aksiyon dizisi altinda tam deterministik olmali (icinde torch/rastgelelik yok).
"""
import numpy as np
import pandas as pd

from config import FEATURES
from env.portfolio_env import PortfolioEnv, DiscretePortfolioEnv
from utils.features import add_features


def _toy_market(seed=0, n=300, k=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    logret = rng.normal(0.0003, 0.012, size=(n, k))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(logret, axis=0)),
                          index=idx, columns=[f"A{i}" for i in range(k)])
    return prices, add_features(prices)


def test_reward_decomposition_sums_to_total():
    prices, feats = _toy_market()
    env = PortfolioEnv(prices, feats, horizon="short", adaptive=True, max_steps=80)
    env.reset()
    rng = np.random.default_rng(1)
    for _ in range(60):
        a = rng.normal(0, 1, size=env.action_dim).astype(np.float32)
        _, r, done, trunc, info = env.step(a)
        rt = info["reward_terms"]
        # v2: odul ozdesligine Diferansiyel Sharpe terimi eklendi (+dsr_term = w_dsr*DSR,
        # online risk-ayar). Ozdeslik: log_return - tx_cost - drawdown - bankruptcy + dsr_term.
        recomputed = (rt["log_return"] - rt["tx_cost"]
                      - rt["drawdown_penalty"] - rt["bankruptcy_penalty"]
                      + rt["dsr_term"])
        assert abs(rt["total"] - recomputed) < 1e-9
        assert abs(float(r) - rt["total"]) < 1e-9
        if done or trunc:
            break


def test_env_is_deterministic_under_fixed_actions():
    prices, feats = _toy_market()
    actions = [np.full(6, 0.1, dtype=np.float32) for _ in range(40)]

    def run():
        env = PortfolioEnv(prices, feats, horizon="medium", adaptive=True, max_steps=50)
        env.reset()
        navs = []
        for a in actions:
            _, _, done, trunc, _ = env.step(a.copy())
            navs.append(env.nav)
            if done or trunc:
                break
        return navs

    assert run() == run()


def test_discrete_env_reward_invariant_holds():
    prices, feats = _toy_market(seed=2)
    env = DiscretePortfolioEnv(prices, feats, horizon="medium",
                               adaptive=False, max_steps=60)
    env.reset()
    for k in range(40):
        _, r, done, trunc, info = env.step(k % env.n_discrete)
        rt = info["reward_terms"]
        recomputed = (rt["log_return"] - rt["tx_cost"]
                      - rt["drawdown_penalty"] - rt["bankruptcy_penalty"]
                      + rt["dsr_term"])
        assert abs(rt["total"] - recomputed) < 1e-9
        if done or trunc:
            break


def test_state_dim_matches_formula():
    prices, feats = _toy_market(k=5)
    env = PortfolioEnv(prices, feats)
    # state_dim = F * N_assets + N   (N = N_assets + nakit)
    assert env.state_dim == env.F * env.N_assets + env.N
    assert env.F == len(FEATURES)
    assert env.state_dim == len(FEATURES) * 5 + 6


def test_random_start_within_bounds_and_seeded():
    """v2: rastgele-baslangic pencere sinirlari icinde + tohumlu (deterministik)."""
    prices, feats = _toy_market(n=400)

    def starts(seed):
        env = PortfolioEnv(prices, feats, horizon="short", max_steps=50,
                           random_start=True, seed=seed)
        out = []
        for _ in range(30):
            env.reset()
            assert env.window <= env.t < env.T - env.max_steps - 1   # sinir icinde
            out.append(env.t)
        return out

    s1 = starts(7)
    assert len(set(s1)) > 1          # gercekten cesitli pencereler
    assert s1 == starts(7)           # ayni tohum -> ayni dizilim (env-yerel RNG)


def test_per_agent_forecast_filtering():
    """v2: forecast yalniz DQN/SAC'a verilir; PPO almaz (ablation karari)."""
    import train
    feats = {"logret": None, "rsi": None, "forecast": None}
    assert "forecast" in train._feats_for(feats, "DQN")
    assert "forecast" in train._feats_for(feats, "SAC")
    assert "forecast" not in train._feats_for(feats, "PPO")


def test_differential_sharpe_first_zero_finite_clipped():
    """v2: DSR ilk cagride 0; sonrasi sonlu ve clip araliginda."""
    from env.portfolio_env import DifferentialSharpe
    ds = DifferentialSharpe(eta=0.04, clip=5.0)
    assert abs(ds.update(0.01)) < 1e-12                 # ilk cagri -> initialize, 0
    vals = [ds.update(r) for r in (0.02, -0.01, 0.03, 0.0, 0.015, -0.02)]
    assert all(np.isfinite(v) for v in vals)
    assert all(abs(v) <= 5.0 + 1e-9 for v in vals)      # clip uygulanir
