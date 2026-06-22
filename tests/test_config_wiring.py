"""config.py wiring testleri (Faz 2, H3).

Golden gate train.py yolunu kapsar; app.py (UI) yolunu burada kilitleriz:
bos hp ile app._make_agent, config dataclass degerlerini kullanmali. Ayrica
env'in HORIZON_PRESETS'i config ile AYNI nesne olmali (tek kaynak kaniti).

Ek guard testler (parametrik episode sayisi + price_noise_std UI kablolama):
  - test_build_env_price_noise_explicit: price_noise_std=X gecilince env'de X set
    olur; X>0 + random_start=True -> _noise_active True; X=0.0 -> False.
  - test_build_env_price_noise_default: param gecilmeyince EnvConfig.price_noise_std
    default'u korunur.
  - test_train_n_iters_exact_count: train(agent, env, n_iters=N) tam N kayit yield
    eder (parametrik episode sayisi sozlesmesi).
"""
import numpy as np
import pandas as pd

from config import DQNConfig, PPOConfig, SACConfig, TD3Config, HORIZON_PRESETS, EnvConfig
from core.factory import build_env
from utils.features import add_features


def _toy_market(seed=0, n=180, k=5):
    """Kucuk senkron piyasa verisi — test_price_noise._market() ile ayni stil."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2019-01-01", periods=n)
    prices = pd.DataFrame(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, (n, k)), axis=0)),
        index=idx, columns=[f"A{i}" for i in range(k)])
    return prices, add_features(prices)


# -------------------------------------------------------------------
# Guard (a) — build_env price_noise_std explicit gecme
# -------------------------------------------------------------------
def test_build_env_price_noise_explicit_active():
    """price_noise_std=X ve random_start=True -> env.price_noise_std == X ve _noise_active True."""
    prices, feats = _toy_market()
    env = build_env("SAC", prices, feats, horizon="short", max_steps=60,
                    random_start=True, seed=42, price_noise_std=0.03)
    assert abs(env.price_noise_std - 0.03) < 1e-12, (
        f"price_noise_std beklenen 0.03, bulunan {env.price_noise_std}")
    assert env._noise_active is True, (
        "_noise_active beklenen True (X>0 + random_start=True)")


def test_build_env_price_noise_zero_inactive():
    """price_noise_std=0.0 -> _noise_active False (gurultu kapaliyken ezberi gerekmez)."""
    prices, feats = _toy_market()
    env = build_env("SAC", prices, feats, horizon="short", max_steps=60,
                    random_start=True, seed=42, price_noise_std=0.0)
    assert abs(env.price_noise_std - 0.0) < 1e-12
    assert env._noise_active is False, (
        "_noise_active beklenen False (price_noise_std=0.0)")


def test_build_env_price_noise_eval_mode_inactive():
    """random_start=False (eval modu) -> X>0 olsa bile _noise_active False (golden korumasi)."""
    prices, feats = _toy_market()
    env = build_env("SAC", prices, feats, horizon="short", max_steps=60,
                    random_start=False, seed=42, price_noise_std=0.05)
    assert abs(env.price_noise_std - 0.05) < 1e-12
    assert env._noise_active is False, (
        "Eval modunda (random_start=False) _noise_active False olmali — golden guvenligi")


# -------------------------------------------------------------------
# Guard (b) — build_env price_noise_std gecilmeyince EnvConfig default
# -------------------------------------------------------------------
def test_build_env_price_noise_default_preserved():
    """price_noise_std parametresi gecilmeyince EnvConfig.price_noise_std default'u korunur."""
    prices, feats = _toy_market()
    env = build_env("SAC", prices, feats, horizon="short", max_steps=60,
                    random_start=True, seed=42)  # price_noise_std yok
    assert abs(env.price_noise_std - EnvConfig.price_noise_std) < 1e-12, (
        f"Default beklenen {EnvConfig.price_noise_std}, bulunan {env.price_noise_std}")


# -------------------------------------------------------------------
# Guard (c) — train(n_iters=N) tam N kayit yield eder
# -------------------------------------------------------------------
def test_train_n_iters_exact_count_dqn():
    """DQN: train(agent, env, n_iters=N) tam N telemetri kaydi yield etmeli."""
    from agents import DQNAgent
    from env.portfolio_env import DiscretePortfolioEnv
    from core.trainer import train

    prices, feats = _toy_market(seed=1)
    env = DiscretePortfolioEnv(prices, feats, horizon="short", max_steps=40, seed=42)
    agent = DQNAgent(env.state_dim, env.n_discrete, seed=42)
    N = 5
    records = list(train(agent, env, n_iters=N))
    assert len(records) == N, (
        f"DQN: n_iters={N} verildi ama {len(records)} kayit uretildi")


def test_train_n_iters_exact_count_sac():
    """SAC: train(agent, env, n_iters=N) tam N telemetri kaydi yield etmeli."""
    from agents import SACAgent
    from core.trainer import train

    prices, feats = _toy_market(seed=2)
    env = build_env("SAC", prices, feats, horizon="short", max_steps=60,
                    random_start=True, seed=42)
    agent = SACAgent(env.state_dim, env.action_dim, seed=42)
    N = 4
    records = list(train(agent, env, n_iters=N))
    assert len(records) == N, (
        f"SAC: n_iters={N} verildi ama {len(records)} kayit uretildi")


def test_env_horizon_presets_is_config_object():
    from env.portfolio_env import HORIZON_PRESETS as env_presets
    assert env_presets is HORIZON_PRESETS  # tek kaynak: ayni nesne


def test_make_agent_uses_config_defaults():
    import app  # modul seviyesinde yan etki yok (main() guard'li)

    dqn = app._make_agent("DQN", 169, 6, {})
    assert dqn.batch_size == DQNConfig.batch_size
    assert dqn.eps_decay == DQNConfig.eps_decay
    assert dqn.target_update == DQNConfig.target_update

    ppo = app._make_agent("PPO", 169, 29, {})
    assert ppo.n_epochs == PPOConfig.n_epochs
    assert ppo.batch_size == PPOConfig.batch_size
    assert abs(ppo.clip - PPOConfig.clip) < 1e-12

    sac = app._make_agent("SAC", 169, 29, {})
    assert sac.batch_size == SACConfig.batch_size
    assert abs(sac.tau - SACConfig.tau) < 1e-12
    assert abs(sac.alpha - SACConfig.alpha) < 1e-12

    td3 = app._make_agent("TD3", 169, 29, {})
    assert td3.batch_size == TD3Config.batch_size
    assert abs(td3.tau - TD3Config.tau) < 1e-12
    assert abs(td3.policy_noise - TD3Config.policy_noise) < 1e-12
    assert td3.policy_delay == TD3Config.policy_delay
