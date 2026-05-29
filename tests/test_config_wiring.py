"""config.py wiring testleri (Faz 2, H3).

Golden gate train.py yolunu kapsar; app.py (UI) yolunu burada kilitleriz:
bos hp ile app._make_agent, config dataclass degerlerini kullanmali. Ayrica
env'in HORIZON_PRESETS'i config ile AYNI nesne olmali (tek kaynak kaniti).
"""
from config import DQNConfig, PPOConfig, SACConfig, HORIZON_PRESETS


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
