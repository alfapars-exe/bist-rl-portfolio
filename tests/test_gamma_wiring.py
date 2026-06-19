"""γ (discount) hp üzerinden TÜM ajanlara ulaşmalı.

Regresyon: vade preset gamma'sı (Kısa 0.95 / Orta 0.99 / Uzun 0.995) önceden yalnız
TD3'e ulaşıyordu; DQN/PPO/SAC builder'ları gamma'yı geçmediğinden o ajanlar her zaman
config default'u (0.99) ile eğitiliyordu. Golden-güvenli: hp'de gamma yoksa her ajan
kendi config default'unu korur (CLI/golden hattı gamma geçmez)."""
from config import DQNConfig, PPOConfig, SACConfig, TD3Config
from core.factory import build_agent

S = 40  # toy state_dim


def test_gamma_override_reaches_all_agents():
    for algo, adim in [("DQN", 6), ("PPO", 29), ("SAC", 29), ("TD3", 29)]:
        ag = build_agent(algo, S, adim, {"gamma": 0.91})
        assert abs(ag.gamma - 0.91) < 1e-9, f"{algo}: gamma override ajana ulaşmadı"


def test_gamma_default_preserved_when_absent():
    # hp'de gamma yoksa config default korunur -> golden-güvenli
    assert abs(build_agent("DQN", S, 6, {}).gamma - DQNConfig.gamma) < 1e-9
    assert abs(build_agent("PPO", S, 29, {}).gamma - PPOConfig.gamma) < 1e-9
    assert abs(build_agent("SAC", S, 29, {}).gamma - SACConfig.gamma) < 1e-9
    assert abs(build_agent("TD3", S, 29, {}).gamma - TD3Config.gamma) < 1e-9
