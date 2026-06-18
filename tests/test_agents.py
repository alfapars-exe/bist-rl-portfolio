"""BaseAgent arayuz sozlesmesi (H2).

Tum ajanlar BaseAgent'tan turemeli ve soyut metotlari (act_eval) uygulayarak
'concrete' (ornenklenebilir) kalmalidir. Yeni bir ajan eklenirse ya da
act_eval kaldirilirsa bu testler kirilir.
"""
from agents import BaseAgent, DQNAgent, PPOAgent, SACAgent, TD3Agent

ALL_AGENTS = (DQNAgent, PPOAgent, SACAgent, TD3Agent)


def test_all_agents_subclass_baseagent():
    for cls in ALL_AGENTS:
        assert issubclass(cls, BaseAgent)


def test_all_agents_are_concrete():
    # Soyut metot kaldiysa __abstractmethods__ dolu olur -> instantiation TypeError verir.
    for cls in ALL_AGENTS:
        assert not cls.__abstractmethods__, \
            f"{cls.__name__} soyut kaldi: {cls.__abstractmethods__}"


def test_all_agents_define_act_eval():
    for cls in ALL_AGENTS:
        assert callable(getattr(cls, "act_eval", None))


def test_td3_act_eval_is_deterministic_and_shaped():
    """TD3 act_eval: deterministik (keşif gürültüsü yok) + (action_dim,) float32."""
    import numpy as np
    sd, ad = 40, 29
    agent = TD3Agent(sd, ad, seed=0)
    s = np.zeros(sd, dtype=np.float32)
    a1 = agent.act_eval(s)
    a2 = agent.act_eval(s)
    assert a1.shape == (ad,) and a1.dtype == np.float32
    assert np.array_equal(a1, a2), "act_eval deterministik olmali (keşif gürültüsü kapalı)"
    # act(explore=True) ise gürültü ekler -> farklı olmalı
    assert not np.array_equal(agent.act(s, explore=True), a1)
