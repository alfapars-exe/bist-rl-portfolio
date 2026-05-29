"""BaseAgent arayuz sozlesmesi (H2).

Tum ajanlar BaseAgent'tan turemeli ve soyut metotlari (act_eval) uygulayarak
'concrete' (ornenklenebilir) kalmalidir. Yeni bir ajan eklenirse ya da
act_eval kaldirilirsa bu testler kirilir.
"""
from agents import BaseAgent, DQNAgent, PPOAgent, SACAgent

ALL_AGENTS = (DQNAgent, PPOAgent, SACAgent)


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
