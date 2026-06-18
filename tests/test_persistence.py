"""core.persistence save/load round-trip — PDF §11 model kaliciligi.

Kaydedilen ajan yuklenince ag agirliklari birebir geri gelmeli ve deterministik
act_eval ayni cikti vermeli. (Golden-irrelevant: yalniz kalicilik.)
"""
import numpy as np
import torch

from core.factory import build_agent
from core.persistence import load_agent, save_agent


def test_dqn_save_load_roundtrip(tmp_path):
    a = build_agent("DQN", 50, 6)
    # Agirliklari boz -> taze ajandan ayirt edilebilir olsun (gercek round-trip testi).
    with torch.no_grad():
        for p in a.q.parameters():
            p.add_(0.7)
    path = save_agent(a, "DQN", tmp_path / "DQN.pt", horizon="medium", adaptive=True)
    b, meta = load_agent(path)

    assert meta["algo"] == "DQN" and meta["horizon"] == "medium" and meta["adaptive"] is True
    # Yuklenen ag agirliklari kaydedilenle birebir esit.
    for t_a, t_b in zip(a.q.state_dict().values(), b.q.state_dict().values()):
        assert torch.equal(t_a, t_b)
    # Deterministik (greedy) act_eval ayni indeksi verir.
    s = np.zeros(50, dtype=np.float32)
    assert int(a.act_eval(s)) == int(b.act_eval(s))


def test_sac_save_load_roundtrip(tmp_path):
    a = build_agent("SAC", 40, 5)
    with torch.no_grad():
        for p in a.pi.parameters():
            p.add_(0.3)
    path = save_agent(a, "SAC", tmp_path / "SAC.pt")
    b, _ = load_agent(path)
    for t_a, t_b in zip(a.pi.state_dict().values(), b.pi.state_dict().values()):
        assert torch.equal(t_a, t_b)
    # SAC act_eval tanh-deterministik -> ayni vektor.
    s = np.zeros(40, dtype=np.float32)
    assert np.allclose(a.act_eval(s), b.act_eval(s))
