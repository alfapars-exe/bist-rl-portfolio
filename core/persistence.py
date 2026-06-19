"""Egitilmis ajan kalici hale getirme — diske kaydet/yukle (PDF §11 sunum sarti).

Sorun: egitilmis ajan yalniz Streamlit session_state'te tutuluyordu; sayfa
yenilenince kayboluyordu. Bu modul ajanin AG AGIRLIKLARINI diske yazar/okur ki
sunumda yeniden egitmeden test edilebilsin.

Tasarim: jenerik. Ajanin nn.Module attribute'lari (DQN: q/q_target, PPO:
policy/value, SAC: pi/q1/q2/...) `vars(agent)` ile toplanir; ayri ajan-basi kod
gerekmez. Yukleme core.factory.build_agent ile ayni mimaride iskelet kurup
state_dict'leri ad'a gore geri yukler (cikarim icin ag agirliklari yeterli —
optimizer/replay buffer kaydedilmez).

N11: Dosya adi semasindan algo_{horizon}_{adaptive}.pt — ayni algoritmay
farkli vade/adaptive ile kaydedince birbirinin uzerine yazmaz.

DAVRANIS: golden-irrelevant. Yalniz kalicilik; egitim/odul/eval sayisal yoluna
dokunmaz.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from core.factory import build_agent

FORMAT = 1

# Tum kaydedilmis modellerin bulundugu dizin (proje koku / models/).
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def model_path(algo: str, horizon: str = "medium", adaptive: bool = True) -> Path:
    """N11: algo_{horizon}_{adaptive}.pt — vade+adaptive farklilastirir.

    Ornek: model_path("DQN", "short", False) -> models/DQN_short_False.pt
    Eski tek-dosya yolunun yerine gecer; farkli vade/adaptive birbirini ezmez.
    """
    suffix = f"{adaptive}".lower()   # true / false — tutarli, kucuk harf
    return MODELS_DIR / f"{algo}_{horizon}_{suffix}.pt"


def _action_dim(agent) -> int:
    """DQN n_actions, PPO/SAC action_dim — ajandan turet."""
    ad = getattr(agent, "action_dim", None)
    if ad is None:
        ad = getattr(agent, "n_actions")
    return int(ad)


def _modules(agent) -> dict:
    """Ajanin tum nn.Module attribute'larinin {ad: state_dict}'i."""
    return {name: m.state_dict()
            for name, m in vars(agent).items() if isinstance(m, nn.Module)}


def save_agent(agent, algo: str, path, *, horizon: str = "medium",
               adaptive: bool = True) -> str:
    """Ajanin ag agirliklarini + meta'yi `path`'e yazar; yolu doner."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format": FORMAT,
        "algo": algo,
        "horizon": horizon,
        "adaptive": bool(adaptive),
        "state_dim": int(agent.state_dim),
        "action_dim": _action_dim(agent),
        "modules": _modules(agent),
    }, path)
    return str(path)


def load_agent(path):
    """Kaydedilmis ajani yukler. (agent, meta) doner; meta: algo/horizon/adaptive.

    build_agent ile ayni mimaride iskelet kurulur (config default hidden=(256,128)
    egitimdekiyle ayni), sonra state_dict'ler ad'a gore yuklenir.
    """
    # weights_only=True (guvenli unpickler): checkpoint yalniz metadata (str/int/bool)
    # + tensor state_dict'leri icerir; rastgele kod calistirma riski yok (SonarCloud S5042).
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    agent = build_agent(ckpt["algo"], int(ckpt["state_dim"]), int(ckpt["action_dim"]))
    for name, sd in ckpt["modules"].items():
        module = getattr(agent, name, None)
        if isinstance(module, nn.Module):
            module.load_state_dict(sd)
    meta = {"algo": ckpt["algo"],
            "horizon": ckpt.get("horizon", "medium"),
            "adaptive": bool(ckpt.get("adaptive", True))}
    return agent, meta
