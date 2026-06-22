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

Isimli kayit (UI-only): named_model_path(name) -> models/{guvenli_isim}.pt
Meta'da name + saved_at (ISO, saniye hassasiyeti) tutulur; geriye-uyumlu.

DAVRANIS: golden-irrelevant. Yalniz kalicilik; egitim/odul/eval sayisal yoluna
dokunmaz.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

from core.contracts import RunSpec
from core.factory import build_agent

FORMAT = 2

# Tum kaydedilmis modellerin bulundugu dizin (proje koku / models/).
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Kullanici tarafindan kaydedilen odul preset JSON dosyalari.
REWARD_PRESETS_DIR = Path(__file__).resolve().parent.parent / "reward_presets"


def reward_preset_path(name: str) -> Path:
    """Odul preset dosya yolu: reward_presets/{guvenli_isim}.json

    Sanitize: harf/rakam/_/- disini _ ile degistir; bos ise 'preset' kullanilir.
    """
    safe = re.sub(r"[^\w\-]", "_", name.strip()) or "preset"
    return REWARD_PRESETS_DIR / f"{safe}.json"


def save_reward_preset(reward_cfg: dict, name: str, description: str = "") -> str:
    """Odul preset'ini JSON'a yazar; yolu doner.

    Format: {name, description, saved_at, format:1, reward_cfg}.
    REWARD_PRESETS_DIR otomatik olusturulur.
    """
    import json
    path = reward_preset_path(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": name,
        "description": description,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "format": 1,
        "reward_cfg": dict(reward_cfg),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def load_reward_preset(name: str) -> "dict | None":
    """Odul preset'ini yukler; reward_cfg dict doner (yoksa None)."""
    import json
    path = reward_preset_path(name)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return dict(data.get("reward_cfg", {}))
    except Exception:
        return None


def list_reward_presets() -> "list[dict]":
    """Tum kayitli odul preset'lerini listeler; saved_at'e gore yeni->eski sirali.

    Her eleman: {path, name, description, saved_at, reward_cfg}.
    Bozuk JSON dosyalari sessizce atlanir.
    """
    import json
    results = []
    if not REWARD_PRESETS_DIR.exists():
        return results
    for fpath in REWARD_PRESETS_DIR.glob("*.json"):
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            results.append({
                "path": str(fpath),
                "name": data.get("name", fpath.stem),
                "description": data.get("description", ""),
                "saved_at": data.get("saved_at", ""),
                "reward_cfg": dict(data.get("reward_cfg", {})),
            })
        except Exception:
            continue
    results.sort(key=lambda x: x["saved_at"], reverse=True)
    return results


def model_path(algo: str, horizon: str | int = "medium", adaptive: bool = True) -> Path:
    """N11: algo_{horizon}_{adaptive}.pt — vade+adaptive farklilastirir.

    Ornek: model_path("DQN", "short", False) -> models/DQN_short_False.pt
    Eski tek-dosya yolunun yerine gecer; farkli vade/adaptive birbirini ezmez.
    CLI/golden bu fonksiyonu kullanir — degistirilmez.
    """
    suffix = f"{adaptive}".lower()
    slot = f"step{horizon}" if isinstance(horizon, int) else str(horizon)
    return MODELS_DIR / f"{algo}_{slot}_{suffix}.pt"


def named_model_path(name: str, saved_at: str = "") -> Path:
    """Kullanici-verilen isimle kayit yolu: models/{guvenli_isim}.pt

    Sanitize: harf/rakam/_/- disini _ ile degistir; bos ise 'model_{saved_at}'.
    saved_at yalnizca fallback icin kullanilir (isim bossa).
    """
    safe = re.sub(r"[^\w\-]", "_", name.strip()) if name.strip() else ""
    if not safe:
        ts = re.sub(r"[^\w\-]", "_", saved_at) if saved_at else "model"
        safe = f"model_{ts}"
    return MODELS_DIR / f"{safe}.pt"


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


def _hidden_from_modules(algo: str, modules: dict) -> tuple[int, int] | None:
    module_name = {"DQN": "q", "PPO": "policy", "SAC": "pi", "TD3": "actor"}.get(algo)
    state = modules.get(module_name, {})
    weights = [v for k, v in state.items() if k.endswith("weight") and getattr(v, "ndim", 0) == 2]
    if len(weights) < 2:
        return None
    return int(weights[0].shape[0]), int(weights[1].shape[0])


def _agent_config(agent, algo: str, modules: dict) -> dict:
    hidden = _hidden_from_modules(algo, modules)
    cfg = {"hidden": list(hidden)} if hidden else {}
    for key in ("gamma", "batch_size", "eps_decay", "target_update", "lam", "clip",
                "ent_coef", "n_epochs", "alpha", "tau", "policy_noise", "noise_clip",
                "policy_delay", "expl_noise"):
        if hasattr(agent, key):
            value = getattr(agent, key)
            if isinstance(value, (str, int, float, bool)):
                cfg[key] = value
    return cfg


def save_agent(agent, algo: str, path, *, horizon: str = "medium",
               adaptive: bool = True, name: str = "",
               saved_at: str | None = None, run_spec: RunSpec | dict | None = None) -> str:
    """Ajanin ag agirliklarini + meta'yi `path`'e yazar; yolu doner.

    name: kullanici-verilen model adi (bos olabilir — meta'da saklanir).
    saved_at: ISO datetime str (sn hassasiyeti); None ise simdi hesaplanir.
    """
    if saved_at is None:
        saved_at = datetime.now().isoformat(timespec="seconds")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    modules = _modules(agent)
    spec_dict = (run_spec.to_dict() if isinstance(run_spec, RunSpec)
                 else dict(run_spec or {}))
    torch.save({
        "format": FORMAT,
        "algo": algo,
        "horizon": horizon,
        "adaptive": bool(adaptive),
        "state_dim": int(agent.state_dim),
        "action_dim": _action_dim(agent),
        "name": name,
        "saved_at": saved_at,
        "modules": modules,
        "agent_config": _agent_config(agent, algo, modules),
        "run_spec": spec_dict,
    }, path)
    return str(path)


def load_agent(path):
    """Kaydedilmis ajani yukler. (agent, meta) doner; meta: algo/horizon/adaptive/name/saved_at.

    build_agent ile ayni mimaride iskelet kurulur (config default hidden=(256,128)
    egitimdekiyle ayni), sonra state_dict'ler ad'a gore yuklenir.
    Eski modellerde name/saved_at yoksa bos string doner (geriye-uyumlu, KeyError yok).
    """
    # weights_only=True (guvenli unpickler): checkpoint yalniz metadata (str/int/bool)
    # + tensor state_dict'leri icerir; rastgele kod calistirma riski yok (SonarCloud S5042).
    ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
    agent_cfg = dict(ckpt.get("agent_config", {}))
    if "hidden" not in agent_cfg:
        hidden = _hidden_from_modules(ckpt["algo"], ckpt["modules"])
        if hidden:
            agent_cfg["hidden"] = hidden
    agent = build_agent(ckpt["algo"], int(ckpt["state_dim"]), int(ckpt["action_dim"]), agent_cfg)
    for name, sd in ckpt["modules"].items():
        module = getattr(agent, name, None)
        if isinstance(module, nn.Module):
            module.load_state_dict(sd)
    meta = {
        "algo": ckpt["algo"],
        "horizon": ckpt.get("horizon", "medium"),
        "adaptive": bool(ckpt.get("adaptive", True)),
        "name": ckpt.get("name", ""),
        "saved_at": ckpt.get("saved_at", ""),
        "format": int(ckpt.get("format", 1)),
        "legacy": int(ckpt.get("format", 1)) < FORMAT,
        "agent_config": agent_cfg,
        "run_spec": ckpt.get("run_spec", {}),
    }
    return agent, meta


def read_meta(path) -> dict:
    """Ajani KURMADAN yalniz meta'yi okur (liste goruntuleme icin hizli yol).

    torch.load ile checkpoint yuklenir; 'modules' (buyuk tensor'lar) goz ardi edilir.
    Eski/meta'siz dosyada guvenli default doner — KeyError/exception yok.
    """
    try:
        ckpt = torch.load(Path(path), map_location="cpu", weights_only=True)
        return {
            "algo": ckpt.get("algo", ""),
            "horizon": ckpt.get("horizon", "medium"),
            "adaptive": bool(ckpt.get("adaptive", True)),
            "name": ckpt.get("name", ""),
            "saved_at": ckpt.get("saved_at", ""),
            "format": int(ckpt.get("format", 1)),
            "legacy": int(ckpt.get("format", 1)) < FORMAT,
            "run_spec": ckpt.get("run_spec", {}),
        }
    except Exception:
        return {"algo": "", "horizon": "medium", "adaptive": True, "name": "", "saved_at": "",
                "format": 0, "legacy": True, "run_spec": {}}
