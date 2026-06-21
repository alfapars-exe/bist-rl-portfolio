"""Guard: tek-kaynak config + doküman⟂config sayısal tutarlılığı (config-drift'i yakalar).

`docs/MIMARI_HAFIZA.md` §0 'derived:config' bloğundaki sayılar `config.py`'den türetilmeli;
STATE_DIM doc değerleri config'ten hesaplananla eşleşmeli. `config.py` leaf modüldür (yalnız
stdlib) → ağır bağımlılık (pyarrow/torch) YOK, her zaman yeşil olmalı. `test_config_wiring.py`
pipeline'ın config'i okuduğunu kanıtlar; bu test doc + config sayılarının paralel kaçmadığını kanıtlar.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config  # leaf modül (stdlib-only)

DOC = ROOT / "docs" / "MIMARI_HAFIZA.md"
N_ASSETS = 28  # BIST 28 evreni (KOZAA/KOZAL hariç); ağırlık vektörü = N_ASSETS + nakit = 29


def _derived_block() -> dict:
    txt = DOC.read_text(encoding="utf-8")
    m = re.search(r"<!--\s*derived:config.*?-->(.*?)<!--\s*/derived:config\s*-->", txt, re.S)
    assert m, "derived:config bloğu MIMARI_HAFIZA.md'de bulunamadı"
    vals = {}
    for line in m.group(1).split("\n"):
        mm = re.match(r"^\s*([a-z_]+)\s*=\s*([0-9.]+)\s*$", line)
        if mm:
            vals[mm.group(1)] = float(mm.group(2))
    return vals


def _state_dim(forecast: bool) -> int:
    f = len(config.FEATURES) + (1 if (forecast and config.ForecastConfig().enabled) else 0)
    macro = len(config.MacroConfig().features) if config.MacroConfig().enabled else 0
    weights = N_ASSETS + 1
    return N_ASSETS * f + weights + macro


def test_config_is_leaf_importable():
    assert config.SEED == 42


def test_horizon_presets_structure():
    keys = {"rebalance", "mom_window", "minvol_window", "eta", "lam", "tau", "gamma",
            "min_days", "max_days", "train_max_steps"}
    assert set(config.HORIZON_PRESETS) == {"short", "medium", "long"}
    for h, d in config.HORIZON_PRESETS.items():
        assert keys <= set(d), f"{h} preset eksik anahtar: {keys - set(d)}"


def test_doc_numbers_derive_from_config():
    d = _derived_block()
    env = config.EnvConfig()
    rew = config.RewardConfig()
    assert d["seed"] == config.SEED
    assert d["n_features"] == len(config.FEATURES)
    assert d["cash_annual_rate"] == env.cash_annual_rate
    assert d["bankruptcy_penalty"] == env.bankruptcy_penalty
    assert d["w_dsr"] == rew.w_dsr
    assert d["w_cvar"] == rew.w_cvar


def test_state_dim_doc_matches_config():
    d = _derived_block()
    assert d["state_dim_dqn"] == _state_dim(forecast=True), "STATE_DIM(DQN/SAC/TD3) doc⟂config drift"
    assert d["state_dim_ppo"] == _state_dim(forecast=False), "STATE_DIM(PPO) doc⟂config drift"
