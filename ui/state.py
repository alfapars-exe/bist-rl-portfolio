"""Session-state varsayilanlari ve anahtar yardimcilari (app.py'den tasindi, P5)."""
from __future__ import annotations

import streamlit as st

from data import BIST28
from env.portfolio_env import HORIZON_PRESETS

ASSET_NAMES = BIST28 + ["CASH"]


def _init_state():
    """st.session_state varsayılanları — ilk yüklemede."""
    defaults = {
        "data_loaded": False,
        "prices": None,
        "px_tr": None,
        "px_te": None,
        "feats_tr": None,
        "feats_te": None,
        "scaler": None,
        "trained_agents": {},     # {(algo, horizon, adaptive): (agent, curve)}
        "test_traces": {},        # aynı anahtar: trajectory listesi
        "baselines": None,        # dict(name -> backtest dict)
        "step_idx": 0,
        "playing": False,
        "selected_algo": "DQN",
        "horizon": "medium",
        "adaptive": True,
        "initial_capital": 100_000.0,
        "train_delay": 0.0,
        "reward_cfg": {},  # kullanıcı ayarları; boş ise env preset'leri kullanır
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _agent_key(algo: str, horizon: str, adaptive: bool) -> tuple:
    return (algo, horizon, bool(adaptive))


def env_rebalance_hint(horizon: str) -> int:
    return int(HORIZON_PRESETS[horizon]["rebalance"])
