"""Session-state varsayilanlari ve anahtar yardimcilari (app.py'den tasindi, P5)."""
from __future__ import annotations

import streamlit as st

from config import DataConfig, EnvConfig, RewardConfig
from data import BIST28
from env.portfolio_env import HORIZON_PRESETS

ASSET_NAMES = BIST28 + ["CASH"]

_dc = DataConfig()
_rc = RewardConfig()


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
        "macro_tr": None, "macro_te": None,      # v6: makro rejim blogu (z-skorlu)
        "regime_tr": None, "regime_te": None,    # v6: ham regime (V7 amplify)
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
        "n_episodes": 12,                          # parametrik episode sayısı (UI)
        "price_noise_std": EnvConfig.price_noise_std,  # fiyat gürültüsü σ (UI kontrolü)
        "episode_clean": True,   # 1. iterasyon orijinal veri (anti-ezber); UI default açık
        "train_rebalance": None,  # rebalans frekansı override (None -> vade preset'i; golden-güvenli)
        # Tarih aralığı — DataConfig tek kaynak
        "data_start": _dc.start,
        "data_split": _dc.train_end,
        "data_end": _dc.end,
        # Genişletilmiş ödül parametreleri — RewardConfig tek kaynak
        # (6 açılan parametre; değer None/boş olunca env preset default'una düşer)
        "reward_w_dsr": _rc.w_dsr,
        "reward_w_cvar": _rc.w_cvar,
        "reward_dsr_eta": _rc.dsr_eta,
        "reward_cvar_alpha": _rc.cvar_alpha,
        "reward_regime_beta": _rc.regime_beta,
        "reward_cvar_amp": _rc.cvar_amp,
        # 4 opt-in deneysel terim (default 0 → davranış değişmez)
        "reward_w_gain": 0.0,
        "reward_gain_floor": 1.0,
        "reward_w_gain_speed": 0.0,
        "reward_w_ruin_timing": 0.0,
        # Adım granülerliği — "daily" no-op (golden-güvenli)
        "granularity": "daily",
        "granularity_n_points": None,   # resample sonrası satır sayısı (uyarı için)
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _agent_key(algo: str, horizon: str, adaptive: bool) -> tuple:
    return (algo, horizon, bool(adaptive))


def env_rebalance_hint(horizon: str) -> int:
    return int(HORIZON_PRESETS[horizon]["rebalance"])
