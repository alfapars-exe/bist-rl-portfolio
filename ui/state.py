"""Session-state varsayilanlari ve anahtar yardimcilari (app.py'den tasindi, P5)."""
from __future__ import annotations

import streamlit as st

from config import DEFAULTS, DataConfig, EnvConfig, RewardConfig
from core.contracts import DataProvenance, RunSpec
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
        "step_days": DEFAULTS.step_days,
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
        "active_run_spec": None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def set_active_run_spec(algo: str, step_days: int, adaptive: bool, hp: dict) -> RunSpec:
    prices = st.session_state.get("prices")
    provenance = DataProvenance.from_value(
        getattr(prices, "attrs", {}).get("provenance") if prices is not None else None)
    feats = st.session_state.get("feats_tr") or {}
    spec = RunSpec(
        algo=algo, step_days=int(step_days), adaptive=bool(adaptive),
        reward_cfg=dict(st.session_state.get("reward_cfg", {}) or {}),
        agent_hp=dict(hp or {}), data_start=str(st.session_state.get("data_start", "")),
        data_split=str(st.session_state.get("data_split", "")),
        data_end=str(st.session_state.get("data_end", "")),
        feature_names=tuple(feats.keys()), provenance=provenance,
    )
    st.session_state.active_run_spec = spec.to_dict()
    return spec


def _agent_key(algo: str, step_days, adaptive: bool) -> tuple:
    if isinstance(step_days, str):
        return (algo, step_days, bool(adaptive))  # legacy checkpoint/UI key
    spec_data = st.session_state.get("active_run_spec")
    if spec_data:
        spec = RunSpec.from_dict(spec_data)
        if spec.algo == algo and spec.step_days == int(step_days) and spec.adaptive == bool(adaptive):
            return (algo, int(step_days), bool(adaptive), spec.fingerprint)
    return (algo, int(step_days), bool(adaptive), "unbound")


def env_rebalance_hint(step_days) -> int:
    return int(step_days) if not isinstance(step_days, str) else int(HORIZON_PRESETS[step_days]["rebalance"])
