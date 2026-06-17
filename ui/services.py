"""UI veri/egitim/test servisleri (app.py'den tasindi, P5).

Streamlit session_state'i ile core katmani arasindaki kopru: veri yukleme,
env/ajan kurulumu (core.factory'ye delege), canli egitim generator'i ve test
trajectory yakalama.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from agents.base import SupportsQValues
from config import SEED, ForecastConfig, MacroConfig
from core.factory import build_agent, build_env
from core.trainer import train as train_loop
from data import align_macro, download_bist, download_macro, train_test_split
from env.portfolio_env import ACTION_NAMES
from utils.baselines import equal_weight
from utils.features import TrainScaler, add_features
from utils.macro import MacroScaler, add_macro_features
from utils.portfolio_tl import compute_tl_step


def _load_data():
    """Veri indir + z-score scaler'ı fit et."""
    with st.spinner("Veri indiriliyor / cache okunuyor ..."):
        prices = download_bist()
    feats_all_raw = add_features(prices)
    px_tr, px_te = train_test_split(prices)
    if ForecastConfig.enabled:                     # v2: forecast feature (train-only fit)
        from forecast.forecaster import build_forecast_feature
        feats_all_raw["forecast"] = build_forecast_feature(
            prices, px_tr, window=ForecastConfig.window, conv_ch=ForecastConfig.conv_ch,
            hidden=ForecastConfig.hidden, epochs=ForecastConfig.epochs,
            lr=ForecastConfig.lr, batch=ForecastConfig.batch, seed=SEED)
    feats_tr_raw = {k: v.loc[px_tr.index] for k, v in feats_all_raw.items()}
    feats_te_raw = {k: v.loc[px_te.index] for k, v in feats_all_raw.items()}

    scaler = TrainScaler().fit(feats_tr_raw)
    st.session_state.prices = prices
    st.session_state.px_tr = px_tr
    st.session_state.px_te = px_te
    st.session_state.feats_tr = scaler.transform(feats_tr_raw)
    st.session_state.feats_te = scaler.transform(feats_te_raw)
    st.session_state.scaler = scaler

    # v6: makro rejim (faiz/dolar/altin) — train-only z-score; ham regime ayri (V7).
    macro_tr = macro_te = regime_tr = regime_te = None
    if MacroConfig.enabled:
        mfeat = add_macro_features(align_macro(download_macro(), prices.index))
        regime_full = mfeat["regime"]
        macro_z = MacroScaler().fit(mfeat.loc[px_tr.index]).transform(mfeat)
        macro_tr = macro_z.loc[px_tr.index].to_numpy(np.float32)
        macro_te = macro_z.loc[px_te.index].to_numpy(np.float32)
        regime_tr = regime_full.loc[px_tr.index].to_numpy(np.float32)
        regime_te = regime_full.loc[px_te.index].to_numpy(np.float32)
    st.session_state.macro_tr = macro_tr
    st.session_state.macro_te = macro_te
    st.session_state.regime_tr = regime_tr
    st.session_state.regime_te = regime_te
    st.session_state.data_loaded = True


def _make_env(is_train: bool, algo: str, horizon: str, adaptive: bool, max_steps: int):
    """UI ortam kurulumu — session_state'i okuyup core.factory.build_env'e delege eder (P3)."""
    px_df = st.session_state.px_tr if is_train else st.session_state.px_te
    feats = st.session_state.feats_tr if is_train else st.session_state.feats_te
    macro = st.session_state.get("macro_tr" if is_train else "macro_te")
    regime = st.session_state.get("regime_tr" if is_train else "regime_te")
    return build_env(
        algo, px_df, feats, horizon=horizon, adaptive=adaptive, max_steps=max_steps,
        random_start=is_train, seed=SEED,          # v2: egitimde rastgele pencere, eval'de sabit
        reward_overrides=st.session_state.get("reward_cfg", {}) or {},
        macro=macro, regime=regime,                # v6: makro rejim blogu + ham regime
    )


def _make_agent(algo: str, state_dim: int, action_dim: int, hp: dict):
    """Shim — SOLID P3: tek dogruluk kaynagi core.factory.build_agent.
    (test_config_wiring app._make_agent'i cagirir; app.py bunu re-export eder.)"""
    return build_agent(algo, state_dim, action_dim, hp)


# =====================================================================
# Eğitim jeneratörü — canlı UI için episod başına yield
# =====================================================================
def train_generator(algo: str, horizon: str, adaptive: bool, hp: dict,
                    rollout_len: int = 400):
    """Episod/update başına bir telemetri kaydı yield eder. Sonsuz akış —
    tüketici (tab_train) 'Durdur' butonuyla keser."""
    max_steps = {"DQN": 252, "PPO": 10_000, "SAC": 1200}[algo]
    env = _make_env(True, algo, horizon, adaptive, max_steps=max_steps)
    action_dim = env.n_discrete if algo == "DQN" else env.action_dim
    agent = _make_agent(algo, env.state_dim, action_dim, hp)
    # Başarı kıyası için tren EW NAV'ı (core.trainer success'i bununla hesaplar)
    ew_tr = equal_weight(st.session_state.px_tr)["nav"]
    # Sonsuz akış (n_iters=None) — core.trainer.train ajan tipine göre dispatch eder;
    # tüketici (tab_train) 'Durdur' ile keser. Telemetri dict'i CLI ile ortaktır.
    yield from train_loop(agent, env, n_iters=None, rollout_len=rollout_len, ew_nav=ew_tr)


# =====================================================================
# Test dönemi — adım adım trajectory yakalama
# =====================================================================
def evaluate_with_trace(agent, algo: str, horizon: str, adaptive: bool) -> list:
    env = _make_env(False, algo, horizon, adaptive, max_steps=10_000)
    s, _ = env.reset()
    trace = []
    done = trunc = False
    # P5 (ISP): hasattr yoklamasi yerine resmi Protocol — ayni semantik, acik niyet.
    has_q = isinstance(agent, SupportsQValues)  # yalnizca DQN introspeksiyonu sunar
    while not (done or trunc):
        date = env.dates[env.t]
        weights_before = env.w.copy()
        state_snapshot = s.copy()

        q_vals = agent.q_values(s) if has_q else None
        a = agent.act_eval(s)                       # ajan-agnostik (BaseAgent.act_eval)
        if isinstance(a, (int, np.integer)):
            action_idx = int(a)
            action_name = (ACTION_NAMES[action_idx]
                           if 0 <= action_idx < len(ACTION_NAMES) else str(action_idx))
        else:
            action_idx = None
            action_name = f"{algo} (sürekli aksiyon)"
        s2, r, done, trunc, info = env.step(a)

        trace.append({
            "step": len(trace),
            "date": str(pd.Timestamp(date).date()),
            "state": state_snapshot,
            "action_idx": action_idx,
            "action_name": action_name,
            "q_values": q_vals,
            "weights_before": weights_before,
            "weights_after": env.w.copy(),
            "reward_terms": info["reward_terms"],
            "nav": float(env.nav),
            "prices_t": info["prices_t"].copy(),
        })
        s = s2
    return trace


def _compute_test_tl_snaps(trace: list, initial_capital: float) -> list:
    """Test trace'i için adım-adım TL türevleri — session_state.initial_capital değişince
    sayfa her yeniden render'da bu yeniden hesaplanır (ucuz, ~1000 adım)."""
    N_risky = trace[0]["prices_t"].shape[0] if trace else 0
    holding_days_prev = np.zeros(N_risky, dtype=np.int64)
    prev_portfolio_tl = float(initial_capital)
    snaps = []
    for t in trace:
        snap = compute_tl_step(
            nav=t["nav"],
            w_now=t["weights_after"],
            w_prev=t["weights_before"],
            prices_t=t["prices_t"],
            initial_capital=initial_capital,
            prev_portfolio_tl=prev_portfolio_tl,
            holding_days_prev=holding_days_prev,
            tx_cost_rate=float(t["reward_terms"].get("tx_cost", 0.0)),
        )
        snap["w_now"] = t["weights_after"]
        snaps.append(snap)
        holding_days_prev = snap["holding_days"]
        prev_portfolio_tl = snap["portfolio_tl"]
    return snaps
