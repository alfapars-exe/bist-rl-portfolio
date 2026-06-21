"""UI veri/egitim/test servisleri (app.py'den tasindi, P5).

Streamlit session_state'i ile core katmani arasindaki kopru: veri yukleme,
env/ajan kurulumu (core.factory'ye delege), canli egitim generator'i ve test
trajectory yakalama.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from agents.base import SupportsQValues
from config import SEED, ForecastConfig, MacroConfig
from core.factory import build_agent, build_env
from core.persistence import load_agent, model_path, save_agent
from core.trainer import train as train_loop
from data import align_macro, download_bist, download_macro, train_test_split
from env.portfolio_env import ACTION_NAMES
from ui.state import _agent_key
from utils.baselines import equal_weight
from utils.features import TrainScaler, add_features
from utils.macro import MacroScaler, add_macro_features
from utils.portfolio_tl import compute_tl_step

# PDF §11: egitilmis modeller diske burada kaydedilir/yuklenir (sunum kaliciligi).
# N11: MODELS_DIR ve model_path artik core.persistence'da tanimlidi; buradan re-export.
from core.persistence import MODELS_DIR  # noqa: E402 (import blogunun sonunda)


def save_trained_agent(algo: str, horizon: str, adaptive: bool):
    """Session'daki egitilmis ajani diske kaydeder; yolu doner (yoksa None).

    N11: dosya adi algo_{horizon}_{adaptive}.pt — farkli vade/adaptive birbirini ezmez.
    """
    entry = st.session_state.trained_agents.get(_agent_key(algo, horizon, adaptive))
    if not entry or entry[0] is None:
        return None
    path = model_path(algo, horizon, adaptive)
    return save_agent(entry[0], algo, path, horizon=horizon, adaptive=adaptive)


def load_saved_agent(algo: str, horizon: str = "medium", adaptive: bool = True):
    """Diskteki modeli yukler, session_state.trained_agents'a koyar; anahtari doner.

    N11: horizon + adaptive parametreleri dosya adini belirler.
    """
    path = model_path(algo, horizon, adaptive)
    if not path.exists():
        return None
    agent, meta = load_agent(path)
    key = _agent_key(meta["algo"], meta["horizon"], meta["adaptive"])
    st.session_state.trained_agents[key] = (agent, [])   # disk'ten geldi; egitim egrisi yok
    return key


def load_saved_agent_from_path(path):
    """N11: Tam yol verilince dogrudan yukler (selectbox ile secilen model icin)."""
    from pathlib import Path as _Path
    path = _Path(path)
    if not path.exists():
        return None
    agent, meta = load_agent(path)
    key = _agent_key(meta["algo"], meta["horizon"], meta["adaptive"])
    st.session_state.trained_agents[key] = (agent, [])
    return key, meta


def _load_data():
    """Veri indir + z-score scaler'ı fit et.

    Tarih aralığı session_state.data_start / data_split / data_end'den okunur
    (sidebar tarih seçici). Scaler/forecaster/MacroScaler YALNIZ px_tr'de fit
    edilir — sızıntı yok.
    """
    from config import DataConfig as _DC
    _dc_defaults = _DC()
    data_start = st.session_state.get("data_start", _dc_defaults.start)
    data_split = st.session_state.get("data_split", _dc_defaults.train_end)
    data_end   = st.session_state.get("data_end",   _dc_defaults.end)

    with st.spinner("Veri indiriliyor / cache okunuyor ..."):
        prices = download_bist(start=data_start, end=data_end)
    # Sentetik-veri görünürlüğü: ağ + cache yoksa download_bist SENTETİK GBM'e düşer
    # (px.attrs["synthetic"]=True). Bu durumda sonuçlar GERÇEK DEĞİLDİR ve NaN/anlamsız
    # değerler çıkabilir → kullanıcıya AÇIK uyarı (sessiz NaN yerine).
    if prices.attrs.get("synthetic"):
        st.error("⚠️ GERÇEK BIST verisi yüklenemedi (ağ erişimi + cache yok) → SENTETİK GBM "
                 "verisi kullanılıyor. Sonuçlar gerçek DEĞİLDİR; NaN/anlamsız değerler "
                 "görülebilir. (Bu ortamda `data/prices.parquet` eksik — deploy ile yüklenmeli.)")
    feats_all_raw = add_features(prices)
    px_tr, px_te = train_test_split(prices, split=data_split)
    if ForecastConfig.enabled:                     # v2: forecast feature (train-only fit)
        from forecast.forecaster import build_forecast_feature
        feats_all_raw["forecast"] = build_forecast_feature(
            prices, px_tr, window=ForecastConfig.window, conv_ch=ForecastConfig.conv_ch,
            hidden=ForecastConfig.hidden, epochs=ForecastConfig.epochs,
            lr=ForecastConfig.lr, batch=ForecastConfig.batch, seed=SEED)
    feats_tr_raw = {k: v.loc[px_tr.index] for k, v in feats_all_raw.items()}
    feats_te_raw = {k: v.loc[px_te.index] for k, v in feats_all_raw.items()}

    # SIZINTI KORUMASI: scaler YALNIZ eğitim kısmında fit edilir, test'e transform uygulanır.
    scaler = TrainScaler().fit(feats_tr_raw)
    st.session_state.prices = prices
    st.session_state.px_tr = px_tr
    st.session_state.px_te = px_te
    st.session_state.feats_tr = scaler.transform(feats_tr_raw)
    st.session_state.feats_te = scaler.transform(feats_te_raw)
    st.session_state.scaler = scaler

    # v6: makro rejim (faiz/dolar/altin) — train-only z-score; ham regime ayri (V7).
    # MacroScaler da YALNIZ px_tr kısmında fit edilir.
    macro_tr = macro_te = regime_tr = regime_te = None
    if MacroConfig.enabled:
        mfeat = add_macro_features(
            align_macro(download_macro(start=data_start, end=data_end), prices.index)
        )
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


def _make_env(is_train: bool, algo: str, horizon: str, adaptive: bool, max_steps: int,
              cash_daily_rate: float | None = None):
    """UI ortam kurulumu — session_state'i okuyup core.factory.build_env'e delege eder (P3).

    N12: cash_daily_rate None verilirse session_state.cash_daily_rate okunur;
         o da yoksa env kendi config default'unu kullanir.
    N10: is_train=True ise max_steps egitim episode uzunlugu (sidebar slider'dan);
         is_train=False ise max_steps=10_000 (tam test donemi — degistirilmez).
    """
    px_df = st.session_state.px_tr if is_train else st.session_state.px_te
    feats = st.session_state.feats_tr if is_train else st.session_state.feats_te
    macro = st.session_state.get("macro_tr" if is_train else "macro_te")
    regime = st.session_state.get("regime_tr" if is_train else "regime_te")
    # Eğitimde UI'dan okunan σ geçilir; eval'de None → env gürültüyü zaten
    # random_start=False ile kapatır, ama yine de None göndererek kasıtsız gürültüyü engelle.
    noise_std = (st.session_state.get("price_noise_std") if is_train else None)
    # N12: nakit faiz — önce parametre, sonra session_state, sonra env default (None).
    if cash_daily_rate is None:
        cash_daily_rate = st.session_state.get("cash_daily_rate", None)
    return build_env(
        algo, px_df, feats, horizon=horizon, adaptive=adaptive, max_steps=max_steps,
        random_start=is_train, seed=SEED,          # v2: egitimde rastgele pencere, eval'de sabit
        reward_overrides=st.session_state.get("reward_cfg", {}) or {},
        price_noise_std=noise_std,                 # UI σ kontrolü (train-only)
        macro=macro, regime=regime,                # v6: makro rejim blogu + ham regime
        cash_daily_rate=cash_daily_rate,           # N12: UI nakit faiz oranı
    )


def _make_agent(algo: str, state_dim: int, action_dim: int, hp: dict):
    """Shim — SOLID P3: tek dogruluk kaynagi core.factory.build_agent.
    (test_config_wiring app._make_agent'i cagirir; app.py bunu re-export eder.)"""
    return build_agent(algo, state_dim, action_dim, hp)


# =====================================================================
# Eğitim jeneratörü — canlı UI için episod başına yield
# =====================================================================
def train_generator(algo: str, horizon: str, adaptive: bool, hp: dict,
                    rollout_len: int = 400, resume_agent=None,
                    n_episodes: int | None = None):
    """Episod/update başına bir telemetri kaydı yield eder.

    n_episodes verilirse (UI'dan gelir) tam o kadar episode/update koşar ve
    generator kendiliğinden biter. None ise sonsuz akış — tüketici (tab_train)
    'Eğitimi Durdur' butonuyla keser.

    resume_agent verilirse (G4) yeni ajan kurulmaz; durdurulan ajan AYNI
    ağırlık/optimizer/replay buffer'la kaldığı yerden öğrenmeye devam eder.

    N10: session_state.train_max_steps varsa o değer kullanılır (sidebar slider);
         yoksa algo'ya özgü sabit fallback.
    """
    _algo_defaults = {"DQN": 252, "PPO": 10_000, "SAC": 1200, "TD3": 1200}
    max_steps = int(st.session_state.get("train_max_steps", _algo_defaults.get(algo, 252)))
    env = _make_env(True, algo, horizon, adaptive, max_steps=max_steps)
    if resume_agent is not None:
        agent = resume_agent
    else:
        action_dim = env.n_discrete if algo == "DQN" else env.action_dim
        agent = _make_agent(algo, env.state_dim, action_dim, hp)
    # Başarı kıyası için tren EW NAV'ı (core.trainer success'i bununla hesaplar)
    ew_tr = equal_weight(st.session_state.px_tr)["nav"]
    # n_episodes=None → sonsuz akış; int → tam o kadar episode/update sonra generator biter.
    # core.trainer.train ajan tipine göre dispatch eder; telemetri dict'i CLI ile ortaktır.
    yield from train_loop(agent, env, n_iters=n_episodes, rollout_len=rollout_len, ew_nav=ew_tr)


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
