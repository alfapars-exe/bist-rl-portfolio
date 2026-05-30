"""BIST 28 Portföy RL — İnteraktif Streamlit Demo.

Tek komutla çalışır:
    streamlit run app.py

Sekmeler:
  1. Veri & MDP      : 28 ticker, MDP tuple, vade preset tablosu, adaptif ödül açıklaması
  2. Eğitim          : Seçili ajanı canlı güncellenen eğri-placeholder'larıyla eğit
  3. Test            : Adım-adım oynatma; durum, aksiyon, Q/ağırlık/ödül panelleri
  4. Karşılaştırma   : Tüm ajanlar + baseline'lar; metrik tablosu, NAV, heatmap

Kullanıcı talepleri karşılanıyor:
  - 3 ajan (DQN/PPO/SAC) radyosu
  - Vade preset'leri (kısa / orta / uzun) — ödül katsayıları + rebalans frekansı
  - Adaptif ödül toggle (rolling vol & turnover EWMA ile η_t, λ_t, τ_t ölçekleme)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import BIST28, SPLIT, START, END, download_bist, train_test_split
from env.portfolio_env import (
    ACTION_NAMES, HORIZON_PRESETS, DiscretePortfolioEnv, PortfolioEnv,
)
from agents import DQNAgent, PPOAgent, SACAgent
from config import SEED, FEATURES, DQNConfig, PPOConfig, SACConfig, ForecastConfig
from core.trainer import train as train_loop
from utils.features import TrainScaler, add_features
from utils.baselines import buy_and_hold_index, equal_weight, mean_variance
from utils.metrics import summary
from utils.portfolio_tl import (
    build_cumulative_trade_log, build_portfolio_table, build_trade_log,
    compute_tl_series, compute_tl_step, step_rows_for_training,
)


# =====================================================================
# Sabitler, yardımcılar
# =====================================================================
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
    st.session_state.data_loaded = True


def _make_env(is_train: bool, algo: str, horizon: str, adaptive: bool, max_steps: int):
    px_df   = st.session_state.px_tr if is_train else st.session_state.px_te
    feats   = st.session_state.feats_tr if is_train else st.session_state.feats_te
    if "forecast" in feats and algo not in ForecastConfig.forecast_agents:
        feats = {k: v for k, v in feats.items() if k != "forecast"}   # v2: PPO forecast almaz
    cls = DiscretePortfolioEnv if algo == "DQN" else PortfolioEnv
    cfg = st.session_state.get("reward_cfg", {}) or {}
    return cls(
        px_df, feats, horizon=horizon, adaptive=adaptive, max_steps=max_steps,
        random_start=is_train, seed=SEED,          # v2: egitimde rastgele pencere, eval'de sabit
        eta_base=cfg.get("eta_base"),
        lambda_base=cfg.get("lambda_base"),
        tau_base=cfg.get("tau_base"),
        vol_target=float(cfg.get("vol_target", 0.02)),
        turnover_target=float(cfg.get("turnover_target", 0.05)),
        ema_alpha=float(cfg.get("ema_alpha", 0.05)),
        bankruptcy_nav=cfg.get("bankruptcy_nav"),
        bankruptcy_penalty=cfg.get("bankruptcy_penalty"),
    )


def _make_agent(algo: str, state_dim: int, action_dim: int, hp: dict):
    if algo == "DQN":
        return DQNAgent(
            state_dim, action_dim,
            hidden=tuple(hp.get("hidden", DQNConfig.hidden)),
            lr=hp.get("lr", DQNConfig.lr),
            eps_decay=hp.get("eps_decay", DQNConfig.eps_decay),
            batch_size=hp.get("batch_size", DQNConfig.batch_size),
            target_update=hp.get("target_update", DQNConfig.target_update),
            seed=SEED,
        )
    if algo == "PPO":
        return PPOAgent(
            state_dim, action_dim,
            hidden=tuple(hp.get("hidden", PPOConfig.hidden)),
            lr_p=hp.get("lr_p", PPOConfig.lr_p), lr_v=hp.get("lr_v", PPOConfig.lr_v),
            clip=hp.get("clip", PPOConfig.clip), ent_coef=hp.get("ent_coef", PPOConfig.ent_coef),
            batch_size=hp.get("batch_size", PPOConfig.batch_size), n_epochs=hp.get("n_epochs", PPOConfig.n_epochs),
            seed=SEED,
        )
    return SACAgent(
        state_dim, action_dim,
        hidden=tuple(hp.get("hidden", SACConfig.hidden)),
        lr_pi=hp.get("lr_pi", SACConfig.lr_pi), lr_q=hp.get("lr_q", SACConfig.lr_q),
        alpha=hp.get("alpha", SACConfig.alpha), tau=hp.get("tau", SACConfig.tau),
        batch_size=hp.get("batch_size", SACConfig.batch_size), seed=SEED,
    )


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
    has_q = hasattr(agent, "q_values")  # yalnizca DQN introspeksiyonu sunar
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


# =====================================================================
# UI yardımcıları
# =====================================================================
def _weights_pie(weights: np.ndarray, title: str = "Portföy Ağırlıkları"):
    df = pd.DataFrame({"asset": ASSET_NAMES, "weight": weights})
    df = df[df["weight"] > 0.005]  # çok küçük dilimleri gizle
    fig = px.pie(df, names="asset", values="weight", title=title, hole=0.35)
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(height=380, showlegend=False)
    return fig


def _q_bar(q_values: np.ndarray, chosen: int):
    colors = ["#d62728" if i == chosen else "#888" for i in range(len(q_values))]
    fig = go.Figure(
        go.Bar(x=ACTION_NAMES[:6], y=q_values, marker_color=colors,
               text=[f"{v:+.3f}" for v in q_values], textposition="outside")
    )
    fig.update_layout(title="Q(s, ·) — seçim kırmızı", height=320,
                      yaxis_title="Q-değeri", margin=dict(t=40, b=40))
    return fig


def _reward_bar(rt: dict):
    names = ["log_return", "tx_cost", "dd_penalty", "dsr", "total"]
    vals = [rt["log_return"], -rt["tx_cost"], -rt["drawdown_penalty"],
            rt.get("dsr_term", 0.0), rt["total"]]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in vals]
    fig = go.Figure(go.Bar(x=names, y=vals, marker_color=colors,
                           text=[f"{v:+.5f}" for v in vals], textposition="outside"))
    fig.update_layout(title="Ödül terimleri (bu adım)", height=280,
                      margin=dict(t=40, b=30))
    return fig


def _state_top_features(state_vec: np.ndarray, top_k: int = 8) -> pd.DataFrame:
    """Durum vektöründen top-k öznitelik çıkar (mutlak değer sıralı). Feature
    sayısı config.FEATURES'tan dinamik okunur (v2: 12 özellik)."""
    n_assets = len(BIST28)
    F = (len(state_vec) - (n_assets + 1)) // n_assets   # state'ten türet (12 ya da 13)
    names = list(FEATURES) + ["forecast"]
    feat_names = (names + [f"f{i}" for i in range(F)])[:F]
    snap = state_vec[: F * n_assets].reshape(F, n_assets)
    rows = []
    for fi, fname in enumerate(feat_names):
        for ai, aname in enumerate(BIST28):
            rows.append((fname, aname, float(snap[fi, ai])))
    df = pd.DataFrame(rows, columns=["feature", "asset", "z-score"])
    df["|z|"] = df["z-score"].abs()
    df = df.sort_values("|z|", ascending=False).head(top_k).drop(columns="|z|")
    return df.reset_index(drop=True)


def _horizon_preset_table() -> pd.DataFrame:
    rows = []
    for h, p in HORIZON_PRESETS.items():
        rows.append({
            "Vade": {"short": "Kısa", "medium": "Orta", "long": "Uzun"}[h],
            "Rebalans (gün)": p["rebalance"],
            "Momentum penceresi": p["mom_window"],
            "Min-vol penceresi": p["minvol_window"],
            "η (tx cost)": p["eta"],
            "λ (DD penalty)": p["lam"],
            "τ (DD eşiği)": p["tau"],
            "γ (discount)": p["gamma"],
        })
    return pd.DataFrame(rows)


# =====================================================================
# Sidebar
# =====================================================================
def _sidebar_reward_editor(preset: dict):
    """⚖️ Ödül & Ceza katsayıları düzenleyicisi — session_state.reward_cfg'i günceller.

    None/boş değerler env'in preset default'larına düşer. Kullanıcı 'Preset'e dön'
    ile tüm override'ları sıfırlayabilir.
    """
    cfg = st.session_state.setdefault("reward_cfg", {})
    with st.sidebar.expander("⚖️ Ödül & Ceza Katsayıları", expanded=False):
        st.caption("Ödül = log-getiri − η·turnover − λ·max(0, DD−τ) − iflas cezası")

        if st.button("↺ Preset'e dön (tüm override'ları sıfırla)",
                     key="reset_reward_cfg", use_container_width=True):
            st.session_state.reward_cfg = {}
            st.rerun()

        st.markdown("**Ödül terimleri** (vade preset'i default olarak)")
        cfg["eta_base"] = st.number_input(
            "η — İşlem (turnover) maliyeti katsayısı",
            value=float(cfg.get("eta_base", preset["eta"])),
            min_value=0.0, max_value=0.1, step=0.0001, format="%.4f",
            help="Ağırlık değişiminin L1 normu bu katsayı ile çarpılıp ödülden düşülür "
                 "(ve NAV'ı azaltır). Büyütünce ajan daha az işlem yapar.",
        )
        cfg["lambda_base"] = st.number_input(
            "λ — Drawdown (DD) ceza katsayısı",
            value=float(cfg.get("lambda_base", preset["lam"])),
            min_value=0.0, max_value=10.0, step=0.05, format="%.3f",
            help="max(0, DD − τ) bu katsayı ile çarpılıp ödülden düşülür.",
        )
        cfg["tau_base"] = st.number_input(
            "τ — DD eşiği (oran)",
            value=float(cfg.get("tau_base", preset["tau"])),
            min_value=0.0, max_value=0.5, step=0.005, format="%.3f",
            help="Tepe-den DD > τ olduğunda ceza başlar. 0.05 = %5'lik DD toleransı.",
        )

        st.markdown("**İflas (simülasyonu durdurma)**")
        cfg["bankruptcy_nav"] = st.number_input(
            "İflas NAV eşiği",
            value=float(cfg.get("bankruptcy_nav", 0.01)),
            min_value=0.0, max_value=0.9, step=0.01, format="%.2f",
            help="NAV bu eşiğin altına düşerse episod iflas olarak sonlandırılır. "
                 "Örn. 0.01 = başlangıç sermayesinin %1'ine inmek.",
        )
        cfg["bankruptcy_penalty"] = st.number_input(
            "İflas ek ceza değeri",
            value=float(cfg.get("bankruptcy_penalty", 10.0)),
            min_value=0.0, max_value=1000.0, step=1.0, format="%.1f",
            help="İflas anında toplam ödüle eklenen negatif terim. Log-ölçeğinde büyük değer "
                 "(normal adım ödülü ~±0.01). Ajan iflasa gitmemeyi öğrenir.",
        )

        st.markdown("**Adaptif şekillendirici hedefleri**")
        cfg["vol_target"] = st.number_input(
            "vol_target (hedef realize vol)",
            value=float(cfg.get("vol_target", 0.02)),
            min_value=0.0001, max_value=0.5, step=0.001, format="%.4f",
            help="Adaptif mod açıkken λ ve τ bu hedefe göre ölçeklenir.",
        )
        cfg["turnover_target"] = st.number_input(
            "turnover_target (hedef turnover)",
            value=float(cfg.get("turnover_target", 0.05)),
            min_value=0.001, max_value=1.0, step=0.005, format="%.3f",
            help="Adaptif mod açıkken η bu hedefe göre ölçeklenir.",
        )
        cfg["ema_alpha"] = st.slider(
            "EMA α (adaptif hafıza)",
            min_value=0.001, max_value=0.5, value=float(cfg.get("ema_alpha", 0.05)),
            step=0.005, format="%.3f",
            help="Büyük α = daha hızlı uyum, küçük α = daha stabil.",
        )

        st.caption("⚠️ Bu ayarları değiştirdikten sonra ajanları **yeniden eğitmek** "
                   "anlamlı olur; eski ajan farklı ortamda öğrenilmiştir.")


def sidebar_controls():
    st.sidebar.title("⚙️ Kontrol Paneli")

    st.sidebar.subheader("💰 Başlangıç Sermayesi")
    st.session_state.initial_capital = st.sidebar.number_input(
        "Başlangıç TL Parası",
        min_value=1_000.0, value=float(st.session_state.initial_capital),
        step=10_000.0, format="%.0f",
        help="Eğitim ve test boyunca bu bakiye üzerinden lot/TL hesaplanır. "
             "Değeri istediğin zaman değiştirebilirsin — eğitilmiş ajanı yeniden eğitmen gerekmez.",
    )

    st.sidebar.subheader("⏱ Eğitim Hızı")
    st.session_state.train_delay = st.sidebar.slider(
        "İter arası gecikme (sn)", 0.0, 3.0, float(st.session_state.train_delay), step=0.1,
        help="0 = tam hız. Büyütünce iter'ler arasında yapay bekleme olur; "
             "canlı adım tablolarını rahat okumak için kullan.",
    )

    st.sidebar.divider()
    algo = st.sidebar.radio("Ajan", ["DQN", "PPO", "SAC"],
                            index=["DQN", "PPO", "SAC"].index(st.session_state.selected_algo))
    st.session_state.selected_algo = algo

    horizon_label = st.sidebar.radio(
        "Vade (Yatırım Ufku)",
        ["Kısa", "Orta", "Uzun"],
        index=["short", "medium", "long"].index(st.session_state.horizon),
        help="Rebalans frekansı + (η, λ, τ, γ) preset'lerini değiştirir.",
    )
    st.session_state.horizon = {"Kısa": "short", "Orta": "medium", "Uzun": "long"}[horizon_label]
    preset = HORIZON_PRESETS[st.session_state.horizon]
    st.sidebar.caption(
        f"Rebalans: {preset['rebalance']}g · η={preset['eta']} · "
        f"λ={preset['lam']} · τ={preset['tau']} · γ={preset['gamma']}"
    )

    _sidebar_reward_editor(preset)

    st.session_state.adaptive = st.sidebar.checkbox(
        "Adaptif ödül aktif", value=st.session_state.adaptive,
        help="Açıkken η_t, λ_t, τ_t rolling vol & turnover EWMA'larına göre ölçeklenir."
    )

    st.sidebar.divider()
    st.sidebar.subheader("📊 Veri")
    if not st.session_state.data_loaded:
        if st.sidebar.button("Veriyi Yükle / İndir", use_container_width=True):
            _load_data()
            st.rerun()
    else:
        st.sidebar.success(f"Veri yüklü: {st.session_state.prices.shape[0]} gün × "
                           f"{st.session_state.prices.shape[1]} hisse")
        if st.sidebar.button("Veriyi yeniden yükle", use_container_width=True):
            for k in ["data_loaded", "prices", "px_tr", "px_te",
                      "feats_tr", "feats_te", "scaler",
                      "trained_agents", "test_traces", "baselines"]:
                st.session_state[k] = False if k == "data_loaded" else (
                    {} if k in ["trained_agents", "test_traces"] else None)
            st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader(f"🎛 Hiperparametreler ({algo})")
    hp = {}
    if algo == "DQN":
        st.sidebar.caption("Epizot sayısı **sınırsız** — istediğin noktada 'Eğitimi Durdur' butonuyla kes.")
        hp["lr"]        = st.sidebar.select_slider("Öğrenme oranı",
            options=[1e-4, 3e-4, 5e-4, 1e-3, 3e-3], value=1e-3)
        hp["eps_decay"] = st.sidebar.slider("ε decay adımı", 2_000, 30_000, 10_000, step=1_000)
        hp["batch_size"]= st.sidebar.select_slider("Batch", options=[32, 64, 128], value=64)
        hp["target_update"] = st.sidebar.slider("Target sync", 100, 2000, 500, step=100)
    elif algo == "PPO":
        st.sidebar.caption("Update sayısı **sınırsız** — istediğin noktada 'Eğitimi Durdur' butonuyla kes.")
        hp["rollout_len"]= st.sidebar.slider("Rollout uzunluğu", 128, 1024, 400, step=64)
        hp["lr_p"]       = st.sidebar.select_slider("Policy LR",
            options=[1e-4, 3e-4, 1e-3], value=3e-4)
        hp["lr_v"]       = st.sidebar.select_slider("Value LR",
            options=[3e-4, 1e-3, 3e-3], value=1e-3)
        hp["clip"]       = st.sidebar.slider("Clip ε", 0.05, 0.4, 0.2, step=0.05)
        hp["ent_coef"]   = st.sidebar.select_slider("Entropi katsayısı",
            options=[0.0, 0.001, 0.005, 0.01, 0.02], value=0.005)
        hp["batch_size"] = st.sidebar.select_slider("Mini-batch", options=[64, 128, 256], value=128)
        hp["n_epochs"]   = st.sidebar.slider("Epoch", 2, 10, 6, step=1)
    else:
        st.sidebar.caption("Epizot sayısı **sınırsız** — istediğin noktada 'Eğitimi Durdur' butonuyla kes.")
        hp["lr_pi"]      = st.sidebar.select_slider("Policy LR",
            options=[1e-4, 3e-4, 1e-3], value=3e-4)
        hp["lr_q"]       = st.sidebar.select_slider("Q LR",
            options=[3e-4, 5e-4, 1e-3], value=5e-4)
        hp["alpha"]      = st.sidebar.slider("Entropi α", 0.0, 0.5, 0.05, step=0.01)
        hp["tau"]        = st.sidebar.select_slider("Soft update τ",
            options=[0.005, 0.01, 0.05], value=0.01)
        hp["batch_size"] = st.sidebar.select_slider("Batch", options=[64, 128, 256], value=128)

    st.sidebar.caption(f"Seed: {SEED} (sabit)")
    return algo, st.session_state.horizon, st.session_state.adaptive, hp


# =====================================================================
# Sekme 1 — Veri & MDP
# =====================================================================
def tab_mdp():
    st.header("📐 Veri ve MDP Formülasyonu")

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Evren: BIST 28 Hisse")
        st.write(f"**Tarih aralığı:** {START} → {END} · **Split:** {SPLIT}")
        st.write(f"**KOZAA.IS ve KOZAL.IS hariç tutuldu.**")
        chips = "  ·  ".join(BIST28)
        st.caption(chips)

        if st.session_state.data_loaded:
            px_df = st.session_state.prices
            norm = (px_df / px_df.iloc[0]).resample("W").last()
            fig_px = px.line(norm, title="Normalize kapanış (haftalık)")
            fig_px.update_layout(height=280, showlegend=False,
                                 margin=dict(t=40, b=20), xaxis_title="",
                                 yaxis_title="NAV (ilk gün = 1.0)")
            st.plotly_chart(fig_px, use_container_width=True)

    with col2:
        st.subheader("MDP Tuple (𝒮, 𝒜, 𝒫, r, γ)")
        mdp = pd.DataFrame([
            ("𝒮 Durum Uzayı", "ℝ³⁹³ — 28 hisse × 13 özellik (12 teknik + 1 forecast, z-skorlu) + 29 boyutlu ağırlık"),
            ("𝒜 Eylem Uzayı (DQN)", "6 şablon: Nakit, Eşit Ağırlık, Top-3/Top-5 Mom, Ters-Vol, Min-Vol"),
            ("𝒜 Eylem Uzayı (PPO/SAC)", "ℝ²⁹ → softmax → portföy simpleksi"),
            ("𝒫 Geçiş", "Piyasa tarafından belirlenen stokastik süreç"),
            ("r Ödül", "log(1+w·r) − η_t·‖Δw‖₁ − λ_t·max(0, DD−τ_t)"),
            ("γ İndirgeme", "0.95 / 0.99 / 0.995 (vadeye göre)"),
            ("Sonlandırma", "veri sonu VEYA NAV<0.01 (iflas) VEYA 252 adım"),
        ], columns=["Bileşen", "Tanım"])
        st.dataframe(mdp, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("⏳ Vade Preset'leri & Adaptif Ödül")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.dataframe(_horizon_preset_table(), hide_index=True, use_container_width=True)
    with c2:
        st.markdown("""
**Adaptif şekillendirici** (`AdaptiveRewardShaper`):
EWMA ile rolling realized vol (`vol_ewma`) ve rolling turnover (`turnover_ewma`) izlenir.

- **η_t** = η_base × max(1, turnover_ewma / turnover_target)
  — çok işlem yapan ajan kendi kendini cezalandırır
- **λ_t** = λ_base × (1 + max(0, vol_ratio − 1))
  — yüksek-vol rejiminde DD cezası büyür
- **τ_t** = τ_base × max(0.7, 1 / vol_ratio)
  — volatil rejimde eşik daralır → daha hassas DD cezası
""")
        st.info(f"Adaptif mod: **{'Açık' if st.session_state.adaptive else 'Kapalı'}** · "
                f"Seçili vade: **{st.session_state.horizon}**")


# =====================================================================
# Sekme 2 — Eğitim (canlı)
# =====================================================================
def tab_train(algo: str, horizon: str, adaptive: bool, hp: dict):
    st.header(f"🎓 Eğitim — {algo} · {horizon.upper()} · "
              f"Adaptif: {'Açık' if adaptive else 'Kapalı'}")

    if not st.session_state.data_loaded:
        st.warning("Önce sidebar'dan 'Veriyi Yükle' butonuna basın.")
        return

    key = _agent_key(algo, horizon, adaptive)
    already = key in st.session_state.trained_agents
    if already:
        st.success(f"Bu konfigürasyon daha önce eğitildi. "
                   f"({len(st.session_state.trained_agents[key][1])} iterasyon)")

    col_run, col_clear = st.columns([1, 1])
    with col_run:
        run = st.button(f"{'Yeniden Eğit' if already else 'Eğit'}", type="primary")
    with col_clear:
        if already and st.button("Bu konfigürasyonu unut"):
            st.session_state.trained_agents.pop(key, None)
            st.session_state.test_traces.pop(key, None)
            st.rerun()

    st.caption("💡 Eğitim başladıktan sonra her iter sonunda **⏹ Eğitimi Durdur** butonu çıkar — "
               "istediğin noktada kesebilirsin, son ajan otomatik kaydedilir.")

    if not run:
        if already:
            _render_training_curves(st.session_state.trained_agents[key][1], algo)
        return

    # --- CANLI EĞİTİM (sınırsız) ---
    status = st.empty()
    status.info("Başlıyor — 'Eğitimi Durdur' ile istediğin noktada kes.")

    # Üst satır: canlı throughput metrikleri
    tp_cols = st.columns(4)
    ph_iter    = tp_cols[0].empty()
    ph_rate    = tp_cols[1].empty()
    ph_avg     = tp_cols[2].empty()
    ph_elapsed = tp_cols[3].empty()

    row1 = st.columns(2)
    row2 = st.columns(2)
    ph_reward  = row1[0].empty()
    ph_gain    = row1[1].empty()
    ph_success = row2[0].empty()
    ph_loss    = row2[1].empty()
    stop_slot  = st.empty()

    # Canlı TL paneli placeholder'ları
    st.markdown("### 💰 Son episod TL izlemesi")
    tl_metric_cols = st.columns(6)
    ph_tl_start = tl_metric_cols[0].empty()
    ph_tl_end   = tl_metric_cols[1].empty()
    ph_tl_net   = tl_metric_cols[2].empty()
    ph_tl_min   = tl_metric_cols[3].empty()
    ph_tl_max   = tl_metric_cols[4].empty()
    ph_tl_dd    = tl_metric_cols[5].empty()
    tl_chart_cols = st.columns(2)
    ph_tl_line  = tl_chart_cols[0].empty()
    ph_tl_bar   = tl_chart_cols[1].empty()
    ph_tl_table = st.empty()
    ph_tl_port  = st.empty()
    ph_bankrupt = st.empty()

    curve = []
    gen = train_generator(algo, horizon, adaptive, hp,
                          rollout_len=int(hp.get("rollout_len", 400)))
    t0 = time.time()
    iter_times = []  # son N iter süresi (iter/sn için)
    trained_agent = None
    stopped_early = False
    initial_capital = float(st.session_state.initial_capital)

    for rec in gen:
        iter_start_elapsed = time.time() - t0
        trained_agent = rec["agent"]
        env = rec["env"]
        curve.append({k: v for k, v in rec.items() if k not in ("agent", "env", "actions")})
        # Her iter sonunda session'a yaz → kullanıcı durdurursa veya refresh etse bile son hali kalır
        st.session_state.trained_agents[key] = (trained_agent, list(curve))
        df = pd.DataFrame(curve)

        fig_r = px.line(df, x="iter", y="reward",
                        title="Kümülatif Ödül (iterasyon başına — çevre ödülü Σr)",
                        markers=True)
        fig_r.update_layout(height=260, margin=dict(t=40, b=20))
        ph_reward.plotly_chart(fig_r, use_container_width=True)

        fig_g = px.line(df, x="iter", y="gain",
                        title="Kazanç (nihai NAV − 1.0)",
                        markers=True)
        fig_g.update_layout(height=260, margin=dict(t=40, b=20))
        ph_gain.plotly_chart(fig_g, use_container_width=True)

        fig_s = px.bar(df, x="iter", y="success",
                       title="Başarı (EW benchmark'a göre 0/1)")
        fig_s.update_layout(height=260, margin=dict(t=40, b=20),
                            yaxis=dict(range=[0, 1.2], tickvals=[0, 1]))
        ph_success.plotly_chart(fig_s, use_container_width=True)

        if "loss" in df.columns:
            fig_l = px.line(df, x="iter", y="loss",
                            title="Ortalama loss (düşüş beklenir)",
                            markers=True)
            fig_l.update_layout(height=260, margin=dict(t=40, b=20))
            ph_loss.plotly_chart(fig_l, use_container_width=True)

        # --- Throughput metrikleri ---
        iter_end_elapsed = time.time() - t0
        iter_times.append(iter_end_elapsed - iter_start_elapsed)
        recent = iter_times[-5:]
        rate = (len(recent) / sum(recent)) if sum(recent) > 0 else 0.0
        avg = sum(iter_times) / len(iter_times)
        ph_iter.metric("Iter", rec["iter"] + 1)
        ph_rate.metric("Iter/sn", f"{rate:.2f}")
        ph_avg.metric("Ort. iter süresi", f"{avg:.2f}s")
        mm = int(iter_end_elapsed // 60); ss = int(iter_end_elapsed % 60)
        ph_elapsed.metric("Toplam elapsed", f"{mm:02d}:{ss:02d}")

        # --- Canlı TL paneli (son episod için env.nav_history / weight_history kullan) ---
        _render_train_tl_panel(
            env=env, algo=algo, rec=rec, initial_capital=initial_capital,
            ph_tl_start=ph_tl_start, ph_tl_end=ph_tl_end, ph_tl_net=ph_tl_net,
            ph_tl_min=ph_tl_min, ph_tl_max=ph_tl_max, ph_tl_dd=ph_tl_dd,
            ph_tl_line=ph_tl_line, ph_tl_bar=ph_tl_bar,
            ph_tl_table=ph_tl_table, ph_tl_port=ph_tl_port,
            ph_bankrupt=ph_bankrupt,
        )

        status.info(f"Iter {rec['iter']+1} · NAV={rec['nav']:.3f} · "
                    f"elapsed {iter_end_elapsed:.1f}s — "
                    f"istediğin yerde 'Eğitimi Durdur' butonuna basabilirsin")

        # Kullanıcı ayarladığı gecikmeyi iter arası uygula (slider canlı okunur).
        delay = float(st.session_state.get("train_delay", 0.0))
        if delay > 0:
            time.sleep(delay)

        # Döngü içi durdurma — her iter sonunda placeholder'a buton render eder.
        # Kullanıcı basarsa bir sonraki iter başlamaz, o ana kadar eğitilen ajan session'da kalmış olur.
        if stop_slot.button("⏹ Eğitimi Durdur", key=f"stop_loop_{rec['iter']}"):
            stopped_early = True
            break

    st.session_state.trained_agents[key] = (trained_agent, curve)
    elapsed = time.time() - t0
    stop_slot.empty()
    if stopped_early:
        status.warning(f"{algo} eğitimi {len(curve)}. iter sonunda durduruldu "
                       f"({elapsed:.1f}s) — son ajan session'a kaydedildi.")
    else:
        status.success(f"{algo} eğitildi ({len(curve)} iter, {elapsed:.1f}s) "
                       f"ve session'a kaydedildi. Tab 3'te test edebilirsiniz.")


def _render_train_tl_panel(env, algo, rec, initial_capital,
                            ph_tl_start, ph_tl_end, ph_tl_net, ph_tl_min, ph_tl_max, ph_tl_dd,
                            ph_tl_line, ph_tl_bar, ph_tl_table, ph_tl_port,
                            ph_bankrupt=None):
    """Son episod/iter için TL türevlerini hesapla ve placeholder'ları güncelle."""
    nav_hist = list(env.nav_history)
    weight_hist = list(env.weight_history)
    rt_hist = list(env.reward_terms_history)
    if len(nav_hist) < 2:
        return
    # env.t reset sonrası max(window, 21)'de başladı; her adımda +1.
    # Şu an env.t = t_start + step_count. weight_hist[k] env'in k'ıncı adım sonrası w'si
    # → prices[t_start + k] ile hizalanır.
    t_end = int(env.t)
    steps_done = int(env.step_count)
    t_start = t_end - steps_done
    tx_rates = [float(rt.get("tx_cost", 0.0)) for rt in rt_hist]
    snaps = compute_tl_series(
        nav_hist=nav_hist, weight_hist=weight_hist,
        prices_matrix=env.prices, initial_capital=initial_capital,
        tx_cost_rates=tx_rates, t_start=t_start,
    )
    if not snaps:
        return

    port_tl_arr = np.array([s["portfolio_tl"] for s in snaps])
    step_pnl_arr = np.array([s["step_pnl_tl"] for s in snaps])
    cum_pnl_arr = np.array([s["cum_pnl_tl"] for s in snaps])
    peak = np.maximum.accumulate(port_tl_arr)
    dd_pct = (peak - port_tl_arr) / np.maximum(peak, 1e-9)

    end_tl = float(port_tl_arr[-1])
    net = end_tl - initial_capital
    net_pct = 100.0 * net / initial_capital
    ph_tl_start.metric("Başlangıç TL", f"{initial_capital:,.0f} ₺")
    ph_tl_end.metric("Bitiş TL", f"{end_tl:,.0f} ₺")
    ph_tl_net.metric("Net Kâr", f"{net:+,.0f} ₺", delta=f"{net_pct:+.2f}%")
    ph_tl_min.metric("Min TL", f"{float(port_tl_arr.min()):,.0f} ₺")
    ph_tl_max.metric("Max TL", f"{float(port_tl_arr.max()):,.0f} ₺")
    ph_tl_dd.metric("Max DD %", f"{100*float(dd_pct.max()):.2f}%")

    # Portfolio TL zaman serisi + başlangıç referans
    idx = np.arange(1, len(snaps) + 1)
    fig_tl = go.Figure()
    fig_tl.add_trace(go.Scatter(x=idx, y=port_tl_arr, mode="lines",
                                name="Portföy TL", line=dict(color="#1f77b4")))
    fig_tl.add_hline(y=initial_capital, line_dash="dot",
                     annotation_text="başlangıç", line_color="#888")
    fig_tl.update_layout(title="Portföy Değeri (TL)", height=280,
                         margin=dict(t=40, b=30), xaxis_title="Gün",
                         yaxis_title="TL")
    ph_tl_line.plotly_chart(fig_tl, use_container_width=True)

    # Adım P&L bar chart (yeşil/kırmızı)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in step_pnl_arr]
    fig_bar = go.Figure(go.Bar(x=idx, y=step_pnl_arr, marker_color=colors))
    fig_bar.update_layout(title="Adım P&L (TL)", height=280,
                          margin=dict(t=40, b=30), xaxis_title="Gün",
                          yaxis_title="TL")
    ph_tl_bar.plotly_chart(fig_bar, use_container_width=True)

    # Tam adım tablosu
    dates_slice = env.dates[t_start + 1 : t_start + 1 + steps_done]
    actions = rec.get("actions") or []
    action_names = None
    action_indices = None
    if algo == "DQN" and actions:
        action_names = [ACTION_NAMES[a] if 0 <= a < len(ACTION_NAMES) else str(a)
                        for a in actions]
    elif algo == "PPO":
        action_names = ["PPO-Gaussian"] * len(snaps)
    elif algo == "SAC":
        action_names = ["SAC-tanh"] * len(snaps)
    df_rows = step_rows_for_training(
        curve_snaps=snaps, dates=np.asarray(dates_slice),
        action_names=action_names, action_indices=action_indices,
        reward_terms_list=rt_hist, initial_capital=initial_capital,
    )
    ph_tl_table.dataframe(df_rows, hide_index=True, use_container_width=True, height=500)

    # Episod sonu portföy panosu (w_prev = sondan bir önceki adım)
    last = snaps[-1]
    w_prev = weight_hist[-2] if len(weight_hist) >= 2 else None
    df_port = build_portfolio_table(BIST28, last, include_cash=True, w_prev=w_prev)
    ph_tl_port.dataframe(df_port, hide_index=True, use_container_width=True)

    # İflas olduysa eğitim panelinin altında uyarı göster
    last_rt = rt_hist[-1] if rt_hist else {}
    if ph_bankrupt is not None:
        if last_rt.get("bankrupt"):
            ph_bankrupt.error(
                f"💀 Bu episod **iflas** ile sonlandı — NAV={nav_hist[-1]:.4f} "
                f"iflas eşiğinin altına düştü. Ajan'a ek ödül cezası "
                f"−{last_rt.get('bankruptcy_penalty', 0):.1f} uygulandı.")
        else:
            ph_bankrupt.empty()


def _render_training_curves(curve: list, algo: str):
    df = pd.DataFrame(curve)
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    c1.plotly_chart(px.line(df, x="iter", y="reward", markers=True,
                            title="Kümülatif Ödül"), use_container_width=True)
    c2.plotly_chart(px.line(df, x="iter", y="gain", markers=True,
                            title="Kazanç (NAV − 1)"), use_container_width=True)
    c3.plotly_chart(px.bar(df, x="iter", y="success", title="Başarı (0/1)"),
                    use_container_width=True)
    if "loss" in df.columns:
        c4.plotly_chart(px.line(df, x="iter", y="loss", markers=True,
                                title="Loss"), use_container_width=True)


# =====================================================================
# Sekme 3 — Test (adım adım oynatma)
# =====================================================================
def tab_test(algo: str, horizon: str, adaptive: bool):
    st.header(f"🎬 Test — {algo} · {horizon.upper()} · "
              f"Adaptif: {'Açık' if adaptive else 'Kapalı'}")

    key = _agent_key(algo, horizon, adaptive)
    if key not in st.session_state.trained_agents:
        st.warning("Bu konfigürasyon henüz eğitilmedi. Tab 2'ye git ve 'Eğit' butonuna bas.")
        return

    if st.button("Test dönemini çalıştır (rollout + trajectory)", type="primary"):
        agent = st.session_state.trained_agents[key][0]
        with st.spinner("Ajan test döneminde adım adım çalıştırılıyor..."):
            trace = evaluate_with_trace(agent, algo, horizon, adaptive)
        st.session_state.test_traces[key] = trace
        st.session_state.step_idx = 0
        st.success(f"{len(trace)} adım yakalandı.")

    trace = st.session_state.test_traces.get(key)
    if not trace:
        st.info("Henüz trajectory yok. Yukarıdaki butona basın.")
        return

    max_step = len(trace) - 1
    # ---- Oynatma kontrolleri ----
    ctrl = st.columns([1, 1, 1, 1, 3])
    if ctrl[0].button("⏮", help="İlk"):
        st.session_state.step_idx = 0
    if ctrl[1].button("◀", help="Önceki"):
        st.session_state.step_idx = max(0, st.session_state.step_idx - 1)
    play_label = "⏸" if st.session_state.playing else "▶"
    if ctrl[2].button(play_label, help="Oynat/Durdur"):
        st.session_state.playing = not st.session_state.playing
    if ctrl[3].button("▶▶", help="Sonraki"):
        st.session_state.step_idx = min(max_step, st.session_state.step_idx + 1)

    st.session_state.step_idx = ctrl[4].slider(
        "Adım", 0, max_step, st.session_state.step_idx,
        label_visibility="collapsed", key="step_slider"
    )
    idx = st.session_state.step_idx
    snap = trace[idx]

    # TL türevleri — başlangıç kapital değişince anında yeniden hesaplanır
    initial_capital = float(st.session_state.initial_capital)
    tl_snaps = _compute_test_tl_snaps(trace, initial_capital)
    tl_now = tl_snaps[idx]
    tl_prev = tl_snaps[idx - 1] if idx > 0 else None

    # Üst bilgi satırı — TL metrikleri dahil (6 kolon)
    info_cols = st.columns(6)
    info_cols[0].metric("Tarih", snap["date"])
    info_cols[1].metric("NAV", f"{snap['nav']:.4f}")
    info_cols[2].metric("Portföy TL", f"{tl_now['portfolio_tl']:,.0f} ₺")
    info_cols[3].metric("Nakit TL", f"{tl_now['cash_tl']:,.0f} ₺")
    cum_pct = 100 * tl_now["cum_pnl_tl"] / initial_capital if initial_capital else 0
    info_cols[4].metric("Kümülatif Kâr",
                        f"{tl_now['cum_pnl_tl']:+,.0f} ₺",
                        delta=f"{cum_pct:+.2f}%")
    info_cols[5].metric("Adım P&L", f"{tl_now['step_pnl_tl']:+,.0f} ₺")

    # Paneller
    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("📊 Durum vektöründen top-8 öznitelik (z-score)")
        df_state = _state_top_features(snap["state"], top_k=8)
        st.dataframe(df_state, hide_index=True, use_container_width=True, height=310)

        if snap["q_values"] is not None:
            st.plotly_chart(_q_bar(snap["q_values"], snap["action_idx"]),
                            use_container_width=True)

    with right:
        st.plotly_chart(_weights_pie(snap["weights_after"]),
                        use_container_width=True)
        st.plotly_chart(_reward_bar(snap["reward_terms"]),
                        use_container_width=True)

    # --- İflas uyarısı (eğer bu adım bankrupt ise) ---
    if snap["reward_terms"].get("bankrupt"):
        st.error(
            f"💀 **İFLAS** — NAV {snap['nav']:.4f} iflas eşiğinin altına düştü "
            f"(başlangıç sermayesi tükendi). Simülasyon bu adımda sonlandı. "
            f"Uygulanan ek ceza: −{snap['reward_terms'].get('bankruptcy_penalty', 0):.1f} "
            f"(ödüle eklendi)."
        )

    # --- TL paneli: portföy tablosu + işlem logu ---
    st.divider()
    w_prev = snap["weights_before"]
    tl_left, tl_right = st.columns([1.3, 1])
    with tl_left:
        st.subheader("💰 Portföy Tablosu")
        st.caption("Durum: **YENİ** = bu adımda alındı · **TUTULUYOR** = pozisyon devam ediyor · "
                   "**ÇIKIŞ** = bu adımda satıldı")
        df_port = build_portfolio_table(BIST28, tl_now, include_cash=True, w_prev=w_prev)
        st.dataframe(df_port, hide_index=True, use_container_width=True, height=360)
    with tl_right:
        st.subheader("🧾 Bu adımın işlem logu")
        df_trade = build_trade_log(BIST28, tl_now, threshold_tl=1.0)
        if df_trade.empty:
            rebal = env_rebalance_hint(horizon)
            st.info(f"Bu adımda işlem yok (rebalans her {rebal} günde bir).")
        else:
            st.dataframe(df_trade, hide_index=True, use_container_width=True, height=280)
        commission_tl = float(tl_now["commission_tl"])
        st.caption(f"Bu adımda komisyon: **{commission_tl:,.2f} ₺** "
                   f"(tx_cost oranı × portföy TL)")

    # --- Kümülatif işlem geçmişi (başından bu adıma kadar) ---
    st.subheader(f"📋 Kümülatif İşlem Geçmişi (adım 0 → {idx})")
    dates_list = [t["date"] for t in trace]
    df_hist = build_cumulative_trade_log(BIST28, tl_snaps, dates_list,
                                         threshold_tl=1.0, up_to_step=idx)
    if df_hist.empty:
        st.info("Henüz işlem kaydı yok.")
    else:
        total_buys = float(df_hist.loc[df_hist["İşlem"] == "ALIŞ", "TL"].sum())
        total_sells = float(df_hist.loc[df_hist["İşlem"] == "SATIŞ", "TL"].sum())
        hc = st.columns(3)
        hc[0].metric("Toplam işlem", f"{len(df_hist):,}")
        hc[1].metric("Toplam ALIŞ TL", f"{total_buys:,.0f} ₺")
        hc[2].metric("Toplam SATIŞ TL", f"{total_sells:,.0f} ₺")
        # En yeni işlemler üstte
        st.dataframe(df_hist.iloc[::-1].reset_index(drop=True),
                     hide_index=True, use_container_width=True, height=320)

    # --- Kümülatif/Step PnL grafikleri (başlangıçtan bu adıma) ---
    st.subheader("📈 TL Kazanç Zaman Serisi (başlangıçtan bu adıma)")
    cum_arr = np.array([s["cum_pnl_tl"] for s in tl_snaps[: idx + 1]])
    step_arr = np.array([s["step_pnl_tl"] for s in tl_snaps[: idx + 1]])
    steps_idx = np.arange(len(cum_arr))
    pnl_left, pnl_right = st.columns(2)
    with pnl_left:
        fig_cum = go.Figure(go.Scatter(x=steps_idx, y=cum_arr, mode="lines",
                                       name="Kümülatif P&L",
                                       line=dict(color="#1f77b4")))
        fig_cum.add_hline(y=0, line_dash="dot", line_color="#888")
        fig_cum.update_layout(title="Kümülatif Kâr/Zarar (TL)", height=300,
                              margin=dict(t=40, b=30), xaxis_title="Adım",
                              yaxis_title="TL")
        st.plotly_chart(fig_cum, use_container_width=True)
    with pnl_right:
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in step_arr]
        fig_step = go.Figure(go.Bar(x=steps_idx, y=step_arr, marker_color=colors))
        fig_step.update_layout(title="Adım P&L (TL)", height=300,
                               margin=dict(t=40, b=30), xaxis_title="Adım",
                               yaxis_title="TL")
        st.plotly_chart(fig_step, use_container_width=True)

    # Adaptif katsayı zaman serisi (o ana kadar)
    st.subheader("📈 Adaptif katsayılar (baştan bu adıma kadar)")
    rt_hist = pd.DataFrame([t["reward_terms"] for t in trace[: idx + 1]])
    rt_hist["step"] = range(len(rt_hist))
    cc = st.columns(3)
    cc[0].plotly_chart(
        px.line(rt_hist, x="step", y="eta_t", title="η_t (tx cost katsayısı)"),
        use_container_width=True)
    cc[1].plotly_chart(
        px.line(rt_hist, x="step", y="lambda_t", title="λ_t (DD penalty katsayısı)"),
        use_container_width=True)
    cc[2].plotly_chart(
        px.line(rt_hist, x="step", y="tau_t", title="τ_t (DD eşiği)"),
        use_container_width=True)

    # Oynatma döngüsü — session_state.playing true iken otomatik ilerle
    if st.session_state.playing and idx < max_step:
        time.sleep(0.15)
        st.session_state.step_idx += 1
        st.rerun()
    elif st.session_state.playing and idx >= max_step:
        st.session_state.playing = False


# =====================================================================
# Sekme 4 — Karşılaştırma
# =====================================================================
def tab_compare():
    st.header("📊 Karşılaştırma — Ajanlar vs. Baseline'lar")

    if not st.session_state.data_loaded:
        st.warning("Önce sidebar'dan veriyi yükleyin.")
        return

    if st.button("Baseline'ları çalıştır (EW + BuyHold + MeanVar)"):
        with st.spinner("Baseline'lar hesaplanıyor..."):
            px_te = st.session_state.px_te
            st.session_state.baselines = {
                "EqualWeight": equal_weight(px_te),
                "BuyHold":     buy_and_hold_index(px_te),
                "MeanVar":     mean_variance(px_te, lookback=120, rebalance=20),
            }
        st.success("Baseline'lar hazır.")

    trained = st.session_state.trained_agents
    traces = st.session_state.test_traces
    baselines = st.session_state.baselines or {}

    if not trained and not baselines:
        st.info("Henüz eğitilmiş ajan yok. Tab 2'de en az bir ajan eğitin, sonra Tab 3'te trajectory çıkarın.")
        return

    # Metrik tablosu
    rows = {}
    nav_map = {}

    for key, (agent, _curve) in trained.items():
        algo, horizon, adaptive = key
        if key not in traces:
            continue
        tr = traces[key]
        nav = np.array([t["nav"] for t in tr])
        rets = np.array([t["reward_terms"]["gross_port_r"] for t in tr])
        W = np.array([t["weights_after"] for t in tr])
        m = summary(nav, rets, W)
        name = f"{algo}-{horizon}{'·A' if adaptive else ''}"
        rows[name] = m
        nav_map[name] = nav

    for bn, bd in baselines.items():
        m = summary(bd["nav"], bd["rets"], bd.get("weights"))
        rows[bn] = m
        nav_map[bn] = bd["nav"]

    if not rows:
        st.info("Ajan metriklerini görmek için önce Tab 3'te test çalıştırın.")
        return

    met_df = pd.DataFrame(rows).T.round(4)
    st.dataframe(met_df, use_container_width=True)
    st.download_button("metrics.csv olarak indir",
                       data=met_df.to_csv().encode("utf-8"),
                       file_name="metrics.csv", mime="text/csv")

    # Hizalanmış NAV eğrileri
    min_len = min(len(v) for v in nav_map.values())
    nav_df = pd.DataFrame({k: v[-min_len:] for k, v in nav_map.items()})
    nav_df.index = st.session_state.px_te.index[-min_len:]
    fig = go.Figure()
    for c in nav_df.columns:
        fig.add_trace(go.Scatter(x=nav_df.index, y=nav_df[c], name=c, mode="lines"))
    fig.update_layout(title="Test dönemi NAV (başlangıç = 1.0)",
                      height=440, yaxis_title="NAV", xaxis_title="Tarih")
    st.plotly_chart(fig, use_container_width=True)

    # Ağırlık heatmap (seçilebilir)
    st.subheader("🔥 Ağırlık Isı Haritası")
    agent_keys_with_traces = [k for k in trained if k in traces]
    if agent_keys_with_traces:
        chosen = st.selectbox(
            "Ajan seç",
            agent_keys_with_traces,
            format_func=lambda k: f"{k[0]}-{k[1]}{'·A' if k[2] else ''}"
        )
        tr = traces[chosen]
        W = np.array([t["weights_after"] for t in tr])
        # Örnekleme: çok uzunsa her N'incisini al
        step = max(1, len(W) // 200)
        Wp = W[::step]
        fig_hm = px.imshow(
            Wp.T,
            aspect="auto",
            origin="lower",
            color_continuous_scale="RdBu_r",
            zmin=0, zmax=min(0.3, float(Wp.max()) * 1.05) if Wp.max() > 0 else 0.3,
            labels=dict(x="Gün (örneklenmiş)", y="Varlık", color="Ağırlık"),
        )
        fig_hm.update_yaxes(ticktext=ASSET_NAMES,
                            tickvals=list(range(len(ASSET_NAMES))))
        fig_hm.update_layout(height=520, title=f"{chosen[0]} ağırlıkları")
        st.plotly_chart(fig_hm, use_container_width=True)

        # DQN → aksiyon dağılımı
        if chosen[0] == "DQN":
            action_counts = pd.Series(
                [t["action_idx"] for t in tr if t["action_idx"] is not None]
            ).value_counts().reindex(range(6), fill_value=0)
            action_counts.index = ACTION_NAMES[:6]
            fig_a = px.bar(action_counts, title="DQN — aksiyon dağılımı (gün sayısı)")
            fig_a.update_layout(height=320, showlegend=False,
                                yaxis_title="Gün sayısı", xaxis_title="Şablon")
            st.plotly_chart(fig_a, use_container_width=True)

        # Adaptif katsayılar — zaman serisi
        st.subheader("📐 Seçili ajanın adaptif katsayıları (test dönemi)")
        rt_df = pd.DataFrame([t["reward_terms"] for t in tr])
        rt_df["date"] = [t["date"] for t in tr]
        sub = st.columns(3)
        sub[0].plotly_chart(px.line(rt_df, x="date", y="eta_t", title="η_t"),
                            use_container_width=True)
        sub[1].plotly_chart(px.line(rt_df, x="date", y="lambda_t", title="λ_t"),
                            use_container_width=True)
        sub[2].plotly_chart(px.line(rt_df, x="date", y="tau_t", title="τ_t"),
                            use_container_width=True)


# =====================================================================
# MAIN
# =====================================================================
def main():
    st.set_page_config(
        page_title="BIST 28 RL Portföy Demosu",
        page_icon="📈",
        layout="wide",
    )
    _init_state()
    algo, horizon, adaptive, hp = sidebar_controls()

    st.title("📈 BIST 28 Pekiştirmeli Öğrenme Portföy Yönetimi")
    st.caption("UYİK 2026 · DQN/PPO/SAC · Vade preset'leri · Adaptif ödül şekillendirici")

    t1, t2, t3, t4 = st.tabs([
        "📐 Veri & MDP",
        "🎓 Eğitim",
        "🎬 Test (Adım-Adım)",
        "📊 Karşılaştırma",
    ])
    with t1: tab_mdp()
    with t2: tab_train(algo, horizon, adaptive, hp)
    with t3: tab_test(algo, horizon, adaptive)
    with t4: tab_compare()


if __name__ == "__main__":
    main()
