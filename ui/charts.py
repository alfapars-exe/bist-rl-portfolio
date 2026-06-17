"""Saf grafik/tablo ureticileri (app.py'den tasindi, P5) — UI-durumsuz.

Hicbiri st.session_state okumaz; girdi -> figur/DataFrame donusumu yapar.
Bu saflik onlari Streamlit'siz test edilebilir kilar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from config import FEATURES, MacroConfig
from data import BIST28
from env.portfolio_env import ACTION_NAMES, HORIZON_PRESETS
from ui.state import ASSET_NAMES


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
    names = ["log_return", "tx_cost", "dd_penalty", "cvar", "dsr", "total"]
    vals = [rt["log_return"], -rt["tx_cost"], -rt["drawdown_penalty"],
            -rt.get("cvar_penalty", 0.0), rt.get("dsr_term", 0.0), rt["total"]]
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
    # v6: state = [F teknik × n_assets] + [M makro] + [n_assets+1 agirlik]. Makro blogunu
    # cikararak teknik oznitelik sayisi F'i state uzunlugundan turet.
    M = len(MacroConfig.features) if MacroConfig.enabled else 0
    F = (len(state_vec) - M - (n_assets + 1)) // n_assets   # 12 (PPO) ya da 13 (DQN/SAC)
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
