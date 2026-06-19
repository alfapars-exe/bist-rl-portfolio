"""Sekme 3 — Test (adım adım oynatma) (app.py'den tasindi, P5)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import BIST28
from ui.charts import _q_bar, _reward_bar, _state_top_features, _weights_pie
from ui.services import _compute_test_tl_snaps, evaluate_with_trace
from ui.state import _agent_key, env_rebalance_hint
from utils.portfolio_tl import (
    build_cumulative_trade_log, build_portfolio_table, build_trade_log,
)


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
                            use_container_width=True, key="test_q_bar")

    with right:
        st.plotly_chart(_weights_pie(snap["weights_after"]),
                        use_container_width=True, key="test_weights_pie")
        st.plotly_chart(_reward_bar(snap["reward_terms"]),
                        use_container_width=True, key="test_reward_bar")

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
        st.plotly_chart(fig_cum, use_container_width=True, key="test_cum_pnl")
    with pnl_right:
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in step_arr]
        fig_step = go.Figure(go.Bar(x=steps_idx, y=step_arr, marker_color=colors))
        fig_step.update_layout(title="Adım P&L (TL)", height=300,
                               margin=dict(t=40, b=30), xaxis_title="Adım",
                               yaxis_title="TL")
        st.plotly_chart(fig_step, use_container_width=True, key="test_step_pnl")

    # Adaptif katsayı zaman serisi (o ana kadar)
    st.subheader("📈 Adaptif katsayılar (baştan bu adıma kadar)")
    rt_hist = pd.DataFrame([t["reward_terms"] for t in trace[: idx + 1]])
    rt_hist["step"] = range(len(rt_hist))
    cc = st.columns(3)
    cc[0].plotly_chart(
        px.line(rt_hist, x="step", y="eta_t", title="η_t (tx cost katsayısı)"),
        use_container_width=True, key="test_eta_t")
    cc[1].plotly_chart(
        px.line(rt_hist, x="step", y="lambda_t", title="λ_t (DD penalty katsayısı)"),
        use_container_width=True, key="test_lambda_t")
    cc[2].plotly_chart(
        px.line(rt_hist, x="step", y="tau_t", title="τ_t (DD eşiği)"),
        use_container_width=True, key="test_tau_t")

    # Oynatma döngüsü — session_state.playing true iken otomatik ilerle
    if st.session_state.playing and idx < max_step:
        time.sleep(0.15)
        st.session_state.step_idx += 1
        st.rerun()
    elif st.session_state.playing and idx >= max_step:
        st.session_state.playing = False
