"""Sekme 4 — Karşılaştırma (app.py'den tasindi, P5)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from env.portfolio_env import ACTION_NAMES
from ui.state import ASSET_NAMES
from utils.baselines import buy_and_hold_index, equal_weight, mean_variance
from utils.metrics import summary


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
    st.dataframe(met_df, width='stretch')
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
    st.plotly_chart(fig, width='stretch', key="compare_nav")

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
        st.plotly_chart(fig_hm, width='stretch', key="compare_heatmap")

        # DQN → aksiyon dağılımı
        if chosen[0] == "DQN":
            action_counts = pd.Series(
                [t["action_idx"] for t in tr if t["action_idx"] is not None]
            ).value_counts().reindex(range(6), fill_value=0)
            action_counts.index = ACTION_NAMES[:6]
            fig_a = px.bar(action_counts, title="DQN — aksiyon dağılımı (gün sayısı)")
            fig_a.update_layout(height=320, showlegend=False,
                                yaxis_title="Gün sayısı", xaxis_title="Şablon")
            st.plotly_chart(fig_a, width='stretch', key="compare_dqn_actions")

        # Adaptif katsayılar — zaman serisi
        st.subheader("📐 Seçili ajanın adaptif katsayıları (test dönemi)")
        rt_df = pd.DataFrame([t["reward_terms"] for t in tr])
        rt_df["date"] = [t["date"] for t in tr]
        sub = st.columns(3)
        sub[0].plotly_chart(px.line(rt_df, x="date", y="eta_t", title="η_t"),
                            width='stretch', key="compare_eta_t")
        sub[1].plotly_chart(px.line(rt_df, x="date", y="lambda_t", title="λ_t"),
                            width='stretch', key="compare_lambda_t")
        sub[2].plotly_chart(px.line(rt_df, x="date", y="tau_t", title="τ_t"),
                            width='stretch', key="compare_tau_t")
