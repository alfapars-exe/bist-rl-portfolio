"""Sekme 1 — Veri & MDP (app.py'den tasindi, P5)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from data import BIST28, SPLIT, START, END
from ui.charts import _horizon_preset_table


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
            st.plotly_chart(fig_px, width='stretch', key="mdp_px")

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
        st.dataframe(mdp, hide_index=True, width='stretch')

    st.divider()
    st.subheader("⏳ Vade Preset'leri & Adaptif Ödül")
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.dataframe(_horizon_preset_table(), hide_index=True, width='stretch')
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
