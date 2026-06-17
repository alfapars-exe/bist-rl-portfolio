"""Sekme 2 — Eğitim (canlı) (app.py'den tasindi, P5)."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import BIST28
from env.portfolio_env import ACTION_NAMES
from ui.services import train_generator
from ui.state import _agent_key
from utils.portfolio_tl import (
    build_portfolio_table, compute_tl_series, step_rows_for_training,
)

# Grafik/tablo serilestirme her N iterde bir (performans — kesif bulgusu:
# onceki surum her iterde 6 plotly + 2 dataframe serialize ediyordu, bu canli
# egitimde %10-40 ek yuk demekti). Metrikler/durum/stop butonu her iter guncel
# kalir; egitim bitiminde son durum HER ZAMAN render edilir.
RENDER_EVERY = 5


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

    # G4: 'Devam Et' yalnız in-memory eğitilmiş + eğrisi olan ajan için anlamlı.
    can_resume = already and bool(st.session_state.trained_agents[key][1])
    col_run, col_resume, col_clear = st.columns([1, 1, 1])
    with col_run:
        run = st.button(f"{'Yeniden Eğit' if already else 'Eğit'}", type="primary")
    resume = False
    with col_resume:
        if can_resume:
            resume = st.button("▶ Devam Et",
                               help="Durdurulan eğitime AYNI ajanla (ağırlık+optimizer+buffer) kaldığı yerden devam")
    with col_clear:
        if already and st.button("Bu konfigürasyonu unut"):
            st.session_state.trained_agents.pop(key, None)
            st.session_state.test_traces.pop(key, None)
            st.rerun()

    st.caption("💡 'Eğit' sıfırdan başlatır · '▶ Devam Et' durdurulan eğitimi aynı ajanla sürdürür · "
               "eğitim sırasında **⏹ Eğitimi Durdur** ile istediğin noktada kesebilirsin.")

    if not (run or resume):
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

    # G4: 'Devam Et' ise mevcut ajanı + curve'ü taşı; iter ofseti = önceki iter sayısı.
    resume_agent = st.session_state.trained_agents[key][0] if resume else None
    curve = list(st.session_state.trained_agents[key][1]) if resume else []
    iter_offset = len(curve)
    gen = train_generator(algo, horizon, adaptive, hp,
                          rollout_len=int(hp.get("rollout_len", 400)),
                          resume_agent=resume_agent)
    t0 = time.time()
    iter_times = []  # son N iter süresi (iter/sn için)
    trained_agent = None
    stopped_early = False
    initial_capital = float(st.session_state.initial_capital)

    last_rec = None
    for rec in gen:
        iter_start_elapsed = time.time() - t0
        trained_agent = rec["agent"]
        env = rec["env"]
        last_rec = rec
        d = {k: v for k, v in rec.items() if k not in ("agent", "env", "actions")}
        d["iter"] = iter_offset + rec["iter"]      # G4: devam'da iterasyon numarası süreklilik
        curve.append(d)
        # Her iter sonunda session'a yaz → kullanıcı durdurursa veya refresh etse bile son hali kalır
        st.session_state.trained_agents[key] = (trained_agent, list(curve))

        # Agir serilestirme (4 egri + TL paneli) yalniz her RENDER_EVERY iterde
        render_now = (len(curve) == 1) or (len(curve) % RENDER_EVERY == 0)
        if render_now:
            _render_live_curves(pd.DataFrame(curve),
                                ph_reward, ph_gain, ph_success, ph_loss)

        # --- Throughput metrikleri ---
        iter_end_elapsed = time.time() - t0
        iter_times.append(iter_end_elapsed - iter_start_elapsed)
        recent = iter_times[-5:]
        rate = (len(recent) / sum(recent)) if sum(recent) > 0 else 0.0
        avg = sum(iter_times) / len(iter_times)
        ph_iter.metric("Iter", iter_offset + rec["iter"] + 1)
        ph_rate.metric("Iter/sn", f"{rate:.2f}")
        ph_avg.metric("Ort. iter süresi", f"{avg:.2f}s")
        mm = int(iter_end_elapsed // 60); ss = int(iter_end_elapsed % 60)
        ph_elapsed.metric("Toplam elapsed", f"{mm:02d}:{ss:02d}")

        # --- Canlı TL paneli (son episod için env.nav_history / weight_history kullan) ---
        if render_now:
            _render_train_tl_panel(
                env=env, algo=algo, rec=rec, initial_capital=initial_capital,
                ph_tl_start=ph_tl_start, ph_tl_end=ph_tl_end, ph_tl_net=ph_tl_net,
                ph_tl_min=ph_tl_min, ph_tl_max=ph_tl_max, ph_tl_dd=ph_tl_dd,
                ph_tl_line=ph_tl_line, ph_tl_bar=ph_tl_bar,
                ph_tl_table=ph_tl_table, ph_tl_port=ph_tl_port,
                ph_bankrupt=ph_bankrupt,
            )

        status.info(f"Iter {iter_offset + rec['iter'] + 1} · NAV={rec['nav']:.3f} · "
                    f"elapsed {iter_end_elapsed:.1f}s — "
                    f"istediğin yerde 'Eğitimi Durdur' butonuna basabilirsin")

        # Kullanıcı ayarladığı gecikmeyi iter arası uygula (slider canlı okunur).
        delay = float(st.session_state.get("train_delay", 0.0))
        if delay > 0:
            time.sleep(delay)

        # Döngü içi durdurma — her iter sonunda placeholder'a buton render eder.
        # Kullanıcı basarsa bir sonraki iter başlamaz, o ana kadar eğitilen ajan session'da kalmış olur.
        if stop_slot.button("⏹ Eğitimi Durdur", key=f"stop_loop_{iter_offset}_{rec['iter']}"):
            stopped_early = True
            break

    st.session_state.trained_agents[key] = (trained_agent, curve)
    elapsed = time.time() - t0
    stop_slot.empty()
    # Son durumu HER ZAMAN render et (throttle yuzunden son iterler atlanmis olabilir)
    if curve:
        _render_live_curves(pd.DataFrame(curve), ph_reward, ph_gain, ph_success, ph_loss)
    if last_rec is not None:
        _render_train_tl_panel(
            env=last_rec["env"], algo=algo, rec=last_rec, initial_capital=initial_capital,
            ph_tl_start=ph_tl_start, ph_tl_end=ph_tl_end, ph_tl_net=ph_tl_net,
            ph_tl_min=ph_tl_min, ph_tl_max=ph_tl_max, ph_tl_dd=ph_tl_dd,
            ph_tl_line=ph_tl_line, ph_tl_bar=ph_tl_bar,
            ph_tl_table=ph_tl_table, ph_tl_port=ph_tl_port,
            ph_bankrupt=ph_bankrupt,
        )
    if stopped_early:
        status.warning(f"{algo} eğitimi {len(curve)}. iter sonunda durduruldu "
                       f"({elapsed:.1f}s) — son ajan session'a kaydedildi.")
    else:
        status.success(f"{algo} eğitildi ({len(curve)} iter, {elapsed:.1f}s) "
                       f"ve session'a kaydedildi. Tab 3'te test edebilirsiniz.")


def _render_live_curves(df, ph_reward, ph_gain, ph_success, ph_loss):
    """4 canli egitim egrisini placeholder'lara cizer (throttle edilmis cagri)."""
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
    if not curve:   # diskten yüklenen model — eğitim eğrisi yok
        st.info("Bu model diskten yüklendi (eğitim eğrisi yok). "
                "Test sekmesinde doğrudan çalıştırabilir veya '▶ Devam Et' ile eğitebilirsiniz.")
        return
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
