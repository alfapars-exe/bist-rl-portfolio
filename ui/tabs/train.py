"""Sekme 2 — Eğitim (canlı) (app.py'den tasindi, P5)."""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data import BIST28
from env.portfolio_env import ACTION_NAMES
from ui.services import train_generator
from ui.state import _agent_key
from utils.metrics import training_diagnostics
from utils.portfolio_tl import (
    build_portfolio_table, build_trade_log, compute_tl_series, step_rows_for_training,
)

# Grafik/tablo serilestirme her N iterde bir (performans — kesif bulgusu:
# onceki surum her iterde 6 plotly + 2 dataframe serialize ediyordu, bu canli
# egitimde %10-40 ek yuk demekti). Metrikler/durum/stop butonu her iter guncel
# kalir; egitim bitiminde son durum HER ZAMAN render edilir.
RENDER_EVERY = 5


def tab_train(algo: str, step_days: int, adaptive: bool, hp: dict):
    st.header(f"🎓 Eğitim — {algo} · {step_days} seans/adım · "
              f"Adaptif: {'Açık' if adaptive else 'Kapalı'}")

    if not st.session_state.data_loaded:
        st.warning("Önce sidebar'dan 'Veriyi Yükle' butonuna basın.")
        return

    key = _agent_key(algo, step_days, adaptive)
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
            _render_episode_browser(key, algo, float(st.session_state.initial_capital))
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
    n_episodes = int(st.session_state.get("n_episodes", 12))
    gen = train_generator(algo, step_days, adaptive, hp,
                          rollout_len=int(hp.get("rollout_len", 400)),
                          resume_agent=resume_agent,
                          n_episodes=n_episodes)
    t0 = time.time()
    iter_times = []  # son N iter süresi (iter/sn için)
    trained_agent = None
    stopped_early = False
    initial_capital = float(st.session_state.initial_capital)

    # İlerleme çubuğu: N episode'a göre doldurulur
    progress_bar = st.progress(0.0, text=f"Episode 0 / {n_episodes}")

    last_rec = None
    # Per-episode telemetri (episode seçici için): her episode'un TL izini sakla (UI-only).
    ep_snaps: list = []
    ep_shared: dict = {"prices": None, "dates": None}
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

        # Per-episode snapshot — selectbox ile sonradan incelemek için (UI-only, golden-etkisiz).
        if ep_shared["prices"] is None:
            ep_shared["prices"] = env.prices
            ep_shared["dates"] = env.dates
        ep_snaps.append({
            "iter": int(d["iter"]),
            "reward": float(rec["reward"]), "nav": float(rec["nav"]),
            "gain": float(rec.get("gain", rec["nav"] - 1.0)),
            "loss": float(rec.get("loss", 0.0)),
            "success": int(rec.get("success", 0)),
            "nav_history": list(env.nav_history),
            "weight_history": [np.asarray(w, dtype=np.float32) for w in env.weight_history],
            "reward_terms_history": [dict(rt) for rt in env.reward_terms_history],
            "t": int(env.t), "step_count": int(env.step_count),
            "actions": list(rec.get("actions") or []),
        })
        st.session_state.setdefault("episode_snaps", {})[key] = ep_snaps
        st.session_state.setdefault("episode_shared", {})[key] = ep_shared

        # Agir serilestirme (4 egri + TL paneli) yalniz her RENDER_EVERY iterde
        render_now = (len(curve) == 1) or (len(curve) % RENDER_EVERY == 0)
        if render_now:
            _render_live_curves(pd.DataFrame(curve),
                                ph_reward, ph_gain, ph_success, ph_loss, seq=d["iter"])

        # --- Throughput metrikleri ---
        iter_end_elapsed = time.time() - t0
        iter_times.append(iter_end_elapsed - iter_start_elapsed)
        recent = iter_times[-5:]
        rate = (len(recent) / sum(recent)) if sum(recent) > 0 else 0.0
        avg = sum(iter_times) / len(iter_times)
        cur_ep = iter_offset + rec["iter"] + 1
        ph_iter.metric("Episode", f"{cur_ep} / {n_episodes}")
        ph_rate.metric("Iter/sn", f"{rate:.2f}")
        ph_avg.metric("Ort. iter süresi", f"{avg:.2f}s")
        mm = int(iter_end_elapsed // 60); ss = int(iter_end_elapsed % 60)
        ph_elapsed.metric("Toplam elapsed", f"{mm:02d}:{ss:02d}")

        # İlerleme çubuğu: Episode i/N
        _prog = min(cur_ep / n_episodes, 1.0)
        progress_bar.progress(_prog, text=f"Episode {cur_ep} / {n_episodes}")

        # --- Canlı TL paneli (son episod için env.nav_history / weight_history kullan) ---
        if render_now:
            _render_train_tl_panel(
                env=env, algo=algo, rec=rec, initial_capital=initial_capital,
                ph_tl_start=ph_tl_start, ph_tl_end=ph_tl_end, ph_tl_net=ph_tl_net,
                ph_tl_min=ph_tl_min, ph_tl_max=ph_tl_max, ph_tl_dd=ph_tl_dd,
                ph_tl_line=ph_tl_line, ph_tl_bar=ph_tl_bar,
                ph_tl_table=ph_tl_table, ph_tl_port=ph_tl_port,
                ph_bankrupt=ph_bankrupt, seq=d["iter"],
            )

        status.info(f"Episode {cur_ep}/{n_episodes} · NAV={rec['nav']:.3f} · "
                    f"elapsed {iter_end_elapsed:.1f}s — "
                    f"'Eğitimi Durdur' ile erken kesilebilir")

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
    progress_bar.progress(1.0, text=f"Tamamlandı — {len(curve)} episode")
    # Son durumu HER ZAMAN render et (throttle yuzunden son iterler atlanmis olabilir).
    # seq="final": dongu-ici render'larin (seq=iter no) HICBIRIYLE cakismaz -> ayni
    # st.empty() slot'una son kez yazar, benzersiz key (StreamlitDuplicateElementKey yok).
    if curve:
        _render_live_curves(pd.DataFrame(curve), ph_reward, ph_gain, ph_success, ph_loss,
                            seq="final")
    if last_rec is not None:
        _render_train_tl_panel(
            env=last_rec["env"], algo=algo, rec=last_rec, initial_capital=initial_capital,
            ph_tl_start=ph_tl_start, ph_tl_end=ph_tl_end, ph_tl_net=ph_tl_net,
            ph_tl_min=ph_tl_min, ph_tl_max=ph_tl_max, ph_tl_dd=ph_tl_dd,
            ph_tl_line=ph_tl_line, ph_tl_bar=ph_tl_bar,
            ph_tl_table=ph_tl_table, ph_tl_port=ph_tl_port,
            ph_bankrupt=ph_bankrupt, seq="final",
        )
    if stopped_early:
        status.warning(f"{algo} eğitimi {len(curve)}. iter sonunda durduruldu "
                       f"({elapsed:.1f}s) — son ajan session'a kaydedildi.")
    else:
        status.success(f"{algo} eğitildi ({len(curve)} iter, {elapsed:.1f}s) "
                       f"ve session'a kaydedildi. Tab 3'te test edebilirsiniz.")

    # --- PDF §9.7 Pedagojik Eğitim Metrikleri ---
    if curve:
        rt_hist = list(last_rec["env"].reward_terms_history) if last_rec is not None else None
        diag = training_diagnostics(curve, reward_terms_history=rt_hist)
        st.markdown("#### §9.7 Eğitim Tanılama Metrikleri")
        d_cols = st.columns(5)
        d_cols[0].metric(
            "Hareketli Ort. Return",
            f"{diag.get('ma_return_last', 0.0):.4f}",
            help="Son 5 episod ödülünün hareketli ortalaması (öğrenme eğilimi)",
        )
        d_cols[1].metric(
            "Başarı Oranı",
            f"{diag.get('success_rate', 0.0):.1%}",
            help="EW benchmark'ı geçen episod oranı",
        )
        if "avg_steps" in diag:
            d_cols[2].metric(
                "Ort. Adım/Episod",
                f"{diag['avg_steps']:.1f}",
                help="Trainer 'steps' alanından hesaplandı",
            )
        else:
            d_cols[2].metric(
                "Ort. Adım/Episod",
                "—",
                help="Trainer kayıtlarında 'steps' alanı yok — trainer'a dokunulmadan atlandı",
            )
        d_cols[3].metric(
            "Drawdown Ceza Adımı",
            str(diag.get("drawdown_penalty_steps", "—")),
            help="reward_terms_history'den: drawdown_penalty > 0 olan adım sayısı",
        )
        d_cols[4].metric(
            "Ort. İşlem Maliyeti",
            f"{diag.get('mean_tx_cost', 0.0):.5f}",
            help="reward_terms_history'den: adım başı ortalama tx_cost",
        )

    # Episode seçici — eğitilen her episode'un detayını (TL izi/grafik/değerler) incele.
    _render_episode_browser(key, algo, initial_capital)


def _render_live_curves(df, ph_reward, ph_gain, ph_success, ph_loss, seq=0):
    """4 canli egitim egrisini placeholder'lara cizer (throttle edilmis cagri).

    seq: render sirasi (iter no). Streamlit 1.5x st.empty() slot'una DONGU icinde
    tekrar cizimde ELEMAN ID'sini her seferinde yeniden kaydeder; sabit key ->
    StreamlitDuplicateElementKey. Render-basina BENZERSIZ key (seq) -> her cizim
    benzersiz; empty() slot yine yalniz son grafigi gosterir (yerinde gunceller).
    """
    fig_r = px.line(df, x="iter", y="reward",
                    title="Kümülatif Ödül (iterasyon başına — çevre ödülü Σr)",
                    markers=True)
    fig_r.update_layout(height=260, margin=dict(t=40, b=20))
    ph_reward.plotly_chart(fig_r, width='stretch', key=f"train_live_reward_{seq}")

    fig_g = px.line(df, x="iter", y="gain",
                    title="Kazanç (nihai NAV − 1.0)",
                    markers=True)
    fig_g.update_layout(height=260, margin=dict(t=40, b=20))
    ph_gain.plotly_chart(fig_g, width='stretch', key=f"train_live_gain_{seq}")

    fig_s = px.bar(df, x="iter", y="success",
                   title="Başarı (EW benchmark'a göre 0/1)")
    fig_s.update_layout(height=260, margin=dict(t=40, b=20),
                        yaxis=dict(range=[0, 1.2], tickvals=[0, 1]))
    ph_success.plotly_chart(fig_s, width='stretch', key=f"train_live_success_{seq}")

    if "loss" in df.columns:
        fig_l = px.line(df, x="iter", y="loss",
                        title="Ortalama loss (düşüş beklenir)",
                        markers=True)
        fig_l.update_layout(height=260, margin=dict(t=40, b=20))
        ph_loss.plotly_chart(fig_l, width='stretch', key=f"train_live_loss_{seq}")


def _render_train_tl_panel(env, algo, rec, initial_capital,
                            ph_tl_start, ph_tl_end, ph_tl_net, ph_tl_min, ph_tl_max, ph_tl_dd,
                            ph_tl_line, ph_tl_bar, ph_tl_table, ph_tl_port,
                            ph_bankrupt=None, seq=0):
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
    ph_tl_line.plotly_chart(fig_tl, width='stretch', key=f"train_tl_line_{seq}")

    # Adım P&L bar chart (yeşil/kırmızı)
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in step_pnl_arr]
    fig_bar = go.Figure(go.Bar(x=idx, y=step_pnl_arr, marker_color=colors))
    fig_bar.update_layout(title="Adım P&L (TL)", height=280,
                          margin=dict(t=40, b=30), xaxis_title="Gün",
                          yaxis_title="TL")
    ph_tl_bar.plotly_chart(fig_bar, width='stretch', key=f"train_tl_bar_{seq}")

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
    ph_tl_table.dataframe(df_rows, hide_index=True, width='stretch', height=500,
                          key=f"train_tl_table_{seq}")

    # Episod sonu portföy panosu (w_prev = sondan bir önceki adım)
    last = snaps[-1]
    w_prev = weight_hist[-2] if len(weight_hist) >= 2 else None
    df_port = build_portfolio_table(BIST28, last, include_cash=True, w_prev=w_prev)
    ph_tl_port.dataframe(df_port, hide_index=True, width='stretch',
                         key=f"train_tl_port_{seq}")

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
                            title="Kümülatif Ödül"), width='stretch',
                    key="train_curve_reward")
    c2.plotly_chart(px.line(df, x="iter", y="gain", markers=True,
                            title="Kazanç (NAV − 1)"), width='stretch',
                    key="train_curve_gain")
    c3.plotly_chart(px.bar(df, x="iter", y="success", title="Başarı (0/1)"),
                    width='stretch', key="train_curve_success")
    if "loss" in df.columns:
        c4.plotly_chart(px.line(df, x="iter", y="loss", markers=True,
                                title="Loss"), width='stretch',
                        key="train_curve_loss")

    # --- PDF §9.7 Pedagojik Eğitim Metrikleri (statik görüntüleme) ---
    diag = training_diagnostics(curve)
    st.markdown("#### §9.7 Eğitim Tanılama Metrikleri")
    d_cols = st.columns(5)
    d_cols[0].metric(
        "Hareketli Ort. Return",
        f"{diag.get('ma_return_last', 0.0):.4f}",
        help="Son 5 episod ödülünün hareketli ortalaması (öğrenme eğilimi)",
    )
    d_cols[1].metric(
        "Başarı Oranı",
        f"{diag.get('success_rate', 0.0):.1%}",
        help="EW benchmark'ı geçen episod oranı",
    )
    if "avg_steps" in diag:
        d_cols[2].metric(
            "Ort. Adım/Episod",
            f"{diag['avg_steps']:.1f}",
            help="Trainer 'steps' alanından hesaplandı",
        )
    else:
        d_cols[2].metric(
            "Ort. Adım/Episod",
            "—",
            help="Trainer kayıtlarında 'steps' alanı yok — trainer'a dokunulmadan atlandı",
        )
    d_cols[3].metric(
        "Drawdown Ceza Adımı",
        str(diag.get("drawdown_penalty_steps", "—")),
        help="reward_terms_history'den: drawdown_penalty > 0 olan adım sayısı (yeniden eğitimde mevcut)",
    )
    d_cols[4].metric(
        "Ort. İşlem Maliyeti",
        f"{diag.get('mean_tx_cost', 0.0):.5f}",
        help="reward_terms_history'den: adım başı ortalama tx_cost (yeniden eğitimde mevcut)",
    )


def _render_episode_browser(key, algo, initial_capital):
    """Saklanan her episode'un TL grafiklerini + değerlerini selectbox ile gösterir.

    Eğitim sırasında `episode_snaps[key]`'e yazılan her episode'un telemetrisi
    (nav/weight/reward_terms history + t/step_count + actions) buradan replay edilir.
    `_render_train_tl_panel` env attribute'larını okuduğundan, snapshot bir
    SimpleNamespace (sahte env) olarak ona geçirilir. UI-only — golden etkisiz.
    """
    if key is None:
        return
    snaps = st.session_state.get("episode_snaps", {}).get(key)
    shared = st.session_state.get("episode_shared", {}).get(key)
    if not snaps or not shared or shared.get("prices") is None:
        return
    st.markdown("---")
    st.markdown("### 🔎 Episode incele (her iterasyonun detayı)")
    n = len(snaps)

    def _label(i):
        s = snaps[i]
        return f"Episode {s['iter'] + 1} / {n}  ·  ödül {s['reward']:.2f} · NAV {s['nav']:.3f}"

    sel = st.selectbox("Hangi episode?", list(range(n)), index=n - 1,
                       format_func=_label, key=f"ep_browse_sel_{key}")
    s = snaps[sel]
    c = st.columns(4)
    c[0].metric("Episode", f"{s['iter'] + 1} / {n}")
    c[1].metric("Kümülatif Ödül (Σr)", f"{s['reward']:.3f}")
    c[2].metric("Kazanç (NAV−1)", f"{s['gain']:+.3f}")
    c[3].metric("Başarı (EW)", "✅" if s["success"] else "—")

    # Canlı TL panelle aynı placeholder yapısı (6 metrik + 2 grafik + tablo + porto)
    mcols = st.columns(6)
    pm = [mcols[i].empty() for i in range(6)]
    ccols = st.columns(2)
    ph_line, ph_bar = ccols[0].empty(), ccols[1].empty()
    ph_table, ph_port, ph_bank = st.empty(), st.empty(), st.empty()

    fake_env = SimpleNamespace(
        nav_history=s["nav_history"], weight_history=s["weight_history"],
        reward_terms_history=s["reward_terms_history"],
        prices=shared["prices"], dates=shared["dates"],
        t=s["t"], step_count=s["step_count"],
    )
    fake_rec = {"actions": s["actions"]}
    _render_train_tl_panel(
        env=fake_env, algo=algo, rec=fake_rec, initial_capital=initial_capital,
        ph_tl_start=pm[0], ph_tl_end=pm[1], ph_tl_net=pm[2],
        ph_tl_min=pm[3], ph_tl_max=pm[4], ph_tl_dd=pm[5],
        ph_tl_line=ph_line, ph_tl_bar=ph_bar,
        ph_tl_table=ph_table, ph_tl_port=ph_port, ph_bankrupt=ph_bank,
        seq=f"browse_{key}_{sel}",
    )

    # ---- Adım kaydırıcısı: seçilen episode'da HER adımdaki aksiyon/holdings/nakit ----
    if int(s["step_count"]) > 1:
        st.markdown("**📋 Adım-adım detay** — bu episode'da seçilen adımda hangi aksiyon, "
                    "hangi hisseler tutuluyor, portföy ve nakit durumu:")
        _t_start = int(s["t"]) - int(s["step_count"])
        _tx = [float(rt.get("tx_cost", 0.0)) for rt in s["reward_terms_history"]]
        _ep_tl = compute_tl_series(
            nav_hist=s["nav_history"], weight_hist=s["weight_history"],
            prices_matrix=shared["prices"], initial_capital=initial_capital,
            tx_cost_rates=_tx, t_start=_t_start,
        )
        if _ep_tl:
            _maxk = len(_ep_tl) - 1
            step_k = st.slider("Adım seç", 0, _maxk, _maxk, key=f"ep_browse_step_{key}_{sel}")
            snap_k = _ep_tl[step_k]
            _acts = s["actions"]
            if algo == "DQN" and step_k < len(_acts):
                _a = int(_acts[step_k])
                act_name = ACTION_NAMES[_a] if 0 <= _a < len(ACTION_NAMES) else str(_a)
            elif algo in ("PPO", "SAC", "TD3"):
                act_name = f"{algo} (sürekli aksiyon)"
            else:
                act_name = "—"
            _di = _t_start + 1 + step_k
            dstr = (str(pd.Timestamp(shared["dates"][_di]).date())
                    if 0 <= _di < len(shared["dates"]) else "")
            sc = st.columns(6)
            sc[0].metric("Adım", f"{step_k + 1} / {int(s['step_count'])}")
            sc[1].metric("Tarih", dstr)
            sc[2].metric("Aksiyon", act_name)
            sc[3].metric("Portföy TL", f"{snap_k['portfolio_tl']:,.0f} ₺")
            sc[4].metric("Nakit TL", f"{snap_k['cash_tl']:,.0f} ₺")
            sc[5].metric("Adım P&L", f"{snap_k['step_pnl_tl']:+,.0f} ₺")
            # w_prev = adımdan ÖNCEKİ ağırlık (weight_history[step_k]); 0'da başlangıç=nakit.
            w_prev_k = s["weight_history"][step_k] if step_k < len(s["weight_history"]) else None
            df_port_k = build_portfolio_table(BIST28, snap_k, include_cash=True, w_prev=w_prev_k)
            df_trade_k = build_trade_log(BIST28, snap_k, threshold_tl=1.0)
            pcol, tcol = st.columns([1.4, 1])
            with pcol:
                st.caption("Portföy — bu adımda tutulan hisseler (ağırlık / lot / TL)")
                st.dataframe(df_port_k, hide_index=True, width='stretch', height=320)
            with tcol:
                st.caption("Bu adımın işlemleri (al / sat)")
                if df_trade_k.empty:
                    st.info("Bu adımda işlem yok (ağırlıklar korunmuş).")
                else:
                    st.dataframe(df_trade_k, hide_index=True, width='stretch', height=280)
