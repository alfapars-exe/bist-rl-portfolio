"""Kontrol paneli (sidebar) — app.py'den tasindi (P5)."""
from __future__ import annotations

import datetime

import streamlit as st

from config import SEED, DataConfig, EnvConfig, RewardConfig
from env.portfolio_env import HORIZON_PRESETS
from ui.services import _load_data, load_saved_agent, model_path, save_trained_agent
from ui.state import _agent_key

_dc = DataConfig()
_rc = RewardConfig()

# Tarih aralığı sınırları (UI kısıtı)
_DATE_MIN = datetime.date(2015, 1, 1)
_DATE_MAX = datetime.date(2024, 12, 31)

# SonarCloud S1192: 3+ kez tekrar eden UI literal'leri tek sabitte topla.
_EP_HINT = ("Eğitim **N episode** koşar (sidebar'daki 'Episode sayısı' değeri); "
            "'Eğitimi Durdur' ile erken kesilebilir.")
_LBL_POLICY_LR = "Policy LR"
_LBL_BATCH = "Batch"


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
            value=float(cfg.get("bankruptcy_nav", EnvConfig.bankruptcy_nav)),
            min_value=0.0, max_value=0.9, step=0.01, format="%.2f",
            help="NAV bu eşiğin altına düşerse episod iflas olarak sonlandırılır. "
                 "Örn. 0.01 = başlangıç sermayesinin %1'ine inmek.",
        )
        cfg["bankruptcy_penalty"] = st.number_input(
            "İflas ek ceza değeri",
            value=float(cfg.get("bankruptcy_penalty", EnvConfig.bankruptcy_penalty)),
            min_value=0.0, max_value=1000.0, step=1.0, format="%.1f",
            help="İflas anında toplam ödüle eklenen negatif terim. Log-ölçeğinde büyük değer "
                 "(normal adım ödülü ~±0.01). Ajan iflasa gitmemeyi öğrenir.",
        )

        st.markdown("**Adaptif şekillendirici hedefleri**")
        cfg["vol_target"] = st.number_input(
            "vol_target (hedef realize vol)",
            value=float(cfg.get("vol_target", EnvConfig.vol_target)),
            min_value=0.0001, max_value=0.5, step=0.001, format="%.4f",
            help="Adaptif mod açıkken λ ve τ bu hedefe göre ölçeklenir.",
        )
        cfg["turnover_target"] = st.number_input(
            "turnover_target (hedef turnover)",
            value=float(cfg.get("turnover_target", EnvConfig.turnover_target)),
            min_value=0.001, max_value=1.0, step=0.005, format="%.3f",
            help="Adaptif mod açıkken η bu hedefe göre ölçeklenir.",
        )
        cfg["ema_alpha"] = st.slider(
            "EMA α (adaptif hafıza)",
            min_value=0.001, max_value=0.5, value=float(cfg.get("ema_alpha", EnvConfig.ema_alpha)),
            step=0.005, format="%.3f",
            help="Büyük α = daha hızlı uyum, küçük α = daha stabil.",
        )

        st.markdown("**DSR & CVaR risk terimleri**")
        cfg["w_dsr"] = st.number_input(
            "w_dsr — Diferansiyel Sharpe ağırlığı",
            value=float(cfg.get("w_dsr", _rc.w_dsr)),
            min_value=0.0, max_value=0.2, step=0.005, format="%.3f",
            help="DSR terimi ağırlığı: online risk-ayarlı Sharpe gradyanı. "
                 "0 = kapalı, 0.05 = hafif etkin.",
        )
        cfg["w_cvar"] = st.number_input(
            "w_cvar — CVaR kuyruk cezası ağırlığı",
            value=float(cfg.get("w_cvar", _rc.w_cvar)),
            min_value=0.0, max_value=0.2, step=0.005, format="%.3f",
            help="CVaR (Conditional Value at Risk) ceza ağırlığı. "
                 "0 = kapalı; kriz dönemlerinde regime_beta ile amplify edilir.",
        )
        cfg["dsr_eta"] = st.number_input(
            "dsr_eta — DSR EWMA oranı",
            value=float(cfg.get("dsr_eta", _rc.dsr_eta)),
            min_value=0.001, max_value=0.1, step=0.001, format="%.3f",
            help="Diferansiyel Sharpe hesabındaki EWMA pencere oranı. "
                 "Küçük = yavaş adaptasyon, büyük = hızlı.",
        )
        cfg["cvar_alpha"] = st.number_input(
            "cvar_alpha — CVaR kuyruk yüzdesi",
            value=float(cfg.get("cvar_alpha", _rc.cvar_alpha)),
            min_value=0.01, max_value=0.2, step=0.005, format="%.3f",
            help="CVaR için kuyruk yüzdesi (α). 0.05 = en kötü %5'lik getiri ortalaması.",
        )
        cfg["regime_beta"] = st.number_input(
            "regime_beta — Kriz amplifikasyon gücü",
            value=float(cfg.get("regime_beta", _rc.regime_beta)),
            min_value=0.0, max_value=5.0, step=0.1, format="%.2f",
            help="CVaR cezasını kriz rejiminde büyüten çarpan. "
                 "0 = rejim bağımsız, 5 = kriz anında 6× ceza.",
        )
        cfg["cvar_amp"] = st.number_input(
            "cvar_amp — Rejim amplifikasyon üsteli",
            value=float(cfg.get("cvar_amp", _rc.cvar_amp)),
            min_value=0.5, max_value=3.0, step=0.1, format="%.2f",
            help="κ = w_cvar·(1 + regime_beta·max(0,regime))^cvar_amp formülündeki üstel. "
                 "1.0 = doğrusal amplifikasyon.",
        )

        st.caption("⚠️ Bu ayarları değiştirdikten sonra ajanları **yeniden eğitmek** "
                   "anlamlı olur; eski ajan farklı ortamda öğrenilmiştir.")

    with st.sidebar.expander("🧪 Deneysel ödül terimleri (opt-in, varsayılan kapalı)",
                             expanded=False):
        st.caption(
            "Bu terimler varsayılan 0 ile tamamen kapalıdır — aktif etmek için "
            "sıfırdan farklı değer girin. Yeni ajan eğitmeden etkisi görülmez."
        )
        cfg["w_gain"] = st.number_input(
            "w_gain — Kazanç-çarpanı ödülü ağırlığı",
            value=float(cfg.get("w_gain", _rc.w_gain)),
            min_value=0.0, max_value=1.0, step=0.05, format="%.2f",
            help="NAV gain_floor eşiğini aştığında verilen ödül ağırlığı. "
                 "2× → w_gain ödül, 3× → 2·w_gain ödül. 0 = kapalı.",
        )
        cfg["gain_floor"] = st.number_input(
            "gain_floor — Ödül eşiği (NAV)",
            value=float(cfg.get("gain_floor", _rc.gain_floor)),
            min_value=1.0, max_value=2.0, step=0.05, format="%.2f",
            help="w_gain ödülünün başlayacağı NAV çarpanı. "
                 "1.0 = başlangıçtan itibaren, 1.5 = %50 büyüme sonrası.",
        )
        cfg["w_gain_speed"] = st.number_input(
            "w_gain_speed — Hız bonusu ağırlığı",
            value=float(cfg.get("w_gain_speed", _rc.w_gain_speed)),
            min_value=0.0, max_value=2.0, step=0.05, format="%.2f",
            help="Erken büyümeye daha yüksek ödül veren hız faktörü. "
                 "0 = zamandan bağımsız, pozitif = erken kazanç daha değerli.",
        )
        cfg["w_ruin_timing"] = st.number_input(
            "w_ruin_timing — İflas-timing ceza ağırlığı",
            value=float(cfg.get("w_ruin_timing", _rc.w_ruin_timing)),
            min_value=0.0, max_value=3.0, step=0.1, format="%.2f",
            help="Erken iflas anına daha sert ceza uygular. "
                 "0 = düz (flat) iflas_penalty, pozitif = erken iflasa üstel ceza.",
        )


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

    st.session_state.n_episodes = st.sidebar.number_input(
        "Episode / iterasyon sayısı",
        min_value=1, max_value=1000,
        value=int(st.session_state.n_episodes),
        step=1,
        help="Eğitim tam bu kadar episode/iterasyon koşar; her episode train fiyatlarının "
             "FARKLI gürültülü realizasyonudur. 'Eğitimi Durdur' erken kesebilir. "
             "(PPO için birim 'update', diğerleri 'episode'.)",
    )

    st.session_state.price_noise_std = st.sidebar.slider(
        "Fiyat gürültüsü σ (anti-ezber)",
        min_value=0.0, max_value=0.01,
        value=float(st.session_state.price_noise_std),
        step=0.0005, format="%.4f",
        help="Her episode train hisse getirilerine eklenen minik gürültü (virgül sonrası "
             "basamaklara etki eder). Eğitim-YALNIZ; eval'de hep KAPALI. Her episode farklı "
             "realizasyon → ezberi önler. 0 = kapalı.",
    )

    st.sidebar.divider()
    _algos = ["DQN", "PPO", "SAC", "TD3"]
    _cur = st.session_state.selected_algo if st.session_state.selected_algo in _algos else "DQN"
    algo = st.sidebar.radio("Ajan", _algos, index=_algos.index(_cur))
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

    # ------------------------------------------------------------------
    # Tarih seçici — train/test aralığı
    # ------------------------------------------------------------------
    with st.sidebar.expander("📅 Tarih Aralığı", expanded=False):
        st.caption(
            "Eğitim başlangıcı → Train/Test ayırım → Test bitişi. "
            "Ayırım sonrası veriler test dönemi olarak kullanılır."
        )
        _start_val = datetime.date.fromisoformat(
            st.session_state.get("data_start", _dc.start)
        )
        _split_val = datetime.date.fromisoformat(
            st.session_state.get("data_split", _dc.train_end)
        )
        _end_val = datetime.date.fromisoformat(
            st.session_state.get("data_end", _dc.end)
        )

        sel_start = st.date_input(
            "Train başlangıcı",
            value=_start_val,
            min_value=_DATE_MIN,
            max_value=_DATE_MAX,
            key="ui_data_start",
            help="Eğitim verisinin başlangıç tarihi (dahil).",
        )
        sel_split = st.date_input(
            "Train/Test ayırım tarihi",
            value=_split_val,
            min_value=_DATE_MIN,
            max_value=_DATE_MAX,
            key="ui_data_split",
            help="Bu tarihten itibaren test verisi başlar (dahil). "
                 "Scaler/forecaster yalnız eğitim kısmında fit edilir (sızıntı yok).",
        )
        sel_end = st.date_input(
            "Test bitişi",
            value=_end_val,
            min_value=_DATE_MIN,
            max_value=_DATE_MAX,
            key="ui_data_end",
            help="Test verisinin bitiş tarihi (dahil).",
        )

        # Sızıntı / tutarlılık doğrulaması
        _date_valid = (sel_start < sel_split <= sel_end)
        if not _date_valid:
            st.sidebar.error(
                "Tarih hatası: Train başlangıcı < Ayırım tarihi ≤ Test bitişi "
                "koşulu sağlanmalı. Veriyi yükleyemezsiniz."
            )
        else:
            st.session_state.data_start = sel_start.isoformat()
            st.session_state.data_split = sel_split.isoformat()
            st.session_state.data_end   = sel_end.isoformat()
            st.caption(
                f"Eğitim: {sel_start} → {sel_split}  |  "
                f"Test: {sel_split} → {sel_end}"
            )

    if not st.session_state.data_loaded:
        _date_valid_outer = (
            datetime.date.fromisoformat(st.session_state.get("data_start", _dc.start))
            < datetime.date.fromisoformat(st.session_state.get("data_split", _dc.train_end))
            <= datetime.date.fromisoformat(st.session_state.get("data_end", _dc.end))
        )
        if st.sidebar.button(
            "Veriyi Yükle / İndir",
            use_container_width=True,
            disabled=not _date_valid_outer,
        ):
            _load_data()
            st.rerun()
        if not _date_valid_outer:
            st.sidebar.caption("Tarih aralığı geçersiz — düzeltin.")
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
    # γ (discount): vade preset default'u; override edilirse hp üzerinden TÜM ajanlara uygulanır.
    hp["gamma"] = st.sidebar.number_input(
        "γ (discount / iskonto)", value=float(preset["gamma"]),
        min_value=0.90, max_value=0.999, step=0.005, format="%.3f",
        key=f"gamma_{st.session_state.horizon}",
        help="İskonto faktörü. Vade preset default verir (Kısa 0.95 / Orta 0.99 / Uzun 0.995); "
             "burada değiştirilebilir — DQN/PPO/SAC/TD3'ün hepsine uygulanır.",
    )
    if algo == "DQN":
        st.sidebar.caption(_EP_HINT)
        hp["lr"]        = st.sidebar.select_slider("Öğrenme oranı",
            options=[1e-4, 3e-4, 5e-4, 1e-3, 3e-3], value=1e-3)
        hp["eps_decay"] = st.sidebar.slider("ε decay adımı", 2_000, 30_000, 10_000, step=1_000)
        hp["batch_size"]= st.sidebar.select_slider(_LBL_BATCH, options=[32, 64, 128], value=64)
        hp["target_update"] = st.sidebar.slider("Target sync", 100, 2000, 500, step=100)
    elif algo == "PPO":
        st.sidebar.caption(_EP_HINT.replace("episode", "update"))
        hp["rollout_len"]= st.sidebar.slider("Rollout uzunluğu", 128, 1024, 400, step=64)
        hp["lr_p"]       = st.sidebar.select_slider(_LBL_POLICY_LR,
            options=[1e-4, 3e-4, 1e-3], value=3e-4)
        hp["lr_v"]       = st.sidebar.select_slider("Value LR",
            options=[3e-4, 1e-3, 3e-3], value=1e-3)
        hp["clip"]       = st.sidebar.slider("Clip ε", 0.05, 0.4, 0.2, step=0.05)
        hp["ent_coef"]   = st.sidebar.select_slider("Entropi katsayısı",
            options=[0.0, 0.001, 0.005, 0.01, 0.02], value=0.005)
        hp["batch_size"] = st.sidebar.select_slider("Mini-batch", options=[64, 128, 256], value=128)
        hp["n_epochs"]   = st.sidebar.slider("Epoch", 2, 10, 6, step=1)
    elif algo == "SAC":
        st.sidebar.caption(_EP_HINT)
        hp["lr_pi"]      = st.sidebar.select_slider(_LBL_POLICY_LR,
            options=[1e-4, 3e-4, 1e-3], value=3e-4)
        hp["lr_q"]       = st.sidebar.select_slider("Q LR",
            options=[3e-4, 5e-4, 1e-3], value=5e-4)
        hp["alpha"]      = st.sidebar.slider("Entropi α", 0.0, 0.5, 0.05, step=0.01)
        hp["tau"]        = st.sidebar.select_slider("Soft update τ",
            options=[0.005, 0.01, 0.05], value=0.01)
        hp["batch_size"] = st.sidebar.select_slider(_LBL_BATCH, options=[64, 128, 256], value=128)
    else:  # TD3 — sürekli/deterministik politika (hocanın tavsiyesi)
        st.sidebar.caption(_EP_HINT)
        hp["lr_pi"]      = st.sidebar.select_slider(_LBL_POLICY_LR,
            options=[1e-4, 3e-4, 1e-3], value=3e-4)
        hp["lr_q"]       = st.sidebar.select_slider("Q LR",
            options=[1e-4, 3e-4, 5e-4, 1e-3], value=3e-4)
        hp["policy_noise"] = st.sidebar.slider("Hedef-politika gürültüsü", 0.0, 0.5, 0.2, step=0.05,
            help="Hedef aksiyona eklenen clamped Gauss gürültüsü (TD3 smoothing).")
        hp["expl_noise"] = st.sidebar.slider("Keşif gürültüsü", 0.0, 0.5, 0.1, step=0.05,
            help="Eğitimde aksiyona eklenen keşif gürültüsü (eval'de kapalı).")
        hp["tau"]        = st.sidebar.select_slider("Soft update τ",
            options=[0.005, 0.01, 0.05], value=0.005)
        hp["batch_size"] = st.sidebar.select_slider(_LBL_BATCH, options=[64, 128, 256], value=128)

    # 💾 Model kalıcılığı (PDF §11): eğitilmiş modeli diske kaydet / diskten yükle.
    st.sidebar.divider()
    st.sidebar.subheader("💾 Model (kaydet / yükle)")
    cur_key = _agent_key(algo, st.session_state.horizon, st.session_state.adaptive)
    has_trained = (cur_key in st.session_state.trained_agents
                   and st.session_state.trained_agents[cur_key][0] is not None)
    if has_trained and st.sidebar.button("💾 Eğitilmiş modeli kaydet", use_container_width=True):
        p = save_trained_agent(algo, st.session_state.horizon, st.session_state.adaptive)
        st.sidebar.success(f"Kaydedildi: {p}")
    if model_path(algo).exists():
        if st.sidebar.button(f"📂 Kaydedilmiş {algo} modelini yükle", use_container_width=True):
            load_saved_agent(algo)
            st.sidebar.success(f"{algo} modeli yüklendi — Test sekmesinde çalıştırılabilir.")
            st.rerun()
    else:
        st.sidebar.caption(f"Kayıtlı {algo} modeli yok (CLI ile üret: python main.py).")

    st.sidebar.caption(f"Seed: {SEED} (sabit)")
    return algo, st.session_state.horizon, st.session_state.adaptive, hp
