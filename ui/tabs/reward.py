"""Sekme 5 — Odul & Ceza Tasarimi (SRP: yalniz UI katmani).

Bu sekme reward_cfg dict'ini doldurur; env/reward.py veya egitim cekirdegine dokunmaz.
Tum widget key'leri 'rt_' onekliyle sidebar'dan ayrilir (widget-state desync onlenir).
"""
from __future__ import annotations

import streamlit as st

from config import EnvConfig, RewardConfig, HORIZON_PRESETS
from core.persistence import (
    save_reward_preset,
    load_reward_preset,
    list_reward_presets,
)

_rc = RewardConfig()
_ec = EnvConfig()

# medium preset = goldenle uyumlu default
_PRESET = HORIZON_PRESETS["medium"]

# Onizleme icin sabit adim sayisi (MAX_EPISODE_STEPS ~ 252)
_MAX_STEPS = _ec.max_episode_steps  # 252


def _gain_bonus(nav: float, step_frac: float,
                w_gain: float, gain_floor: float, w_gain_speed: float) -> float:
    """Kazanc-carpani bonus formulu (pür-Python, yan etki yok).

    gain_bonus = w_gain * max(0, nav - gain_floor) * (1 + w_gain_speed * (1 - step_frac))
    """
    if w_gain == 0.0:
        return 0.0
    excess = max(0.0, nav - gain_floor)
    speed = 1.0 + w_gain_speed * (1.0 - step_frac)
    return w_gain * excess * speed


def _ruin_penalty(bankruptcy_penalty: float, w_ruin_timing: float,
                  step_frac: float) -> float:
    """Iflas-timing ceza formulu.

    ruin_pen = bankruptcy_penalty * (1 + w_ruin_timing * (1 - step_frac))
    """
    return bankruptcy_penalty * (1.0 + w_ruin_timing * (1.0 - step_frac))


def tab_reward():
    """Odul & Ceza Tasarimi sekmesi — reward_cfg'yi doldurur."""
    st.header("⚖️ Ödül & Ceza Tasarımı")
    st.caption(
        "Ödül = log-getiri − η·turnover − λ·max(0, DD−τ) − iflas cezası "
        "+ DSR − CVaR + kazanç-bonusu.  \n"
        "Bu sekmedeki değerler `reward_cfg`'e yazılır; yeni ajan eğitince etkili olur. "
        "**CLI/golden bunlardan ETKİLENMEZ** (preset kullanır)."
    )

    cfg = st.session_state.setdefault("reward_cfg", {})

    # ------------------------------------------------------------------ #
    # 1) Adaptif sekillendirici
    # ------------------------------------------------------------------ #
    with st.expander("⚙️ Adaptif Şekillendirici", expanded=True):
        st.caption(
            "Açıkken η_t, λ_t, τ_t rolling vol & turnover EWMA'larına göre ölçeklenir."
        )
        adaptive_val = st.checkbox(
            "Adaptif ödül aktif",
            value=bool(st.session_state.get("adaptive", True)),
            key="rt_adaptive",
            help="Açıkken hedef vol/turnover'a göre katsayılar dinamik ölçeklenir.",
        )
        st.session_state["adaptive"] = adaptive_val
        cfg["adaptive"] = adaptive_val

        c1, c2, c3 = st.columns(3)
        with c1:
            vol_t = st.number_input(
                "vol_target",
                min_value=0.0001, max_value=0.5,
                value=float(cfg.get("vol_target", _ec.vol_target)),
                step=0.001, format="%.4f",
                key="rt_vol_target",
                help="Hedef realize vol; adaptif mod bu hedefe göre λ ve τ'yi ölçekler.",
            )
            cfg["vol_target"] = vol_t
        with c2:
            turn_t = st.number_input(
                "turnover_target",
                min_value=0.001, max_value=1.0,
                value=float(cfg.get("turnover_target", _ec.turnover_target)),
                step=0.005, format="%.3f",
                key="rt_turnover_target",
                help="Hedef turnover; adaptif mod bu hedefe göre η'yı ölçekler.",
            )
            cfg["turnover_target"] = turn_t
        with c3:
            ema_a = st.slider(
                "EMA α (adaptif hafıza)",
                min_value=0.001, max_value=0.5,
                value=float(cfg.get("ema_alpha", _ec.ema_alpha)),
                step=0.005, format="%.3f",
                key="rt_ema_alpha",
                help="Büyük α = hızlı uyum, küçük α = stabil.",
            )
            cfg["ema_alpha"] = ema_a

    # ------------------------------------------------------------------ #
    # 2) Temel odul/ceza katsayilari
    # ------------------------------------------------------------------ #
    with st.expander("💸 Temel Ödül / Ceza Katsayıları", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            eta = st.number_input(
                "η — İşlem (turnover) maliyeti",
                min_value=0.0, max_value=0.1,
                value=float(cfg.get("eta_base", _PRESET["eta"])),
                step=0.0001, format="%.4f",
                key="rt_eta_base",
                help="Ağırlık değişiminin L1 normu bu katsayı ile çarpılıp ödülden düşülür.",
            )
            cfg["eta_base"] = eta
        with c2:
            lam = st.number_input(
                "λ — Drawdown ceza katsayısı",
                min_value=0.0, max_value=10.0,
                value=float(cfg.get("lambda_base", _PRESET["lam"])),
                step=0.05, format="%.3f",
                key="rt_lambda_base",
                help="max(0, DD − τ) bu katsayı ile çarpılıp ödülden düşülür.",
            )
            cfg["lambda_base"] = lam
        with c3:
            tau = st.number_input(
                "τ — DD eşiği (oran)",
                min_value=0.0, max_value=0.5,
                value=float(cfg.get("tau_base", _PRESET["tau"])),
                step=0.005, format="%.3f",
                key="rt_tau_base",
                help="Tepe-den DD > τ olduğunda ceza başlar. 0.05 = %5 tolerans.",
            )
            cfg["tau_base"] = tau

    # ------------------------------------------------------------------ #
    # 3) Iflas parametreleri
    # ------------------------------------------------------------------ #
    with st.expander("🛑 İflas (Simülasyonu Durdurma)", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            bnav = st.number_input(
                "İflas NAV eşiği",
                min_value=0.0, max_value=0.9,
                value=float(cfg.get("bankruptcy_nav", _ec.bankruptcy_nav)),
                step=0.01, format="%.2f",
                key="rt_bankruptcy_nav",
                help="NAV bu eşiğin altına düşerse episod iflas olarak biter.",
            )
            cfg["bankruptcy_nav"] = bnav
        with c2:
            bpen = st.number_input(
                "İflas ek ceza değeri",
                min_value=0.0, max_value=1000.0,
                value=float(cfg.get("bankruptcy_penalty", _ec.bankruptcy_penalty)),
                step=1.0, format="%.1f",
                key="rt_bankruptcy_penalty",
                help="İflas anında toplam ödüle eklenen negatif terim.",
            )
            cfg["bankruptcy_penalty"] = bpen

    # ------------------------------------------------------------------ #
    # 4) Risk terimleri (DSR / CVaR)
    # ------------------------------------------------------------------ #
    with st.expander("📉 Risk Terimleri (DSR / CVaR)", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            w_dsr = st.number_input(
                "w_dsr — DSR ağırlığı",
                min_value=0.0, max_value=0.2,
                value=float(cfg.get("w_dsr", _rc.w_dsr)),
                step=0.005, format="%.3f",
                key="rt_w_dsr",
                help="Diferansiyel Sharpe ödül terimi ağırlığı. 0 = kapalı.",
            )
            cfg["w_dsr"] = w_dsr
            dsr_eta = st.number_input(
                "dsr_eta — DSR EWMA",
                min_value=0.001, max_value=0.1,
                value=float(cfg.get("dsr_eta", _rc.dsr_eta)),
                step=0.001, format="%.3f",
                key="rt_dsr_eta",
                help="DSR hesabındaki EWMA pencere oranı.",
            )
            cfg["dsr_eta"] = dsr_eta
        with c2:
            w_cvar = st.number_input(
                "w_cvar — CVaR ağırlığı",
                min_value=0.0, max_value=0.2,
                value=float(cfg.get("w_cvar", _rc.w_cvar)),
                step=0.005, format="%.3f",
                key="rt_w_cvar",
                help="CVaR kuyruk cezası ağırlığı. 0 = kapalı.",
            )
            cfg["w_cvar"] = w_cvar
            cvar_alpha = st.number_input(
                "cvar_alpha — CVaR kuyruk %",
                min_value=0.01, max_value=0.2,
                value=float(cfg.get("cvar_alpha", _rc.cvar_alpha)),
                step=0.005, format="%.3f",
                key="rt_cvar_alpha",
                help="CVaR kuyruk yüzdesi. 0.05 = en kötü %5.",
            )
            cfg["cvar_alpha"] = cvar_alpha
        with c3:
            regime_beta = st.number_input(
                "regime_beta — Kriz amp.",
                min_value=0.0, max_value=5.0,
                value=float(cfg.get("regime_beta", _rc.regime_beta)),
                step=0.1, format="%.2f",
                key="rt_regime_beta",
                help="CVaR cezasını kriz rejiminde büyüten çarpan.",
            )
            cfg["regime_beta"] = regime_beta
            cvar_amp = st.number_input(
                "cvar_amp — Amp. üsteli",
                min_value=0.5, max_value=3.0,
                value=float(cfg.get("cvar_amp", _rc.cvar_amp)),
                step=0.1, format="%.2f",
                key="rt_cvar_amp",
                help="Rejim amplifikasyon üsteli κ formülünde.",
            )
            cfg["cvar_amp"] = cvar_amp

    # ------------------------------------------------------------------ #
    # 5) Kazanc-carpani odul (OPT-IN, default 0 = kapali)
    # ------------------------------------------------------------------ #
    with st.expander("🚀 Kazanç-Çarpanı Ödülü (OPT-IN, varsayılan kapalı)", expanded=False):
        st.caption(
            "w_gain = 0 iken tamamen kapalıdır (golden etkilenmez). "
            "Formül: `gain_bonus = w_gain × max(0, nav − gain_floor) × (1 + w_gain_speed × (1 − step_frac))`"
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            w_gain = st.number_input(
                "w_gain — Kazanç-çarpanı ödülü",
                min_value=0.0, max_value=1.0,
                value=float(cfg.get("w_gain", _rc.w_gain)),
                step=0.05, format="%.2f",
                key="rt_w_gain",
                help="NAV gain_floor eşiğini aştığında verilen ödül ağırlığı. 0 = kapalı.",
            )
            cfg["w_gain"] = w_gain
        with c2:
            gain_floor = st.number_input(
                "gain_floor — Ödül eşiği (NAV)",
                min_value=1.0, max_value=2.0,
                value=float(cfg.get("gain_floor", _rc.gain_floor)),
                step=0.05, format="%.2f",
                key="rt_gain_floor",
                help="w_gain ödülünün başlayacağı NAV çarpanı. 1.0 = baştan, 1.5 = %50 büyüme sonrası.",
            )
            cfg["gain_floor"] = gain_floor
        with c3:
            w_gain_speed = st.number_input(
                "w_gain_speed — Hız bonusu",
                min_value=0.0, max_value=2.0,
                value=float(cfg.get("w_gain_speed", _rc.w_gain_speed)),
                step=0.05, format="%.2f",
                key="rt_w_gain_speed",
                help="Erken büyümeye daha yüksek ödül. 0 = zamandan bağımsız.",
            )
            cfg["w_gain_speed"] = w_gain_speed

        # Canli onizleme
        st.markdown("**Canlı Önizleme**")
        if w_gain == 0.0:
            st.info("Kapalı (w_gain = 0) — kazanç-çarpanı ödülü aktif değil.")
        else:
            bonus_2x_start = _gain_bonus(2.0, 0.0, w_gain, gain_floor, w_gain_speed)
            bonus_2x_end   = _gain_bonus(2.0, 1.0, w_gain, gain_floor, w_gain_speed)
            bonus_3x_start = _gain_bonus(3.0, 0.0, w_gain, gain_floor, w_gain_speed)
            bonus_3x_end   = _gain_bonus(3.0, 1.0, w_gain, gain_floor, w_gain_speed)

            prev_c1, prev_c2 = st.columns(2)
            with prev_c1:
                st.metric(
                    label="NAV 2× (nav=2.0) — episod BAŞINDA",
                    value=f"{bonus_2x_start:.4f}",
                    help="step_frac=0 (1. adım)",
                )
                st.metric(
                    label="NAV 2× (nav=2.0) — episod SONUNDA",
                    value=f"{bonus_2x_end:.4f}",
                    delta=f"{bonus_2x_end - bonus_2x_start:+.4f} vs başı",
                    help=f"step_frac≈1 ({_MAX_STEPS}. adım)",
                )
            with prev_c2:
                st.metric(
                    label="NAV 3× (nav=3.0) — episod BAŞINDA",
                    value=f"{bonus_3x_start:.4f}",
                    help="step_frac=0 (1. adım)",
                )
                st.metric(
                    label="NAV 3× (nav=3.0) — episod SONUNDA",
                    value=f"{bonus_3x_end:.4f}",
                    delta=f"{bonus_3x_end - bonus_3x_start:+.4f} vs başı",
                    help=f"step_frac≈1 ({_MAX_STEPS}. adım)",
                )
            st.caption(
                f"Formül: `gain_bonus = {w_gain} × max(0, nav − {gain_floor}) "
                f"× (1 + {w_gain_speed} × (1 − step_frac))`  |  "
                f"max_steps = {_MAX_STEPS}"
            )

    # ------------------------------------------------------------------ #
    # 6) Iflas-zamani cezasi (OPT-IN, default 0 = kapali)
    # ------------------------------------------------------------------ #
    with st.expander("⏱️ İflas-Zamanı Cezası (OPT-IN, varsayılan kapalı)", expanded=False):
        st.caption(
            "w_ruin_timing = 0 iken düz (flat) iflas_penalty uygulanır. "
            "Formül: `ruin_pen = bankruptcy_penalty × (1 + w_ruin_timing × (1 − step_frac))`"
        )
        w_ruin = st.number_input(
            "w_ruin_timing — İflas-timing ceza ağırlığı",
            min_value=0.0, max_value=3.0,
            value=float(cfg.get("w_ruin_timing", _rc.w_ruin_timing)),
            step=0.1, format="%.2f",
            key="rt_w_ruin_timing",
            help="Erken iflasa daha sert ceza. 0 = düz bankruptcy_penalty.",
        )
        cfg["w_ruin_timing"] = w_ruin

        # Canli onizleme
        st.markdown("**Canlı Önizleme** — İflas cezası episodun farklı anlarında")
        bpen_now = float(cfg.get("bankruptcy_penalty", _ec.bankruptcy_penalty))
        ruin_early  = _ruin_penalty(bpen_now, w_ruin, 0.05)
        ruin_mid    = _ruin_penalty(bpen_now, w_ruin, 0.50)
        ruin_late   = _ruin_penalty(bpen_now, w_ruin, 0.99)

        prev_c1, prev_c2, prev_c3 = st.columns(3)
        with prev_c1:
            st.metric(
                "Erken iflas (step_frac=0.05)",
                f"{ruin_early:.2f}",
                delta=f"{ruin_early - bpen_now:+.2f} vs flat",
            )
        with prev_c2:
            st.metric(
                "Orta iflas (step_frac=0.50)",
                f"{ruin_mid:.2f}",
                delta=f"{ruin_mid - bpen_now:+.2f} vs flat",
            )
        with prev_c3:
            st.metric(
                "Geç iflas (step_frac=0.99)",
                f"{ruin_late:.2f}",
                delta=f"{ruin_late - bpen_now:+.2f} vs flat",
            )
        st.caption(
            f"Flat (w_ruin_timing=0) ceza = {bpen_now:.1f}  |  "
            f"Formül: `{bpen_now:.1f} × (1 + {w_ruin:.2f} × (1 − step_frac))`"
        )

    st.divider()

    # ------------------------------------------------------------------ #
    # 7) Preset KAYDET / YUKLE
    # ------------------------------------------------------------------ #
    st.subheader("💾 Preset Kaydet / Yükle")

    save_col, load_col = st.columns([1, 1])

    with save_col:
        st.markdown("**Mevcut ayarları kaydet**")
        preset_name = st.text_input(
            "Preset adı",
            value="",
            key="rt_preset_name",
            placeholder="Örn: agresif_kazanc",
            help="Harf/rakam/_ ve - kullanılabilir; diğerleri _ ile değiştirilir.",
        )
        preset_desc = st.text_area(
            "Açıklama (isteğe bağlı)",
            value="",
            key="rt_preset_desc",
            placeholder="Bu preset ne için?",
            height=68,
        )
        if st.button("💾 Kaydet", key="rt_save_btn"):
            _name = preset_name.strip() or "preset"
            try:
                saved_path = save_reward_preset(cfg, _name, preset_desc)
                st.success(f"Kaydedildi: `{saved_path.split('/')[-1].split(chr(92))[-1]}`")
            except Exception as exc:
                st.error(f"Kaydetme hatası: {exc}")

    with load_col:
        st.markdown("**Kayıtlı preset yükle**")
        presets = list_reward_presets()
        if presets:
            def _preset_label(p: dict) -> str:
                desc_short = (p["description"][:30] + "…") if len(p["description"]) > 30 else p["description"]
                return f"{p['name']} · {p['saved_at']} · {desc_short}"

            preset_labels = [_preset_label(p) for p in presets]
            sel_label = st.selectbox(
                "Preset seç",
                options=preset_labels,
                key="rt_preset_selectbox",
                help="Kayıtlı preset'ler — isim · tarih · açıklama",
            )
            sel_idx = preset_labels.index(sel_label) if sel_label in preset_labels else 0
            sel_preset = presets[sel_idx]

            if st.button("📂 Yükle", key="rt_load_btn"):
                loaded = load_reward_preset(sel_preset["name"])
                if loaded is not None:
                    st.session_state.reward_cfg = loaded
                    st.success(f"Yüklendi: **{sel_preset['name']}**")
                    st.rerun()
                else:
                    st.error("Preset yüklenemedi — dosya silinmiş olabilir.")
        else:
            st.info("Henüz kayıtlı preset yok. Ayarları yapıp 'Kaydet' butonunu kullan.")

    st.divider()

    # "Preset'e don" reset butonu
    if st.button(
        "↺ Preset'e dön — tüm override'ları sıfırla",
        key="rt_reset_btn",
        help="reward_cfg'yi temizler; env'in kendi preset default'larına döner.",
    ):
        st.session_state.reward_cfg = {}
        st.rerun()

    st.caption(
        "Bu sekmedeki değerler `st.session_state.reward_cfg` dict'ine yazılır. "
        "Yeni ajan eğitildiğinde `build_env(reward_overrides=reward_cfg)` bunları env'e iletir. "
        "**CLI/golden testleri etkilenmez** — onlar config.py preset'ini kullanır."
    )
