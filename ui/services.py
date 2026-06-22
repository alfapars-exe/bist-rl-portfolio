"""UI veri/egitim/test servisleri (app.py'den tasindi, P5).

Streamlit session_state'i ile core katmani arasindaki kopru: veri yukleme,
env/ajan kurulumu (core.factory'ye delege), canli egitim generator'i ve test
trajectory yakalama.
"""
from __future__ import annotations

from pathlib import Path
from math import ceil

import numpy as np
import pandas as pd
import streamlit as st

from agents.base import SupportsQValues
from config import DEFAULTS, SEED, ForecastConfig, MacroConfig, validate_train_range
from core.factory import build_agent, build_env
from core.episodes import episode_metrics
from core.persistence import (load_agent, model_path, named_model_path,
                              read_meta, save_agent, MODELS_DIR as _MODELS_DIR)
from core.trainer import train as train_loop
from data import (align_macro, download_bist, download_macro,
                  resample_to_step_days, train_test_split)
from env.portfolio_env import ACTION_NAMES
from ui.state import _agent_key
from utils.baselines import equal_weight
from utils.features import TrainScaler, add_features
from utils.macro import MacroScaler, add_macro_features
from utils.portfolio_tl import compute_tl_step

# PDF §11: egitilmis modeller diske burada kaydedilir/yuklenir (sunum kaliciligi).
# N11: MODELS_DIR ve model_path artik core.persistence'da tanimlidi; _MODELS_DIR olarak yukarda import edildi.
MODELS_DIR = _MODELS_DIR  # noqa: N816  — dis erisim icin re-export (sidebar import eder)


def save_trained_agent(algo: str, step_days: int, adaptive: bool, name: str = ""):
    """Session'daki egitilmis ajani diske kaydeder; yolu doner (yoksa None).

    N11: isim verilmisse named_model_path(name).pt kullanilir (UI isimli kayit).
         isim bossa fallback: {algo}_{horizon} ismiyle named_model_path.
         Geriye-uyumluluk: model_path(algo,horizon,adaptive) CLI/golden yolu KORUNUR —
         bu fonksiyon yalnizca UI "Kaydet" butonundan cagirilir.
    name: kullanici-girilen model adi; bos olursa "{algo}_{horizon}" kullanilir.
    """
    entry = st.session_state.trained_agents.get(_agent_key(algo, step_days, adaptive))
    if not entry or entry[0] is None:
        return None
    effective_name = name.strip() if name.strip() else f"{algo}_step{step_days}"
    path = named_model_path(effective_name)
    return save_agent(entry[0], algo, path,
                      horizon=f"step{step_days}", adaptive=adaptive,
                      name=effective_name, run_spec=st.session_state.get("active_run_spec"))


def list_saved_models() -> list[dict]:
    """models/*.pt dosyalarini tarar; her biri icin read_meta ile meta okur.

    Donus: [{path, name, saved_at, algo, horizon, adaptive}, ...]
    saved_at'e gore yeniden-eskiye sirali. Bozuk/okunamayan dosyalar atlanmaz;
    meta bos string'lerle doldurulur (read_meta guvenli default doner).
    """
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for pt in _MODELS_DIR.glob("*.pt"):
        meta = read_meta(pt)
        # Eger name meta'da bossa dosya adini goster (eski format geriye-uyumlu)
        display_name = meta["name"] if meta["name"] else pt.stem
        results.append({
            "path": str(pt),
            "name": display_name,
            "saved_at": meta["saved_at"],
            "algo": meta["algo"],
            "horizon": meta["horizon"],
            "adaptive": meta["adaptive"],
            "step_days": int(meta.get("run_spec", {}).get("step_days", 1)),
            "legacy": bool(meta.get("legacy", True)),
        })
    # saved_at'e gore yeniden→eskiye sirala (ISO string karsilastirmasi dogru calisir)
    results.sort(key=lambda x: x["saved_at"], reverse=True)
    return results


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
    if meta.get("run_spec"):
        st.session_state.active_run_spec = meta["run_spec"]
        step_days = int(meta["run_spec"].get("step_days", 1))
        key = _agent_key(meta["algo"], step_days, meta["adaptive"])
    else:
        key = _agent_key(meta["algo"], meta["horizon"], meta["adaptive"])
    st.session_state.trained_agents[key] = (agent, [])
    return key, meta


def _load_data():
    """Veri indir + granülerliğe resample + z-score scaler'ı fit et.

    Tarih aralığı session_state.data_start / data_split / data_end'den okunur
    (sidebar tarih seçici). Scaler/forecaster/MacroScaler YALNIZ px_tr'de fit
    edilir — sızıntı yok.

    Granülerlik akışı (golden-güvenli Approach 1):
      1. prices_daily  = download_bist(...)           — her zaman GÜNLÜK
      2. feats_daily   = add_features(prices_daily)   — GÜNLÜK (add_features DEĞİŞMEZ)
      3. Forecast da prices_daily üzerinde hesaplanır (GÜNLÜK)
      4. prices / feats_all_raw = resample_to_granularity(...)  — g="daily" → no-op
      5. train_test_split RESAMPLE'LANMIŞ fiyat üzerinde yapılır
      6. Scaler RESAMPLE'LANMIŞ feats_tr'de fit edilir (sızıntı korunur)
      7. Makro: download_macro GÜNLÜK → align (GÜNLÜK prices_daily.index) →
         add_macro_features (GÜNLÜK) → resample → MacroScaler resample'lı px_tr'de fit
    """
    from config import DataConfig as _DC
    _dc_defaults = _DC()
    data_start = st.session_state.get("data_start", _dc_defaults.start)
    data_split = st.session_state.get("data_split", _dc_defaults.train_end)
    data_end   = st.session_state.get("data_end",   _dc_defaults.end)
    step_days = int(st.session_state.get("step_days", DEFAULTS.step_days))

    with st.spinner("Veri indiriliyor / cache okunuyor ..."):
        prices_daily = download_bist(start=data_start, end=data_end)
    # Sentetik-veri görünürlüğü: ağ + cache yoksa download_bist SENTETİK GBM'e düşer
    # (px.attrs["synthetic"]=True). Bu durumda sonuçlar GERÇEK DEĞİLDİR ve NaN/anlamsız
    # değerler çıkabilir → kullanıcıya AÇIK uyarı (sessiz NaN yerine).
    source = prices_daily.attrs.get("provenance", {}).get("source", "unknown")
    if source != "real":
        st.warning(
            f"Veri kaynagi {source.upper()}. Egitim ve test devam eder; modeller, "
            "CSV'ler ve ekran bu provenance etiketini korur."
        )

    # --- 1. Feature'lar her zaman GÜNLÜK hesaplanır (add_features DEĞİŞMEZ) ---
    feats_daily = add_features(prices_daily)

    # --- 2. Forecast feature GÜNLÜK prices_daily üzerinde fit edilir ---
    if ForecastConfig.enabled:                     # v2: forecast feature (train-only fit)
        from forecast.forecaster import build_forecast_feature
        # Forecast fit için geçici günlük train split kullanılır (sızıntısız)
        _px_tr_daily, _ = train_test_split(prices_daily, split=data_split)
        feats_daily["forecast"] = build_forecast_feature(
            prices_daily, _px_tr_daily,
            window=ForecastConfig.window, conv_ch=ForecastConfig.conv_ch,
            hidden=ForecastConfig.hidden, epochs=ForecastConfig.epochs,
            lr=ForecastConfig.lr, batch=ForecastConfig.batch, seed=SEED)

    # Split daily data first, then independently select N-session endpoints. A
    # period can never mix train and test observations.
    px_tr_daily, px_te_daily = train_test_split(prices_daily, split=data_split)
    px_tr = resample_to_step_days(px_tr_daily, step_days)
    px_te = resample_to_step_days(px_te_daily, step_days)
    prices = pd.concat([px_tr, px_te])
    prices.attrs.update(prices_daily.attrs)
    feats_tr_raw = {
        k: resample_to_step_days(v.loc[px_tr_daily.index], step_days) for k, v in feats_daily.items()
    }
    feats_te_raw = {
        k: resample_to_step_days(v.loc[px_te_daily.index], step_days) for k, v in feats_daily.items()
    }
    ok, message = validate_train_range(len(px_tr), step_days=step_days)
    if not ok:
        raise ValueError(message)

    # SIZINTI KORUMASI: scaler YALNIZ eğitim kısmında fit edilir, test'e transform uygulanır.
    scaler = TrainScaler().fit(feats_tr_raw)
    st.session_state.prices = prices
    st.session_state.px_tr = px_tr
    st.session_state.px_te = px_te
    st.session_state.feats_tr = scaler.transform(feats_tr_raw)
    st.session_state.feats_te = scaler.transform(feats_te_raw)
    st.session_state.scaler = scaler

    # Nokta sayısını kaydet (uyarı için sidebar/tab kullanabilir)
    st.session_state.granularity_n_points = len(prices)
    st.session_state.data_provenance = prices_daily.attrs.get("provenance", {})

    # --- 6. Makro: GÜNLÜK indir → GÜNLÜK align → add_macro_features (GÜNLÜK) → resample ---
    # MacroScaler RESAMPLE'LANMIŞ px_tr'de fit edilir (sızıntı korunur).
    macro_tr = macro_te = regime_tr = regime_te = None
    if MacroConfig.enabled:
        mfeat_daily = add_macro_features(
            align_macro(download_macro(start=data_start, end=data_end), prices_daily.index)
        )
        # Makro + regime'i granülerliğe resample et
        regime_daily = mfeat_daily["regime"].to_frame("regime")
        mtr_daily, mte_daily = train_test_split(mfeat_daily, split=data_split)
        rtr_daily, rte_daily = train_test_split(regime_daily, split=data_split)
        mtr = resample_to_step_days(mtr_daily, step_days)
        mte = resample_to_step_days(mte_daily, step_days)
        scaler_m = MacroScaler().fit(mtr)
        macro_tr = scaler_m.transform(mtr).to_numpy(np.float32)
        macro_te = scaler_m.transform(mte).to_numpy(np.float32)
        regime_tr = resample_to_step_days(rtr_daily, step_days)["regime"].to_numpy(np.float32)
        regime_te = resample_to_step_days(rte_daily, step_days)["regime"].to_numpy(np.float32)
    st.session_state.macro_tr = macro_tr
    st.session_state.macro_te = macro_te
    st.session_state.regime_tr = regime_tr
    st.session_state.regime_te = regime_te
    st.session_state.data_loaded = True


def _make_env(is_train: bool, algo: str, step_days: int, adaptive: bool, max_steps: int,
              cash_daily_rate: float | None = None, *,
              force_noise: bool = False, noise_eval: float = 0.0,
              seed_override: int | None = None):
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
    start_index = None
    if not is_train:
        context = min(len(st.session_state.px_tr), max(21, ceil(DEFAULTS.minvol_window / step_days)) + 1)
        px_context = st.session_state.px_tr.iloc[-context:]
        px_df = pd.concat([px_context, px_df])
        counts = np.concatenate([
            np.asarray(st.session_state.px_tr.attrs.get(
                "session_counts", np.ones(len(st.session_state.px_tr), dtype=int)))[-context:],
            np.asarray(st.session_state.px_te.attrs.get(
                "session_counts", np.ones(len(st.session_state.px_te), dtype=int))),
        ])
        px_df.attrs.update(st.session_state.px_te.attrs)
        px_df.attrs["session_counts"] = counts
        feats = {k: pd.concat([st.session_state.feats_tr[k].iloc[-context:], v])
                 for k, v in st.session_state.feats_te.items()}
        if macro is not None:
            macro = np.concatenate([st.session_state.macro_tr[-context:], macro], axis=0)
        if regime is not None:
            regime = np.concatenate([st.session_state.regime_tr[-context:], regime], axis=0)
        start_index = context
    # Eğitimde UI'dan okunan σ geçilir; eval'de None → env gürültüyü zaten
    # random_start=False ile kapatır, ama yine de None göndererek kasıtsız gürültüyü engelle.
    noise_std = (st.session_state.get("price_noise_std") if is_train else None)
    # Episode-clean (kullanici istegi: 1. iterasyon ORIJINAL veri, 2+ farkli noise'lu).
    # Yalniz egitimde + UI toggle (default True) acikken. Eval'de noise zaten kapali -> etkisiz.
    ep_clean = bool(is_train and st.session_state.get("episode_clean", True))
    # Rebalans frekansı override (sidebar): None -> preset (golden-güvenli). Train+eval'e
    # AYNI değer uygulanır (model hangi frekansla eğitildiyse onunla test edilsin).
    # N12: nakit faiz — önce parametre, sonra session_state, sonra env default (None).
    if cash_daily_rate is None:
        cash_daily_rate = st.session_state.get("cash_daily_rate", None)
    return build_env(
        algo, px_df, feats, adaptive=adaptive, max_steps=max_steps,
        random_start=is_train,                     # v2: egitimde rastgele pencere, eval'de sabit
        seed=(int(seed_override) if seed_override is not None else SEED),  # noise-episode: per-episode tohum
        reward_overrides=st.session_state.get("reward_cfg", {}) or {},
        price_noise_std=(float(noise_eval) if force_noise else noise_std),  # UI σ (train) / noise-episode (eval)
        force_price_noise=bool(force_noise),       # gurultu-artirimli coklu-episode: eval'de gurultu ac
        episode_clean=ep_clean,                    # 1. iterasyon orijinal (anti-ezber)
        macro=macro, regime=regime,                # v6: makro rejim blogu + ham regime
        cash_daily_rate=cash_daily_rate,           # N12: UI nakit faiz oranı
        rebalance_freq=1, step_days=int(step_days),
        gamma=float(st.session_state.get("gamma_daily", DEFAULTS.gamma)),
        mom_window=DEFAULTS.mom_window, minvol_window=DEFAULTS.minvol_window,
        start_index=start_index,
    )


def evaluate_noise_episodes_ui(agent, algo: str, step_days: int, adaptive: bool, *,
                               n_episodes: int = 5, noise_std: float = 0.01) -> list:
    """Egitilmis ajani test araligi boyunca SIRAYLA birden cok episode'da kosturur;
    HER episode icin tam adim-adim trace toplar (Test sekmesindeki detay tablosu icin).

    Episode 0 = orijinal (gurultusuz) referans — normal testle AYNI kurulum (_make_env
    eval yolu: context penceresi + start_index) -> episode 0 normal test sonucuyla
    ortusur. Episode 1..N = ayni test serisine getiri-seviyesinde Gauss gurultusu
    (force_price_noise; episode basina FARKLI tohum) eklenmis YENI patikalar (anti-ezber).
    Her episode TUM veri tarih araligini kapsar ve SIRAYLA kosar — biri tam BITMEDEN
    (N gun varsa N adim) digeri BASLAMAZ. Donen her episode dict'i: episode, noise_std,
    steps, nav, dates, trace (adim-adim), + final_nav/total_return/max_drawdown/sharpe.
    """
    results = []
    for i in range(int(n_episodes)):
        nstd = 0.0 if i == 0 else float(noise_std)
        env = _make_env(False, algo, step_days, adaptive, max_steps=10_000,
                        force_noise=(nstd > 0.0), noise_eval=nstd, seed_override=SEED + i)
        # Tam aralik, adim-adim trace (tekli-test ile AYNI dongey reuse eder) — episode
        # done'a (veri sonu) kadar kosar; sonraki episode ancak bu bittikten sonra baslar.
        trace = _run_trace_loop(env, agent, algo, light=True)
        nav = np.asarray(env.nav_history, dtype=float)
        rets = np.asarray(getattr(env, "ret_history", []), dtype=float)
        results.append(dict(
            episode=i, noise_std=nstd, steps=len(trace),
            nav=nav, dates=[t["date"] for t in trace], trace=trace,
            **episode_metrics(nav, rets),
        ))
    return results


def _make_agent(algo: str, state_dim: int, action_dim: int, hp: dict):
    """Shim — SOLID P3: tek dogruluk kaynagi core.factory.build_agent.
    (test_config_wiring app._make_agent'i cagirir; app.py bunu re-export eder.)"""
    return build_agent(algo, state_dim, action_dim, hp)


# =====================================================================
# Eğitim jeneratörü — canlı UI için episod başına yield
# =====================================================================
def train_generator(algo: str, step_days: int, adaptive: bool, hp: dict,
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
    max_steps = max(1, len(st.session_state.px_tr))
    env = _make_env(True, algo, step_days, adaptive, max_steps=max_steps)
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
def evaluate_with_trace(agent, algo: str, step_days: int, adaptive: bool) -> list:
    env = _make_env(False, algo, step_days, adaptive, max_steps=10_000)
    return _run_trace_loop(env, agent, algo)


def _run_trace_loop(env, agent, algo: str, light: bool = False) -> list:
    """Bir env'i adim-adim kosturup TAM trace dondurur (ORTAK cekirdek: tekli test
    playback'i + gurultu-artirimli per-episode tablo ayni dongey reuse eder). Episode
    veri sonuna (done) kadar SIRAYLA kosar. light=True -> 'state'/'q_values' atlanir
    (cok-episode bellek; detay tablosu icin gereksiz). light=False -> tam trace."""
    s, _ = env.reset()
    trace = []
    done = trunc = False
    # P5 (ISP): hasattr yoklamasi yerine resmi Protocol — ayni semantik, acik niyet.
    has_q = (not light) and isinstance(agent, SupportsQValues)  # yalnizca DQN introspeksiyonu
    while not (done or trunc):
        decision_date = env.dates[env.t]
        weights_before = env.w.copy()
        state_snapshot = None if light else s.copy()

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
            "date": str(pd.Timestamp(info["date"]).date()),
            "decision_date": str(pd.Timestamp(decision_date).date()),
            "state": state_snapshot,
            "action_idx": action_idx,
            "action_name": action_name,
            "q_values": q_vals,
            "weights_before": weights_before,
            "target_weights": info["target_weights"].copy(),
            "weights_after": env.w.copy(),
            "reward_terms": info["reward_terms"],
            "period_length": info["period_length"],
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
            target_weights=t.get("target_weights"),
            holding_period_days=int(t.get("period_length", 1)),
        )
        snap["w_now"] = t["weights_after"]
        snaps.append(snap)
        holding_days_prev = snap["holding_days"]
        prev_portfolio_tl = snap["portfolio_tl"]
    return snaps
