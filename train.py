"""Main training + backtest driver — step_days + adaptive reward.

Trains DQN (discrete, 6 templates), PPO (continuous), SAC (continuous) ve TD3
(continuous, hocanin tavsiyesi) on BIST 28 — 2015-2021 train, 2022-2024 test
setinde backtest eder. Tüm ajanlar PyTorch'tadır ve
özellikler `utils.features.TrainScaler` ile train-only z-score standardize edilir.
"""
from __future__ import annotations

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from math import ceil
import numpy as np
import pandas as pd

from data import (download_bist, train_test_split, download_macro, align_macro,
                  resample_to_step_days)
from utils.features import add_features, TrainScaler
from utils.macro import add_macro_features, MacroScaler
from utils.metrics import summary, training_diagnostics
from utils.baselines import equal_weight, mean_variance, buy_and_hold_index
from config import DEFAULTS, SEED, TrainConfig, EnvConfig, ForecastConfig, MacroConfig, TD3Config  # noqa: F401
from core.contracts import DataProvenance, RunSpec
from core.factory import build_agent, build_env
from core.features import select_features
from core.persistence import save_agent, model_path
from core.rollout import evaluate as rollout_evaluate
from core.trainer import train as train_loop

BASE = Path(__file__).resolve().parent
RES  = BASE / "results"
RES.mkdir(exist_ok=True)
MODELS = BASE / "models"
MODELS.mkdir(exist_ok=True)

# -------------------- veri (P6: modul-global state yerine acik DataBundle) --------------------
@dataclass
class DataBundle:
    """Egitim/degerlendirme veri paketi — prepare_data() uretir, tuketiciler
    acikca alir (DIP/testability: global durum yok, sahte bundle enjekte edilebilir)."""
    px: pd.DataFrame
    px_tr: pd.DataFrame
    px_te: pd.DataFrame
    feats_tr: dict
    feats_te: dict
    scaler: TrainScaler
    # v6: makro rejim blogu (z-skorlu state) + ham regime (V7 odul amplify). None -> V5.
    macro_tr: "np.ndarray | None" = None
    macro_te: "np.ndarray | None" = None
    regime_tr: "np.ndarray | None" = None
    regime_te: "np.ndarray | None" = None
    step_days: int = DEFAULTS.step_days
    provenance: DataProvenance = DataProvenance()


def prepare_data(step_days: int = DEFAULTS.step_days) -> DataBundle:
    """BIST verisini yukler, train/test ayirir, train-only z-score uygular;
    DataBundle dondurur (onceki surum modul globallerini dolduruyordu)."""
    px_daily = download_bist()
    feats_all_raw = add_features(px_daily)
    px_tr_daily, px_te_daily = train_test_split(px_daily)
    if ForecastConfig.enabled:                     # v2: forecast feature (train-only fit)
        from forecast.forecaster import build_forecast_feature
        feats_all_raw["forecast"] = build_forecast_feature(
            px_daily, px_tr_daily, window=ForecastConfig.window, conv_ch=ForecastConfig.conv_ch,
            hidden=ForecastConfig.hidden, epochs=ForecastConfig.epochs,
            lr=ForecastConfig.lr, batch=ForecastConfig.batch, seed=SEED)
    px_tr = resample_to_step_days(px_tr_daily, step_days)
    px_te = resample_to_step_days(px_te_daily, step_days)
    px = pd.concat([px_tr, px_te])
    px.attrs.update(px_daily.attrs)
    feats_tr_raw = {k: resample_to_step_days(v.loc[px_tr_daily.index], step_days)
                    for k, v in feats_all_raw.items()}
    feats_te_raw = {k: resample_to_step_days(v.loc[px_te_daily.index], step_days)
                    for k, v in feats_all_raw.items()}
    scaler = TrainScaler().fit(feats_tr_raw)
    feats_tr = scaler.transform(feats_tr_raw)
    feats_te = scaler.transform(feats_te_raw)
    print(f"Train: {px_tr.shape}, Test: {px_te.shape}, tickers: {px.shape[1]}")

    # v6: makro rejim (faiz/dolar/altin) — train-only z-score (leak-safe), ham regime ayri.
    macro_tr = macro_te = regime_tr = regime_te = None
    if MacroConfig.enabled:
        mraw = align_macro(download_macro(), px_daily.index)
        mfeat = add_macro_features(mraw)                       # (T,4) ham
        regime_full = mfeat["regime"]                          # ham ∈[-1,1] -> V7 amplify
        mtr = resample_to_step_days(mfeat.loc[px_tr_daily.index], step_days)
        mte = resample_to_step_days(mfeat.loc[px_te_daily.index], step_days)
        msc = MacroScaler().fit(mtr)
        macro_tr = msc.transform(mtr).to_numpy(np.float32)
        macro_te = msc.transform(mte).to_numpy(np.float32)
        regime_tr = resample_to_step_days(
            regime_full.loc[px_tr_daily.index].to_frame(), step_days).iloc[:, 0].to_numpy(np.float32)
        regime_te = resample_to_step_days(
            regime_full.loc[px_te_daily.index].to_frame(), step_days).iloc[:, 0].to_numpy(np.float32)
        print(f"Makro: {mfeat.shape[1]} oznitelik (regime/slope/usd_try/gold_tl)")

    return DataBundle(px=px, px_tr=px_tr, px_te=px_te,
                      feats_tr=feats_tr, feats_te=feats_te, scaler=scaler,
                      macro_tr=macro_tr, macro_te=macro_te,
                      regime_tr=regime_tr, regime_te=regime_te,
                      step_days=int(step_days),
                      provenance=DataProvenance.from_value(px_daily.attrs.get("provenance")))


def _feats_for(feats: dict, algo: str) -> dict:
    """Shim — SOLID P2: tek dogruluk kaynagi core.features.select_features.
    (test_env bu adi cagirir; geriye-uyumluluk icin korunur.)"""
    return select_features(feats, algo)


def _ew_nav_train(bundle: DataBundle) -> np.ndarray:
    """Egitim seti icin esit-agirlikli benchmark NAV dizisi (C3: basari metriginde
    kullanilir; train fonksiyonlari bu diziyi ew_nav olarak train_loop'a gecer).
    Sifir boylukta guvenli (None yerine bos dizi degil)."""
    return equal_weight(bundle.px_tr)["nav"]


# -------------------- DQN training --------------------
def train_dqn(bundle: DataBundle, n_episodes: int = TrainConfig.dqn_episodes,
              adaptive: bool = True, step_days: int | None = None):
    step_days = bundle.step_days if step_days is None else int(step_days)
    train_max_steps = len(bundle.px_tr)
    env = build_env("DQN", bundle.px_tr, bundle.feats_tr, adaptive=adaptive,
                    max_steps=train_max_steps, random_start=EnvConfig.random_start, seed=SEED,
                    macro=bundle.macro_tr, regime=bundle.regime_tr, rebalance_freq=1,
                    step_days=step_days, gamma=DEFAULTS.gamma,
                    mom_window=DEFAULTS.mom_window, minvol_window=DEFAULTS.minvol_window)
    agent = build_agent("DQN", env.state_dim, env.n_discrete, seed=SEED)
    # C3: egitim seti EW benchmark'i — trainer success_vs_benchmark'e gecer.
    ew_nav_tr = _ew_nav_train(bundle)
    curve = []
    for rec in train_loop(agent, env, n_iters=n_episodes, ew_nav=ew_nav_tr):
        # C3: rec["success"] kernel'den gelir (success_vs_benchmark); override yok.
        curve.append(dict(episode=rec["episode"], reward=rec["reward"],
                          train_nav=rec["train_nav"], eps=rec["eps"],
                          gain=rec["gain"], success=rec["success"],
                          steps=len(rec.get("actions") or [])))
        print(f"[DQN] ep {rec['episode']:02d}  ret={rec['reward']:+.3f}  "
              f"NAV={rec['train_nav']:.3f}  eps={rec['eps']:.3f}")
    return agent, curve


# -------------------- PPO training --------------------
def train_ppo(bundle: DataBundle, n_updates: int = TrainConfig.ppo_updates,
              rollout_len: int = TrainConfig.ppo_rollout_len,
              adaptive: bool = True, step_days: int | None = None):
    # T7: PPO on-policy; rollout_len zaten episode'u belirler; max_steps > rollout_len olsun.
    # Vade bazlı train_max_steps ile uyumlu (büyük değer verirsek sorun yok ama
    # train_max_steps * 5 ile rollout'ların kesintisiz akmasına izin verelim).
    step_days = bundle.step_days if step_days is None else int(step_days)
    train_max_steps = len(bundle.px_tr)
    env = build_env("PPO", bundle.px_tr, bundle.feats_tr, adaptive=adaptive,
                    max_steps=train_max_steps, random_start=EnvConfig.random_start, seed=SEED,
                    macro=bundle.macro_tr, regime=bundle.regime_tr, rebalance_freq=1,
                    step_days=step_days, gamma=DEFAULTS.gamma,
                    mom_window=DEFAULTS.mom_window, minvol_window=DEFAULTS.minvol_window)
    agent = build_agent("PPO", env.state_dim, env.action_dim, seed=SEED)
    # C3: egitim seti EW benchmark'i.
    ew_nav_tr = _ew_nav_train(bundle)
    curve = []
    for rec in train_loop(agent, env, n_iters=n_updates, rollout_len=rollout_len,
                          ew_nav=ew_nav_tr):
        # C3: rec["success"] kernel'den gelir; override yok.
        curve.append(dict(update=rec["update"], p_loss=rec["p_loss"], v_loss=rec["v_loss"],
                          ent=rec["ent"], kl=rec["kl"], mean_nav=rec["mean_nav"],
                          reward=rec["reward"], gain=rec["gain"],
                          success=rec["success"], steps=rollout_len))
        print(f"[PPO] upd {rec['update']:02d}  p_loss={rec['p_loss']:.3f} "
              f"v_loss={rec['v_loss']:.3f} ent={rec['ent']:.2f} kl={rec['kl']:.3f}")
    return agent, curve


# -------------------- SAC training --------------------
def train_sac(bundle: DataBundle, n_episodes: int = TrainConfig.sac_episodes,
              adaptive: bool = True, step_days: int | None = None):
    step_days = bundle.step_days if step_days is None else int(step_days)
    env = build_env("SAC", bundle.px_tr, bundle.feats_tr, adaptive=adaptive,
                    max_steps=len(bundle.px_tr),
                    random_start=EnvConfig.random_start, seed=SEED,
                    macro=bundle.macro_tr, regime=bundle.regime_tr, rebalance_freq=1,
                    step_days=step_days, gamma=DEFAULTS.gamma,
                    mom_window=DEFAULTS.mom_window, minvol_window=DEFAULTS.minvol_window)
    agent = build_agent("SAC", env.state_dim, env.action_dim, seed=SEED)
    # C3: egitim seti EW benchmark'i.
    ew_nav_tr = _ew_nav_train(bundle)
    curve = []
    for rec in train_loop(agent, env, n_iters=n_episodes, ew_nav=ew_nav_tr):
        # C3: rec["success"] kernel'den gelir; override yok.
        curve.append(dict(episode=rec["episode"], train_nav=rec["train_nav"], steps=rec["steps"],
                          reward=rec["reward"], gain=rec["gain"],
                          success=rec["success"]))
        print(f"[SAC] ep {rec['episode']:02d}  NAV={rec['train_nav']:.3f}  buf={len(agent.buffer)}")
    return agent, curve


# -------------------- TD3 training --------------------
def train_td3(bundle: DataBundle, n_episodes: int = TrainConfig.td3_episodes,
              adaptive: bool = True, step_days: int | None = None):
    # TD3 surekli-kontrol (hocanin tavsiyesi) — SAC ile ayni off-policy rejim.
    # Her episode secilen train tarih araliginin tamamini kullanir.
    step_days = bundle.step_days if step_days is None else int(step_days)
    env = build_env("TD3", bundle.px_tr, bundle.feats_tr, adaptive=adaptive,
                    max_steps=len(bundle.px_tr),
                    random_start=EnvConfig.random_start, seed=SEED,
                    macro=bundle.macro_tr, regime=bundle.regime_tr, rebalance_freq=1,
                    step_days=step_days, gamma=DEFAULTS.gamma,
                    mom_window=DEFAULTS.mom_window, minvol_window=DEFAULTS.minvol_window)
    agent = build_agent("TD3", env.state_dim, env.action_dim, seed=SEED)
    # C3: egitim seti EW benchmark'i.
    ew_nav_tr = _ew_nav_train(bundle)
    curve = []
    for rec in train_loop(agent, env, n_iters=n_episodes, ew_nav=ew_nav_tr):
        # C3: rec["success"] kernel'den gelir; override yok.
        curve.append({"episode": rec["episode"], "train_nav": rec["train_nav"], "steps": rec["steps"],
                      "reward": rec["reward"], "gain": rec["gain"],
                      "success": rec["success"]})
        print(f"[TD3] ep {rec['episode']:02d}  NAV={rec['train_nav']:.3f}  buf={len(agent.buffer)}")
    return agent, curve


# -------------------- Evaluation --------------------
def evaluate(bundle: DataBundle, agent, algo: str, adaptive: bool = True,
             step_days: int | None = None):
    # T7: eval env TAM test donemini kosturur — max_steps = len(test) (backtest
    # tum test penceresini kapsar, egitim episode uzunlugundan bagimsiz).
    step_days = bundle.step_days if step_days is None else int(step_days)
    context = min(len(bundle.px_tr), max(21, ceil(DEFAULTS.minvol_window / step_days)) + 1)
    px_context = bundle.px_tr.iloc[-context:]
    px_eval = pd.concat([px_context, bundle.px_te])
    px_eval.attrs.update(bundle.px_te.attrs)
    px_eval.attrs["session_counts"] = np.concatenate([
        np.asarray(bundle.px_tr.attrs.get(
            "session_counts", np.ones(len(bundle.px_tr), dtype=int)))[-context:],
        np.asarray(bundle.px_te.attrs.get("session_counts", np.ones(len(bundle.px_te), dtype=int))),
    ])
    feats_eval = {k: pd.concat([bundle.feats_tr[k].iloc[-context:], bundle.feats_te[k]])
                  for k in bundle.feats_te}
    macro_eval = (None if bundle.macro_te is None else
                  np.concatenate([bundle.macro_tr[-context:], bundle.macro_te], axis=0))
    regime_eval = (None if bundle.regime_te is None else
                   np.concatenate([bundle.regime_tr[-context:], bundle.regime_te], axis=0))
    eval_max_steps = len(bundle.px_te) + 10
    env = build_env(algo, px_eval, feats_eval, adaptive=adaptive,
                    max_steps=eval_max_steps, macro=macro_eval, regime=regime_eval,
                    rebalance_freq=1, step_days=step_days, gamma=DEFAULTS.gamma,
                    mom_window=DEFAULTS.mom_window, minvol_window=DEFAULTS.minvol_window,
                    start_index=context)
    return rollout_evaluate(agent, env)


# -------------------- Main --------------------
def run(step_days: int = DEFAULTS.step_days):
    """Tam egitim + backtest akisi: seed -> veri -> 3 ajan -> eval -> CSV.
    main.py bunu DOGRUDAN cagirir (runpy yerine). Modul import'u yan etkisizdir (M1)."""
    np.random.seed(SEED)
    bundle = prepare_data(step_days=step_days)
    t0 = time.time()
    print("=" * 60)
    dqn_agent, dqn_curve = train_dqn(bundle)
    print("DQN total time:", round(time.time() - t0, 1), "s")

    t1 = time.time()
    ppo_agent, ppo_curve = train_ppo(bundle)
    print("PPO total time:", round(time.time() - t1, 1), "s")

    t2 = time.time()
    sac_agent, sac_curve = train_sac(bundle)
    print("SAC total time:", round(time.time() - t2, 1), "s")

    t3 = time.time()
    td3_agent, td3_curve = train_td3(bundle)          # hocanin tavsiyesi — SAC'tan SONRA
    print("TD3 total time:", round(time.time() - t3, 1), "s")

    print("=" * 60)

    # ---- T2 (C1): once ham eval; sonra ORTAK min_len ile hizala; metrics.csv
    # hizali dizilerden uretilir -> RL + baseline ayni gun sayisinda kiyaslanir. ----
    raw_results = {}
    for name, agent in [("DQN", dqn_agent), ("PPO", ppo_agent),
                        ("SAC", sac_agent), ("TD3", td3_agent)]:
        bt = evaluate(bundle, agent, name)
        raw_results[name] = bt

    bh = buy_and_hold_index(bundle.px_te)
    ew = equal_weight(bundle.px_te)
    mv = mean_variance(bundle.px_te, lookback=120, rebalance=20)
    for name, d in [("BuyHold", bh), ("EqualWeight", ew), ("MeanVar", mv)]:
        raw_results[name] = d

    date_indexes = [pd.DatetimeIndex(v.get("dates", bundle.px_te.index[:len(v["nav"])]))
                    for v in raw_results.values()]
    common_dates = date_indexes[0]
    for idx in date_indexes[1:]:
        common_dates = common_dates.intersection(idx)
    common_dates = common_dates.sort_values()
    if len(common_dates) < 2:
        raise RuntimeError("Stratejiler arasinda yeterli ortak degerleme tarihi yok")

    results = {}
    for name, bt in raw_results.items():
        # Hizalama: son min_len elemanı al (ortak takvim kuyruğu) VE ortak pencere
        # başlangıcına YENİDEN-TABANLA (NAV[0]=1) -> FinalNAV/CAGR tüm stratejiler
        # için AYNI pencere büyümesini ölçer; baseline'lar RL'in görmediği ilk
        # ~window günü dahil etmez (C1 adil karşılaştırma tam olarak sağlanır).
        idx = pd.DatetimeIndex(bt.get("dates", bundle.px_te.index[:len(bt["nav"])]))
        nav_series = pd.Series(np.asarray(bt["nav"], dtype=float), index=idx)
        nav_aligned = nav_series.reindex(common_dates).to_numpy()
        rets_aligned = np.concatenate([[0.0], np.diff(nav_aligned) / nav_aligned[:-1]])
        w_aligned    = bt.get("weights")
        if w_aligned is not None:
            w_aligned = pd.DataFrame(np.asarray(w_aligned), index=idx).reindex(common_dates).to_numpy()
        turn_aligned = None
        if bt.get("turnover") is not None:
            turn_aligned = pd.Series(np.asarray(bt["turnover"], dtype=float), index=idx).reindex(common_dates).to_numpy()
        # Hizali diziler uzerinden metrik hesapla (C1 duzeltme).
        m = summary(nav_aligned, rets_aligned, w_aligned, dates=common_dates,
                    turnover_values=turn_aligned)
        results[name] = dict(backtest=bt, nav_aligned=nav_aligned, metrics=m)
        if name in ("DQN", "PPO", "SAC", "TD3"):
            print(f"[TEST] {name:<3}  CAGR={m['CAGR']:+.2%}  Sharpe={m['Sharpe']:+.2f}  "
                  f"MaxDD={m['MaxDD']:+.2%}  Final={m['FinalNAV']:.3f}")

    print("-" * 60)
    for name in ("BuyHold", "EqualWeight", "MeanVar"):
        m = results[name]["metrics"]
        print(f"[TEST] {name:<12}  CAGR={m['CAGR']:+.2%}  Sharpe={m['Sharpe']:+.2f}  "
              f"MaxDD={m['MaxDD']:+.2%}  Final={m['FinalNAV']:.3f}")

    # navs_aligned.csv: hizali NAV dizileri (T2: metrics.csv ile AYNI pencere).
    dfn = pd.DataFrame({k: v["nav_aligned"] for k, v in results.items()})
    dfn.index = common_dates
    dfn.to_csv(RES / "navs_aligned.csv")

    # metrics.csv: hizali dizilerden hesaplanan metrikler (C1 duzeltme).
    met_df = pd.DataFrame({k: v["metrics"] for k, v in results.items()}).T
    met_df.to_csv(RES / "metrics.csv")
    print(met_df.round(4))

    pd.DataFrame(dqn_curve).to_csv(RES / "dqn_curve.csv", index=False)
    pd.DataFrame(ppo_curve).to_csv(RES / "ppo_curve.csv", index=False)
    pd.DataFrame(sac_curve).to_csv(RES / "sac_curve.csv", index=False)
    pd.DataFrame(td3_curve).to_csv(RES / "td3_curve.csv", index=False)

    # PDF §9.7 toplu egitim teshisleri (gozlemsel; golden metriklerini etkilemez).
    curves = {"DQN": dqn_curve, "PPO": ppo_curve, "SAC": sac_curve, "TD3": td3_curve}
    diag = {n: training_diagnostics(curves[n], results[n]["backtest"].get("reward_terms_history"))
            for n in ("DQN", "PPO", "SAC", "TD3")}
    pd.DataFrame(diag).T.to_csv(RES / "training_diagnostics.csv")
    print(pd.DataFrame(diag).T.round(4))

    # PDF §11: egitilmis modelleri diske kaydet (sunumda yeniden egitmeden test).
    config_by_algo = {"DQN": asdict(__import__("config").DQNConfig()),
                      "PPO": asdict(__import__("config").PPOConfig()),
                      "SAC": asdict(__import__("config").SACConfig()),
                      "TD3": asdict(__import__("config").TD3Config())}
    for name, agent in [("DQN", dqn_agent), ("PPO", ppo_agent),
                        ("SAC", sac_agent), ("TD3", td3_agent)]:
        spec = RunSpec(name, step_days=bundle.step_days, adaptive=True,
                       agent_hp=config_by_algo[name], feature_names=tuple(bundle.feats_tr),
                       provenance=bundle.provenance)
        save_agent(agent, name, model_path(name, bundle.step_days, True),
                   horizon=f"step{bundle.step_days}", adaptive=True, run_spec=spec)
    print("Modeller kaydedildi:", MODELS)

    for name in ["DQN", "PPO", "SAC", "TD3"]:
        W = results[name]["backtest"]["weights"]
        cols = list(bundle.px_te.columns) + ["CASH"]
        pd.DataFrame(W, columns=cols).to_csv(RES / f"weights_{name}.csv", index=False)
        bt = results[name]["backtest"]
        trace = pd.DataFrame({
            "date": bt["dates"][1:],
            "gross_return": [x["gross_port_r"] for x in bt["reward_terms_history"]],
            "net_return": bt["rets"][1:], "turnover": bt["turnover"][1:],
            "period_length": bt["period_lengths"][1:],
        })
        trace.to_csv(RES / f"backtest_{name}.csv", index=False)

    (RES / "run_manifest.json").write_text(json.dumps({
        "step_days": bundle.step_days, "seed": SEED,
        "provenance": bundle.provenance.to_dict(),
    }, ensure_ascii=True, indent=2), encoding="utf-8")

    print("=" * 60)
    print("DONE. Total wall time:", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    run()
