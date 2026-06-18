"""Main training + backtest driver — yeni paket yapısı + horizon + adaptive reward.

Trains DQN (discrete, 6 templates), PPO (continuous), SAC (continuous) ve TD3
(continuous, hocanin tavsiyesi) on BIST 28 — 2015-2021 train, 2022-2024 test
setinde backtest eder. Tüm ajanlar PyTorch'tadır ve
özellikler `utils.features.TrainScaler` ile train-only z-score standardize edilir.
"""
from __future__ import annotations

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from data import download_bist, train_test_split, download_macro, align_macro
from utils.features import add_features, TrainScaler
from utils.macro import add_macro_features, MacroScaler
from utils.metrics import summary, training_diagnostics
from utils.baselines import equal_weight, mean_variance, buy_and_hold_index
from config import SEED, TrainConfig, EnvConfig, ForecastConfig, MacroConfig, TD3Config  # noqa: F401
from core.factory import build_agent, build_env
from core.features import select_features
from core.persistence import save_agent
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


def prepare_data() -> DataBundle:
    """BIST verisini yukler, train/test ayirir, train-only z-score uygular;
    DataBundle dondurur (onceki surum modul globallerini dolduruyordu)."""
    px = download_bist()
    feats_all_raw = add_features(px)
    px_tr, px_te = train_test_split(px)
    if ForecastConfig.enabled:                     # v2: forecast feature (train-only fit)
        from forecast.forecaster import build_forecast_feature
        feats_all_raw["forecast"] = build_forecast_feature(
            px, px_tr, window=ForecastConfig.window, conv_ch=ForecastConfig.conv_ch,
            hidden=ForecastConfig.hidden, epochs=ForecastConfig.epochs,
            lr=ForecastConfig.lr, batch=ForecastConfig.batch, seed=SEED)
    feats_tr_raw = {k: v.loc[px_tr.index] for k, v in feats_all_raw.items()}
    feats_te_raw = {k: v.loc[px_te.index] for k, v in feats_all_raw.items()}
    scaler = TrainScaler().fit(feats_tr_raw)
    feats_tr = scaler.transform(feats_tr_raw)
    feats_te = scaler.transform(feats_te_raw)
    print(f"Train: {px_tr.shape}, Test: {px_te.shape}, tickers: {px.shape[1]}")

    # v6: makro rejim (faiz/dolar/altin) — train-only z-score (leak-safe), ham regime ayri.
    macro_tr = macro_te = regime_tr = regime_te = None
    if MacroConfig.enabled:
        mraw = align_macro(download_macro(), px.index)
        mfeat = add_macro_features(mraw)                       # (T,4) ham
        regime_full = mfeat["regime"]                          # ham ∈[-1,1] -> V7 amplify
        msc = MacroScaler().fit(mfeat.loc[px_tr.index])        # YALNIZ train (sizintisiz)
        macro_z = msc.transform(mfeat)                         # (T,4) z-skorlu -> state
        macro_tr = macro_z.loc[px_tr.index].to_numpy(np.float32)
        macro_te = macro_z.loc[px_te.index].to_numpy(np.float32)
        regime_tr = regime_full.loc[px_tr.index].to_numpy(np.float32)
        regime_te = regime_full.loc[px_te.index].to_numpy(np.float32)
        print(f"Makro: {macro_z.shape[1]} oznitelik (regime/slope/usd_try/gold_tl)")

    return DataBundle(px=px, px_tr=px_tr, px_te=px_te,
                      feats_tr=feats_tr, feats_te=feats_te, scaler=scaler,
                      macro_tr=macro_tr, macro_te=macro_te,
                      regime_tr=regime_tr, regime_te=regime_te)


def _feats_for(feats: dict, algo: str) -> dict:
    """Shim — SOLID P2: tek dogruluk kaynagi core.features.select_features.
    (test_env bu adi cagirir; geriye-uyumluluk icin korunur.)"""
    return select_features(feats, algo)


# -------------------- DQN training --------------------
def train_dqn(bundle: DataBundle, n_episodes: int = TrainConfig.dqn_episodes,
              horizon: str = "medium", adaptive: bool = True):
    env = build_env("DQN", bundle.px_tr, bundle.feats_tr, horizon=horizon, adaptive=adaptive,
                    max_steps=252, random_start=EnvConfig.random_start, seed=SEED,
                    macro=bundle.macro_tr, regime=bundle.regime_tr)
    agent = build_agent("DQN", env.state_dim, env.n_discrete, seed=SEED)
    curve = []
    for rec in train_loop(agent, env, n_iters=n_episodes):
        curve.append(dict(episode=rec["episode"], reward=rec["reward"],
                          train_nav=rec["train_nav"], eps=rec["eps"],
                          gain=rec["gain"], success=int(rec["nav"] > 1.0),
                          steps=len(rec.get("actions") or [])))
        print(f"[DQN] ep {rec['episode']:02d}  ret={rec['reward']:+.3f}  "
              f"NAV={rec['train_nav']:.3f}  eps={rec['eps']:.3f}")
    return agent, curve


# -------------------- PPO training --------------------
def train_ppo(bundle: DataBundle, n_updates: int = TrainConfig.ppo_updates,
              rollout_len: int = TrainConfig.ppo_rollout_len,
              horizon: str = "medium", adaptive: bool = True):
    env = build_env("PPO", bundle.px_tr, bundle.feats_tr, horizon=horizon, adaptive=adaptive,
                    max_steps=10_000, random_start=EnvConfig.random_start, seed=SEED,
                    macro=bundle.macro_tr, regime=bundle.regime_tr)
    agent = build_agent("PPO", env.state_dim, env.action_dim, seed=SEED)
    curve = []
    for rec in train_loop(agent, env, n_iters=n_updates, rollout_len=rollout_len):
        curve.append(dict(update=rec["update"], p_loss=rec["p_loss"], v_loss=rec["v_loss"],
                          ent=rec["ent"], kl=rec["kl"], mean_nav=rec["mean_nav"],
                          reward=rec["reward"], gain=rec["gain"],
                          success=int(rec["mean_nav"] > 1.0), steps=rollout_len))
        print(f"[PPO] upd {rec['update']:02d}  p_loss={rec['p_loss']:.3f} "
              f"v_loss={rec['v_loss']:.3f} ent={rec['ent']:.2f} kl={rec['kl']:.3f}")
    return agent, curve


# -------------------- SAC training --------------------
def train_sac(bundle: DataBundle, n_episodes: int = TrainConfig.sac_episodes,
              max_steps_per_episode: int = TrainConfig.sac_episode_len,
              horizon: str = "medium", adaptive: bool = True):
    env = build_env("SAC", bundle.px_tr, bundle.feats_tr, horizon=horizon, adaptive=adaptive,
                    max_steps=max_steps_per_episode,
                    random_start=EnvConfig.random_start, seed=SEED,
                    macro=bundle.macro_tr, regime=bundle.regime_tr)
    agent = build_agent("SAC", env.state_dim, env.action_dim, seed=SEED)
    curve = []
    for rec in train_loop(agent, env, n_iters=n_episodes):
        curve.append(dict(episode=rec["episode"], train_nav=rec["train_nav"], steps=rec["steps"],
                          reward=rec["reward"], gain=rec["gain"],
                          success=int(rec["nav"] > 1.0)))
        print(f"[SAC] ep {rec['episode']:02d}  NAV={rec['train_nav']:.3f}  buf={len(agent.buffer)}")
    return agent, curve


# -------------------- TD3 training --------------------
def train_td3(bundle: DataBundle, n_episodes: int = TrainConfig.td3_episodes,
              max_steps_per_episode: int = TrainConfig.td3_episode_len,
              horizon: str = "medium", adaptive: bool = True):
    # TD3 surekli-kontrol (hocanin tavsiyesi) — SAC ile ayni off-policy rejim.
    env = build_env("TD3", bundle.px_tr, bundle.feats_tr, horizon=horizon, adaptive=adaptive,
                    max_steps=max_steps_per_episode,
                    random_start=EnvConfig.random_start, seed=SEED,
                    macro=bundle.macro_tr, regime=bundle.regime_tr)
    agent = build_agent("TD3", env.state_dim, env.action_dim, seed=SEED)
    curve = []
    for rec in train_loop(agent, env, n_iters=n_episodes):
        curve.append({"episode": rec["episode"], "train_nav": rec["train_nav"], "steps": rec["steps"],
                      "reward": rec["reward"], "gain": rec["gain"],
                      "success": int(rec["nav"] > 1.0)})
        print(f"[TD3] ep {rec['episode']:02d}  NAV={rec['train_nav']:.3f}  buf={len(agent.buffer)}")
    return agent, curve


# -------------------- Evaluation --------------------
def evaluate(bundle: DataBundle, agent, algo: str, horizon: str = "medium", adaptive: bool = True):
    env = build_env(algo, bundle.px_te, bundle.feats_te, horizon=horizon, adaptive=adaptive,
                    max_steps=10_000, macro=bundle.macro_te, regime=bundle.regime_te)
    return rollout_evaluate(agent, env)


# -------------------- Main --------------------
def run():
    """Tam egitim + backtest akisi: seed -> veri -> 3 ajan -> eval -> CSV.
    main.py bunu DOGRUDAN cagirir (runpy yerine). Modul import'u yan etkisizdir (M1)."""
    np.random.seed(SEED)
    bundle = prepare_data()
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
    results = {}
    for name, agent in [("DQN", dqn_agent), ("PPO", ppo_agent),
                        ("SAC", sac_agent), ("TD3", td3_agent)]:
        bt = evaluate(bundle, agent, name)
        m = summary(bt["nav"], bt["rets"], bt["weights"])
        results[name] = dict(backtest=bt, metrics=m)
        print(f"[TEST] {name:<3}  CAGR={m['CAGR']:+.2%}  Sharpe={m['Sharpe']:+.2f}  "
              f"MaxDD={m['MaxDD']:+.2%}  Final={m['FinalNAV']:.3f}")

    print("-" * 60)
    bh = buy_and_hold_index(bundle.px_te)
    ew = equal_weight(bundle.px_te)
    mv = mean_variance(bundle.px_te, lookback=120, rebalance=20)
    for name, d in [("BuyHold", bh), ("EqualWeight", ew), ("MeanVar", mv)]:
        m = summary(d["nav"], d["rets"], d.get("weights"))
        results[name] = dict(backtest=d, metrics=m)
        print(f"[TEST] {name:<12}  CAGR={m['CAGR']:+.2%}  Sharpe={m['Sharpe']:+.2f}  "
              f"MaxDD={m['MaxDD']:+.2%}  Final={m['FinalNAV']:.3f}")

    min_len = min(len(v["backtest"]["nav"]) for v in results.values())
    dfn = pd.DataFrame({k: v["backtest"]["nav"][-min_len:] for k, v in results.items()})
    dfn.index = bundle.px_te.index[-min_len:]
    dfn.to_csv(RES / "navs_aligned.csv")

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
    for name, agent in [("DQN", dqn_agent), ("PPO", ppo_agent),
                        ("SAC", sac_agent), ("TD3", td3_agent)]:
        save_agent(agent, name, MODELS / f"{name}.pt", horizon="medium", adaptive=True)
    print("Modeller kaydedildi:", MODELS)

    for name in ["DQN", "PPO", "SAC", "TD3"]:
        W = results[name]["backtest"]["weights"]
        cols = list(bundle.px_te.columns) + ["CASH"]
        pd.DataFrame(W, columns=cols).to_csv(RES / f"weights_{name}.csv", index=False)

    print("=" * 60)
    print("DONE. Total wall time:", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    run()
