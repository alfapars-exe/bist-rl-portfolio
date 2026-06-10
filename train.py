"""Main training + backtest driver — yeni paket yapısı + horizon + adaptive reward.

Trains DQN (discrete, 6 templates), PPO (continuous), SAC (continuous) on BIST 28
2015-2021 ve 2022-2024 test setinde backtest eder. Tüm ajanlar PyTorch'tadır ve
özellikler `utils.features.TrainScaler` ile train-only z-score standardize edilir.
"""
from __future__ import annotations

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path
import numpy as np
import pandas as pd

from data import download_bist, train_test_split
from utils.features import add_features, TrainScaler
from utils.metrics import summary
from utils.baselines import equal_weight, mean_variance, buy_and_hold_index
from env.portfolio_env import PortfolioEnv, DiscretePortfolioEnv
from agents import DQNAgent, PPOAgent, SACAgent
from config import SEED, DQNConfig, PPOConfig, SACConfig, TrainConfig, EnvConfig, ForecastConfig
from core.features import select_features
from core.rollout import evaluate as rollout_evaluate
from core.trainer import train as train_loop

BASE = Path(__file__).resolve().parent
RES  = BASE / "results"
RES.mkdir(exist_ok=True)

# -------------------- veri (M1: import yan etkisi yok; run()/prepare_data yukler) --------------------
px = px_tr = px_te = feats_tr = feats_te = scaler = None


def prepare_data():
    """BIST verisini yukler, train/test ayirir, train-only z-score uygular;
    modul globallerini doldurur (train_*/evaluate bunlari kullanir)."""
    global px, px_tr, px_te, feats_tr, feats_te, scaler
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


def make_env(px_, feats_, discrete: bool, horizon: str = "medium",
             adaptive: bool = True, max_steps: int = 10_000,
             random_start: bool = False, seed: int = SEED):
    cls = DiscretePortfolioEnv if discrete else PortfolioEnv
    return cls(
        px_, feats_,
        horizon=horizon, adaptive=adaptive,
        max_steps=max_steps, random_start=random_start, seed=seed,
    )


def _feats_for(feats: dict, algo: str) -> dict:
    """Shim — SOLID P2: tek dogruluk kaynagi core.features.select_features.
    (test_env bu adi cagirir; geriye-uyumluluk icin korunur.)"""
    return select_features(feats, algo)


# -------------------- DQN training --------------------
def train_dqn(n_episodes: int = TrainConfig.dqn_episodes, horizon: str = "medium", adaptive: bool = True):
    env = make_env(px_tr, _feats_for(feats_tr, "DQN"), discrete=True, horizon=horizon,
                   adaptive=adaptive, max_steps=252,
                   random_start=EnvConfig.random_start, seed=SEED)
    agent = DQNAgent(
        env.state_dim, env.n_discrete,
        hidden=DQNConfig.hidden, lr=DQNConfig.lr, eps_decay=DQNConfig.eps_decay,
        batch_size=DQNConfig.batch_size, target_update=DQNConfig.target_update, seed=SEED,
    )
    curve = []
    for rec in train_loop(agent, env, n_iters=n_episodes):
        curve.append(dict(episode=rec["episode"], reward=rec["reward"],
                          train_nav=rec["train_nav"], eps=rec["eps"]))
        print(f"[DQN] ep {rec['episode']:02d}  ret={rec['reward']:+.3f}  "
              f"NAV={rec['train_nav']:.3f}  eps={rec['eps']:.3f}")
    return agent, curve


# -------------------- PPO training --------------------
def train_ppo(n_updates: int = TrainConfig.ppo_updates, rollout_len: int = TrainConfig.ppo_rollout_len,
              horizon: str = "medium", adaptive: bool = True):
    env = make_env(px_tr, _feats_for(feats_tr, "PPO"), discrete=False, horizon=horizon,
                   adaptive=adaptive, max_steps=10_000,
                   random_start=EnvConfig.random_start, seed=SEED)
    agent = PPOAgent(env.state_dim, env.action_dim, hidden=PPOConfig.hidden,
                     lr_p=PPOConfig.lr_p, lr_v=PPOConfig.lr_v, batch_size=PPOConfig.batch_size,
                     n_epochs=PPOConfig.n_epochs, seed=SEED)
    curve = []
    for rec in train_loop(agent, env, n_iters=n_updates, rollout_len=rollout_len):
        curve.append(dict(update=rec["update"], p_loss=rec["p_loss"], v_loss=rec["v_loss"],
                          ent=rec["ent"], kl=rec["kl"], mean_nav=rec["mean_nav"]))
        print(f"[PPO] upd {rec['update']:02d}  p_loss={rec['p_loss']:.3f} "
              f"v_loss={rec['v_loss']:.3f} ent={rec['ent']:.2f} kl={rec['kl']:.3f}")
    return agent, curve


# -------------------- SAC training --------------------
def train_sac(n_episodes: int = TrainConfig.sac_episodes, max_steps_per_episode: int = TrainConfig.sac_episode_len,
              horizon: str = "medium", adaptive: bool = True):
    env = make_env(px_tr, _feats_for(feats_tr, "SAC"), discrete=False, horizon=horizon,
                   adaptive=adaptive, max_steps=max_steps_per_episode,
                   random_start=EnvConfig.random_start, seed=SEED)
    agent = SACAgent(env.state_dim, env.action_dim, hidden=SACConfig.hidden,
                     lr_pi=SACConfig.lr_pi, lr_q=SACConfig.lr_q, alpha=SACConfig.alpha,
                     seed=SEED, batch_size=SACConfig.batch_size)
    curve = []
    for rec in train_loop(agent, env, n_iters=n_episodes):
        curve.append(dict(episode=rec["episode"], train_nav=rec["train_nav"], steps=rec["steps"]))
        print(f"[SAC] ep {rec['episode']:02d}  NAV={rec['train_nav']:.3f}  buf={len(agent.buffer)}")
    return agent, curve


# -------------------- Evaluation --------------------
def evaluate(agent, algo: str, horizon: str = "medium", adaptive: bool = True):
    env = make_env(px_te, _feats_for(feats_te, algo), discrete=(algo == "DQN"),
                   horizon=horizon, adaptive=adaptive, max_steps=10_000)
    return rollout_evaluate(agent, env)


# -------------------- Main --------------------
def run():
    """Tam egitim + backtest akisi: seed -> veri -> 3 ajan -> eval -> CSV.
    main.py bunu DOGRUDAN cagirir (runpy yerine). Modul import'u yan etkisizdir (M1)."""
    np.random.seed(SEED)
    prepare_data()
    t0 = time.time()
    print("=" * 60)
    dqn_agent, dqn_curve = train_dqn()
    print("DQN total time:", round(time.time() - t0, 1), "s")

    t1 = time.time()
    ppo_agent, ppo_curve = train_ppo()
    print("PPO total time:", round(time.time() - t1, 1), "s")

    t2 = time.time()
    sac_agent, sac_curve = train_sac()
    print("SAC total time:", round(time.time() - t2, 1), "s")

    print("=" * 60)
    results = {}
    for name, agent in [("DQN", dqn_agent), ("PPO", ppo_agent), ("SAC", sac_agent)]:
        bt = evaluate(agent, name)
        m = summary(bt["nav"], bt["rets"], bt["weights"])
        results[name] = dict(backtest=bt, metrics=m)
        print(f"[TEST] {name:<3}  CAGR={m['CAGR']:+.2%}  Sharpe={m['Sharpe']:+.2f}  "
              f"MaxDD={m['MaxDD']:+.2%}  Final={m['FinalNAV']:.3f}")

    print("-" * 60)
    bh = buy_and_hold_index(px_te)
    ew = equal_weight(px_te)
    mv = mean_variance(px_te, lookback=120, rebalance=20)
    for name, d in [("BuyHold", bh), ("EqualWeight", ew), ("MeanVar", mv)]:
        m = summary(d["nav"], d["rets"], d.get("weights"))
        results[name] = dict(backtest=d, metrics=m)
        print(f"[TEST] {name:<12}  CAGR={m['CAGR']:+.2%}  Sharpe={m['Sharpe']:+.2f}  "
              f"MaxDD={m['MaxDD']:+.2%}  Final={m['FinalNAV']:.3f}")

    min_len = min(len(v["backtest"]["nav"]) for v in results.values())
    dfn = pd.DataFrame({k: v["backtest"]["nav"][-min_len:] for k, v in results.items()})
    dfn.index = px_te.index[-min_len:]
    dfn.to_csv(RES / "navs_aligned.csv")

    met_df = pd.DataFrame({k: v["metrics"] for k, v in results.items()}).T
    met_df.to_csv(RES / "metrics.csv")
    print(met_df.round(4))

    pd.DataFrame(dqn_curve).to_csv(RES / "dqn_curve.csv", index=False)
    pd.DataFrame(ppo_curve).to_csv(RES / "ppo_curve.csv", index=False)
    pd.DataFrame(sac_curve).to_csv(RES / "sac_curve.csv", index=False)

    for name in ["DQN", "PPO", "SAC"]:
        W = results[name]["backtest"]["weights"]
        cols = list(px_te.columns) + ["CASH"]
        pd.DataFrame(W, columns=cols).to_csv(RES / f"weights_{name}.csv", index=False)

    print("=" * 60)
    print("DONE. Total wall time:", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    run()
