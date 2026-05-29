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
from config import SEED, DQNConfig, PPOConfig, SACConfig

BASE = Path(__file__).resolve().parent
RES  = BASE / "results"
RES.mkdir(exist_ok=True)

np.random.seed(SEED)

# -------------------- load data --------------------
px = download_bist()
feats_all_raw = add_features(px)
px_tr, px_te = train_test_split(px)
feats_tr_raw = {k: v.loc[px_tr.index] for k, v in feats_all_raw.items()}
feats_te_raw = {k: v.loc[px_te.index] for k, v in feats_all_raw.items()}

scaler = TrainScaler().fit(feats_tr_raw)
feats_tr = scaler.transform(feats_tr_raw)
feats_te = scaler.transform(feats_te_raw)

print(f"Train: {px_tr.shape}, Test: {px_te.shape}, tickers: {px.shape[1]}")


def make_env(px_, feats_, discrete: bool, horizon: str = "medium",
             adaptive: bool = True, max_steps: int = 10_000):
    cls = DiscretePortfolioEnv if discrete else PortfolioEnv
    return cls(
        px_, feats_,
        horizon=horizon, adaptive=adaptive,
        max_steps=max_steps,
    )


# -------------------- DQN training --------------------
def train_dqn(n_episodes: int = 6, horizon: str = "medium", adaptive: bool = True):
    env = make_env(px_tr, feats_tr, discrete=True, horizon=horizon,
                   adaptive=adaptive, max_steps=252)
    agent = DQNAgent(
        env.state_dim, env.n_discrete,
        hidden=DQNConfig.hidden, lr=DQNConfig.lr, eps_decay=DQNConfig.eps_decay,
        batch_size=DQNConfig.batch_size, target_update=DQNConfig.target_update, seed=SEED,
    )
    curve = []
    for ep in range(n_episodes):
        s, _ = env.reset()
        done = trunc = False
        ep_reward = 0.0
        while not (done or trunc):
            a = agent.act(s)
            s2, r, done, trunc, _ = env.step(a)
            agent.remember(s, a, r, s2, float(done))
            agent.train_step()
            s = s2; ep_reward += r
        curve.append(dict(episode=ep, reward=ep_reward,
                          train_nav=float(env.nav), eps=agent.eps()))
        print(f"[DQN] ep {ep:02d}  ret={ep_reward:+.3f}  NAV={env.nav:.3f}  eps={agent.eps():.3f}")
    return agent, curve


# -------------------- PPO training --------------------
def train_ppo(n_updates: int = 18, rollout_len: int = 400,
              horizon: str = "medium", adaptive: bool = True):
    env = make_env(px_tr, feats_tr, discrete=False, horizon=horizon,
                   adaptive=adaptive, max_steps=10_000)
    agent = PPOAgent(env.state_dim, env.action_dim, hidden=PPOConfig.hidden,
                     lr_p=PPOConfig.lr_p, lr_v=PPOConfig.lr_v, batch_size=PPOConfig.batch_size,
                     n_epochs=PPOConfig.n_epochs, seed=SEED)
    curve = []
    s, _ = env.reset()
    for upd in range(n_updates):
        ep_navs = []
        for _ in range(rollout_len):
            a, lp, v = agent.act(s)
            s2, r, done, trunc, _ = env.step(a)
            agent.remember(s, a, r, done or trunc, v, lp)
            s = s2
            if done or trunc:
                ep_navs.append(float(env.nav))
                s, _ = env.reset()
        import torch
        with torch.no_grad():
            last_v = float(agent.value(
                torch.as_tensor(s, dtype=torch.float32,
                                device=agent.device).unsqueeze(0)
            ).item())
        info = agent.train(last_v)
        mean_nav = float(np.mean(ep_navs)) if ep_navs else float(env.nav)
        curve.append(dict(update=upd, **info, mean_nav=mean_nav))
        print(f"[PPO] upd {upd:02d}  p_loss={info['p_loss']:.3f} "
              f"v_loss={info['v_loss']:.3f} ent={info['ent']:.2f} kl={info['kl']:.3f}")
    return agent, curve


# -------------------- SAC training --------------------
def train_sac(n_episodes: int = 3, max_steps_per_episode: int = 1200,
              horizon: str = "medium", adaptive: bool = True):
    env = make_env(px_tr, feats_tr, discrete=False, horizon=horizon,
                   adaptive=adaptive, max_steps=max_steps_per_episode)
    agent = SACAgent(env.state_dim, env.action_dim, hidden=SACConfig.hidden,
                     lr_pi=SACConfig.lr_pi, lr_q=SACConfig.lr_q, alpha=SACConfig.alpha,
                     seed=SEED, batch_size=SACConfig.batch_size)
    curve = []
    for ep in range(n_episodes):
        s, _ = env.reset()
        done = trunc = False; step = 0
        while not (done or trunc):
            if len(agent.buffer) < 500:
                a = np.random.randn(env.action_dim).astype(np.float32) * 0.5
            else:
                a = agent.act(s)
            s2, r, done, trunc, _ = env.step(a)
            agent.remember(s, a, r, s2, float(done))
            if len(agent.buffer) > 500 and step % 4 == 0:
                agent.train_step()
            s = s2; step += 1
        curve.append(dict(episode=ep, train_nav=float(env.nav), steps=step))
        print(f"[SAC] ep {ep:02d}  NAV={env.nav:.3f}  buf={len(agent.buffer)}")
    return agent, curve


# -------------------- Evaluation --------------------
def evaluate(agent, algo: str, horizon: str = "medium", adaptive: bool = True):
    env = make_env(px_te, feats_te, discrete=(algo == "DQN"),
                   horizon=horizon, adaptive=adaptive, max_steps=10_000)
    s, _ = env.reset()
    done = trunc = False
    while not (done or trunc):
        if algo == "DQN":
            a = agent.act(s, greedy=True)
        elif algo == "PPO":
            a, _, _ = agent.act(s)
        else:  # SAC
            a = agent.act(s, deterministic=True)
        s, r, done, trunc, _ = env.step(a)
    nav = np.array(env.nav_history[1:])
    rets = np.array(env.ret_history)
    W = np.array(env.weight_history[1:])
    offset = env.window
    dates = list(env.dates[offset: offset + len(nav)])
    return dict(nav=nav, rets=rets, weights=W, dates=dates,
                reward_terms_history=env.reward_terms_history)


# -------------------- Main --------------------
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 60)
    dqn_agent, dqn_curve = train_dqn(n_episodes=6)
    print("DQN total time:", round(time.time() - t0, 1), "s")

    t1 = time.time()
    ppo_agent, ppo_curve = train_ppo(n_updates=18, rollout_len=400)
    print("PPO total time:", round(time.time() - t1, 1), "s")

    t2 = time.time()
    sac_agent, sac_curve = train_sac(n_episodes=3, max_steps_per_episode=1200)
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
