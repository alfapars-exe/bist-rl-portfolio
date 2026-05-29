"""Generator-tabanli egitim cekirdegi (Faz 3, H1).

train.py (CLI, toplu) ve app.py (UI, canli/artimli) ARTIK ayni egitim dongusunu
paylasir. Her algoritmanin dongusu kendi generator'inda kalir:
  - DQN / SAC : off-policy, adim-bazli (act -> step -> remember -> train_step)
  - PPO       : on-policy, rollout-bazli (rollout_len adim topla -> train)
Her iterasyonda ORTAK bir telemetri dict'i yield edilir. CLI generator'i n_iters
ile sinirli tuketir; UI n_iters=None ile sonsuz akisi 'Durdur' ile keser.

Dispatch (train) ajan tipine gore dogru generator'i secen TEK noktadir.

DAVRANIS KORUNUR: her dongunun RNG tuketen cagri sirasi (agent.act,
agent.train_step / agent.train, np.random warmup, env.step) onceki train.py
donguleriyle birebir aynidir. Eklenen alanlar (loss/actions/success/agent/env)
gozlemdir, RNG tuketmez. Golden-master <=1e-6 bunu dogrular.
"""
from __future__ import annotations

import itertools
from typing import Iterator, Optional

import numpy as np

from agents import DQNAgent, PPOAgent, SACAgent
from utils.metrics import success_vs_benchmark


def _counter(n: Optional[int]):
    """n is None ise sonsuz (UI), degilse range(n) (CLI)."""
    return itertools.count() if n is None else range(n)


# --------------------------------------------------------------------- DQN
def train_dqn(agent, env, n_episodes: Optional[int] = None,
              ew_nav: Optional[np.ndarray] = None) -> Iterator[dict]:
    for ep in _counter(n_episodes):
        s, _ = env.reset()
        done = trunc = False
        ep_reward = 0.0
        losses, actions = [], []
        while not (done or trunc):
            a = agent.act(s)
            actions.append(int(a))
            s2, r, done, trunc, _ = env.step(a)
            agent.remember(s, a, r, s2, float(done))
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            s = s2
            ep_reward += r
        nav_agent = np.array(env.nav_history[1:])
        success = (success_vs_benchmark(nav_agent, ew_nav[:len(nav_agent)])
                   if ew_nav is not None else 0)
        yield {
            "algo": "DQN", "iter": ep, "episode": ep,
            "reward": float(ep_reward),
            "nav": float(env.nav), "train_nav": float(env.nav),
            "gain": float(env.nav) - 1.0,
            "eps": agent.eps(),
            "loss": float(np.mean(losses)) if losses else 0.0,
            "success": int(success),
            "actions": actions,
            "agent": agent, "env": env,
        }


# --------------------------------------------------------------------- PPO
def train_ppo(agent, env, n_updates: Optional[int] = None,
              rollout_len: int = 400,
              ew_nav: Optional[np.ndarray] = None) -> Iterator[dict]:
    import torch
    s, _ = env.reset()
    for upd in _counter(n_updates):
        ep_navs = []
        for _ in range(rollout_len):
            a, lp, v = agent.act(s)
            s2, r, done, trunc, _ = env.step(a)
            agent.remember(s, a, r, done or trunc, v, lp)
            s = s2
            if done or trunc:
                ep_navs.append(float(env.nav))
                s, _ = env.reset()
        with torch.no_grad():
            last_v = float(agent.value(
                torch.as_tensor(s, dtype=torch.float32,
                                device=agent.device).unsqueeze(0)
            ).item())
        info = agent.train(last_v)
        mean_nav = float(np.mean(ep_navs)) if ep_navs else float(env.nav)
        success = (int(mean_nav >= float(ew_nav[min(len(ew_nav) - 1, env.step_count)]))
                   if ew_nav is not None else 0)
        yield {
            "algo": "PPO", "iter": upd, "update": upd,
            "p_loss": float(info["p_loss"]), "v_loss": float(info["v_loss"]),
            "ent": float(info["ent"]), "kl": float(info["kl"]),
            "mean_nav": mean_nav, "nav": mean_nav,
            "reward": -float(info["p_loss"]),
            "gain": mean_nav - 1.0,
            "loss": float(info["v_loss"]),
            "success": int(success),
            "agent": agent, "env": env,
        }


# --------------------------------------------------------------------- SAC
def train_sac(agent, env, n_episodes: Optional[int] = None,
              warmup: int = 500, train_every: int = 4,
              ew_nav: Optional[np.ndarray] = None) -> Iterator[dict]:
    for ep in _counter(n_episodes):
        s, _ = env.reset()
        done = trunc = False
        step = 0
        losses = []
        while not (done or trunc):
            if len(agent.buffer) < warmup:
                a = np.random.randn(env.action_dim).astype(np.float32) * 0.5
            else:
                a = agent.act(s)
            s2, r, done, trunc, _ = env.step(a)
            agent.remember(s, a, r, s2, float(done))
            if len(agent.buffer) > warmup and step % train_every == 0:
                loss = agent.train_step()
                if loss is not None:
                    losses.append(loss)
            s = s2
            step += 1
        nav_agent = np.array(env.nav_history[1:])
        success = (success_vs_benchmark(nav_agent, ew_nav[:len(nav_agent)])
                   if ew_nav is not None else 0)
        yield {
            "algo": "SAC", "iter": ep, "episode": ep,
            "reward": float(np.sum(env.ret_history)),
            "nav": float(env.nav), "train_nav": float(env.nav),
            "gain": float(env.nav) - 1.0,
            "steps": step,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "success": int(success),
            "agent": agent, "env": env,
        }


# --------------------------------------------------------------------- dispatch
def train(agent, env, *, n_iters: Optional[int] = None,
          rollout_len: int = 400,
          ew_nav: Optional[np.ndarray] = None) -> Iterator[dict]:
    """Ajan tipine gore dogru egitim generator'ini secen TEK dispatch noktasi.

    (Strategy pattern: dallanma burada bir kez yapilir, tuketicilerde degil.)
    """
    if isinstance(agent, DQNAgent):
        return train_dqn(agent, env, n_episodes=n_iters, ew_nav=ew_nav)
    if isinstance(agent, PPOAgent):
        return train_ppo(agent, env, n_updates=n_iters, rollout_len=rollout_len, ew_nav=ew_nav)
    if isinstance(agent, SACAgent):
        return train_sac(agent, env, n_episodes=n_iters, ew_nav=ew_nav)
    raise TypeError(f"Bilinmeyen ajan tipi: {type(agent).__name__}")
