"""Ajan-agnostik degerlendirme/rollout dongusu (Faz 3, H2).

Onceki train.py:evaluate ve app.py:evaluate_with_trace'in `if algo == 'DQN' ...`
dallanmasini ortadan kaldirir: BaseAgent.act_eval(state) kullanilir. Env zaten
hem ayrik (DiscretePortfolioEnv: idx) hem surekli (PortfolioEnv: vektor)
aksiyonu kabul eder, dolayisiyla dongu ajan tipinden bagimsizdir.
"""
from __future__ import annotations

import numpy as np


def evaluate(agent, env) -> dict:
    """Egitilmis ajani env uzerinde bir kez kosturur; backtest cikti dict'i dondurur.

    Onceki train.py:evaluate ile birebir ayni RNG tuketimi: act_eval
    (DQN greedy / PPO ornek / SAC deterministik) -> env.step.
    """
    s, _ = env.reset()
    done = trunc = False
    while not (done or trunc):
        a = agent.act_eval(s)
        s, r, done, trunc, _ = env.step(a)
    nav = np.array(env.nav_history[1:])
    rets = np.array(env.ret_history)
    W = np.array(env.weight_history[1:])
    offset = env.window
    dates = list(env.dates[offset: offset + len(nav)])
    return dict(nav=nav, rets=rets, weights=W, dates=dates,
                reward_terms_history=env.reward_terms_history)
