"""Ajan-agnostik degerlendirme/rollout dongusu (Faz 3, H2).

Onceki train.py:evaluate ve app.py:evaluate_with_trace'in `if algo == 'DQN' ...`
dallanmasini ortadan kaldirir: BaseAgent.act_eval(state) kullanilir. Env zaten
hem ayrik (DiscretePortfolioEnv: idx) hem surekli (PortfolioEnv: vektor)
aksiyonu kabul eder, dolayisiyla dongu ajan tipinden bagimsizdir.
"""
from __future__ import annotations

import numpy as np

from core.contracts import BacktestResult


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
    nav = np.array(env.nav_history)
    rets = np.concatenate([[0.0], np.array(env.ret_history)])
    initial_w = np.asarray(env.weight_history[0])
    result = BacktestResult(
        nav=nav,
        rets=rets,
        dates=[env.episode_start_date] + list(env.date_history),
        weights_before=np.vstack([initial_w, np.asarray(env.weights_before_history)]),
        target_weights=np.vstack([initial_w, np.asarray(env.target_weight_history)]),
        weights_after=np.asarray(env.weight_history),
        turnover=np.concatenate([[0.0], np.asarray(env.turnover_history, dtype=float)]),
        reward_terms_history=env.reward_terms_history,
        period_lengths=np.concatenate([[0], np.asarray(env.period_length_history, dtype=np.int32)]),
        discounts=np.concatenate([[1.0], np.asarray(env.discount_history, dtype=float)]),
        provenance=env.provenance,
    )
    return result.to_dict()
