"""Walk-forward dogrulama (Faz V5) — backtest overfitting'e karsi.

Alanin 1 numarali riski backtest overfitting / zayif genelleme (Liu 2022
arXiv:2209.05559; Velay 2023 arXiv:2306.10950). Bu modul train donemini
genisleyen pencerelere boler; her fold'da train-sub uzerinde egitip val-sub
uzerinde degerlendirir. Fold'lar arasi metrik stabilitesi = genelleme gostergesi.

SIZINTISIZLIK: her fold KENDI train-sub'inda TrainScaler ile olceklenir (val-sub
ayni istatistiklerle DONUSTURULUR, orada fit EDILMEZ). Caller forecast feature'i
DAHIL ETMEMELI (full-train forecaster fold val'ini gormus olur); teknik feature
ile cagrilmali (main.py boyle yapar).
"""
from __future__ import annotations

import numpy as np

from core.rollout import evaluate
from core.trainer import train as train_loop
from env.portfolio_env import DiscretePortfolioEnv, PortfolioEnv
from utils.features import TrainScaler
from utils.metrics import summary


def _slice(feats_raw, idx):
    return {k: v.loc[idx] for k, v in feats_raw.items()}


def walk_forward(prices, feats_raw, agent_factory, *, discrete: bool = False,
                 n_folds: int = 3, val_frac: float = 0.2, purge: int = 5,
                 n_iters: int = 10, rollout_len: int = 400, seed: int = 42,
                 horizon: str = "short", adaptive: bool = True) -> dict:
    """Genisleyen-pencere walk-forward.

    agent_factory(state_dim, action_dim, seed) -> ajan. Doner:
    {"folds": [metrik dict...], "mean": {...}, "std": {...}}.
    """
    T = len(prices)
    val_len = max(1, int(T * val_frac / n_folds))
    Cls = DiscretePortfolioEnv if discrete else PortfolioEnv
    fold_metrics = []
    for i in range(n_folds):
        val_end = T - (n_folds - 1 - i) * val_len
        val_start = val_end - val_len
        tr_end = val_start - purge
        if tr_end <= 80:                       # yeterli train yoksa fold'u atla
            continue
        tr_idx = prices.index[:tr_end]
        va_idx = prices.index[val_start:val_end]

        sc = TrainScaler().fit(_slice(feats_raw, tr_idx))     # fold-yerel (sizintisiz)
        f_tr = sc.transform(_slice(feats_raw, tr_idx))
        f_va = sc.transform(_slice(feats_raw, va_idx))

        tr_env = Cls(prices.loc[tr_idx], f_tr, horizon=horizon, adaptive=adaptive,
                     max_steps=252, random_start=True, seed=seed)
        action_dim = tr_env.n_discrete if discrete else tr_env.action_dim
        agent = agent_factory(tr_env.state_dim, action_dim, seed)
        for _ in train_loop(agent, tr_env, n_iters=n_iters, rollout_len=rollout_len):
            pass

        va_env = Cls(prices.loc[va_idx], f_va, horizon=horizon, adaptive=adaptive,
                     max_steps=10_000, random_start=False, seed=seed)
        bt = evaluate(agent, va_env)
        if len(bt["nav"]) > 0:
            fold_metrics.append(summary(bt["nav"], bt["rets"], bt["weights"]))

    keys = list(fold_metrics[0].keys()) if fold_metrics else []
    mean = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std = {k: float(np.std([m[k] for m in fold_metrics])) for k in keys}
    return {"folds": fold_metrics, "mean": mean, "std": std}
