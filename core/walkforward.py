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

from collections import deque

import numpy as np

from config import HORIZON_PRESETS
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
                 horizon: str = "short", adaptive: bool = True,
                 macro=None, regime=None) -> dict:
    """Genisleyen-pencere walk-forward.

    agent_factory(state_dim, action_dim, seed) -> ajan. Doner:
    {"folds": [metrik dict...], "mean": {...}, "std": {...}}.

    macro: (T, F_macro) numpy dizisi (tam veri uzunlugu, fold icinde dilimlenir).
           None -> makrosuz ortam (V5 davranisi).
    regime: (T,) numpy dizisi (tam veri uzunlugu, fold icinde dilimlenir).
            None -> rejim amplifikasyonu kapali.
    NOT: forecast feature WF'de DAHIL EDILMEZ (sızıntı-güvenli mevcut karar KORUNUR).
    """
    T = len(prices)
    val_len = max(1, int(T * val_frac / n_folds))
    env_cls = DiscretePortfolioEnv if discrete else PortfolioEnv
    # Egitim env'i max_steps + random_start ile kurulur; env reset'i gecerli ve CESITLI bir
    # rastgele-baslangic araligi icin lo < tr_end - max_steps - 1 ister, aksi halde sessizce
    # sabit-baslangica duser (bkz. PortfolioEnv._reset_state) ve fold tek-pencereye dejenere
    # olur. Bu yuzden fold-atlama esigini sabit 80 yerine env'in episode-pencere gereksinimine
    # baglariz (lo, env ile ayni: max(window=20, minvol_window, 21)).
    train_max_steps = 252
    lo = max(20, HORIZON_PRESETS[horizon]["minvol_window"], 21)
    min_train = lo + train_max_steps + 8       # +8: dejenere olmayan baslangic cesitliligi
    fold_metrics = []
    for i in range(n_folds):
        val_end = T - (n_folds - 1 - i) * val_len
        val_start = val_end - val_len
        tr_end = val_start - purge
        if tr_end < min_train:                 # random_start icin yeterli/cesitli train yok -> atla
            continue
        tr_idx = prices.index[:tr_end]
        va_idx = prices.index[val_start:val_end]

        sc = TrainScaler().fit(_slice(feats_raw, tr_idx))     # fold-yerel (sizintisiz)
        f_tr = sc.transform(_slice(feats_raw, tr_idx))
        f_va = sc.transform(_slice(feats_raw, va_idx))

        # T6 (C6): macro/regime fold dilimleri (None gecilirse None kalir -> makrosuz).
        macro_tr = macro[:tr_end] if macro is not None else None
        macro_va = macro[val_start:val_end] if macro is not None else None
        regime_tr = regime[:tr_end] if regime is not None else None
        regime_va = regime[val_start:val_end] if regime is not None else None

        tr_env = env_cls(prices.loc[tr_idx], f_tr, horizon=horizon, adaptive=adaptive,
                         max_steps=train_max_steps, random_start=True, seed=seed,
                         macro=macro_tr, regime=regime_tr)
        action_dim = tr_env.n_discrete if discrete else tr_env.action_dim
        agent = agent_factory(tr_env.state_dim, action_dim, seed)
        # generator'i sonuna kadar tuket (egitim yan-etkili; ciktiya gerek yok)
        deque(train_loop(agent, tr_env, n_iters=n_iters, rollout_len=rollout_len), maxlen=0)

        va_env = env_cls(prices.loc[va_idx], f_va, horizon=horizon, adaptive=adaptive,
                         max_steps=10_000, random_start=False, seed=seed,
                         macro=macro_va, regime=regime_va)
        bt = evaluate(agent, va_env)
        if len(bt["nav"]) > 0:
            fold_metrics.append(summary(bt["nav"], bt["rets"], bt["weights"]))

    keys = list(fold_metrics[0].keys()) if fold_metrics else []
    mean = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std = {k: float(np.std([m[k] for m in fold_metrics])) for k in keys}
    return {"folds": fold_metrics, "mean": mean, "std": std}
