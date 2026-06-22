"""core.walkforward — fold yapisi + sizintisizlik (fold-yerel olcekleme) (Faz V5)."""
import numpy as np
import pandas as pd

from agents import PPOAgent
from core.walkforward import walk_forward
from utils.features import add_features


def _prices(seed=0, n=600, k=5):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-01", periods=n)
    return pd.DataFrame(100.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (n, k)), axis=0)),
                        index=idx, columns=[f"A{i}" for i in range(k)])


def _ppo_factory(state_dim, action_dim, seed):
    return PPOAgent(state_dim, action_dim, seed=seed)


def test_walk_forward_returns_fold_metrics():
    px = _prices()
    feats_raw = add_features(px)                       # teknik feat (forecast YOK -> leak-safe)
    rep = walk_forward(px, feats_raw, _ppo_factory, n_folds=2,
                       n_iters=2, rollout_len=120, seed=0, step_days=1)
    assert len(rep["folds"]) >= 1
    for key in ("CAGR", "Sharpe", "MaxDD", "FinalNAV"):
        assert key in rep["mean"] and key in rep["std"]
        assert np.isfinite(rep["mean"][key])
